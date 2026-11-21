// Qwen3-VL text path, layers 0..49, on the NPU (one attention + one MLP kernel per layer).
#include <cmath>

#include "common.hpp"
#include "pipeline.hpp"

namespace axd {

namespace {
constexpr int kPage = 256;
constexpr int kKvHeads = 8;
constexpr double kRopeTheta = 5000000.0;
}  // namespace

TextEncoder::TextEncoder(const ModelDir &dir, const TextWeights &w) : w_(w) {
    vocab_ = std::make_unique<Model>(dir.axmodel("text-vocab-embedding"));
    attn_ = std::make_unique<Model>(dir.axmodel("text-attention"));
    mlp_ = std::make_unique<Model>(dir.axmodel("text-mlp"));
    for (int i = 0; i < 2; ++i) hidden_[i] = DeviceBuffer(size_t(kPage) * kTextDim * 2, "text-hidden");
    ids_ = DeviceBuffer(kMaxTextTokens * 4, "text-ids");
    count_ = DeviceBuffer(4, "text-count");
    positions_ = DeviceBuffer(3 * kMaxTextTokens * 4, "text-positions");
    cos_ = DeviceBuffer(kMaxTextTokens * 64 * 4, "text-cos");
    sin_ = DeviceBuffer(kMaxTextTokens * 64 * 4, "text-sin");
    paged_ = DeviceBuffer(size_t(2) * kPage * kKvHeads * kHeadDim * 2, "text-paged-kv");
    page_out_ = DeviceBuffer(size_t(2) * kPage * kKvHeads * kHeadDim * 2, "text-page-out");
    indptr_ = DeviceBuffer(2 * 4, "text-indptr");
    indices_ = DeviceBuffer(4, "text-indices");
    cache_loc_ = DeviceBuffer(2 * 4, "text-cache-location");
    valid_ = DeviceBuffer(4, "text-valid-tokens");
    last_page_len_ = DeviceBuffer(32 * 4, "text-last-page-len");
    for (auto *b : {&indptr_, &indices_, &cache_loc_, &last_page_len_, &paged_, &page_out_}) b->buf().fill_zero();

    // RoPE tables [256,64] float32 (rotate-half layout, theta 5e6)
    std::vector<float> c(kMaxTextTokens * 64), s(kMaxTextTokens * 64);
    for (int p = 0; p < kMaxTextTokens; ++p)
        for (int i = 0; i < 64; ++i) {
            float inv = 1.0f / float(std::pow(kRopeTheta, double(2 * i) / kHeadDim));
            float a = float(p) * inv;
            c[p * 64 + i] = std::cos(a);
            s[p * 64 + i] = std::sin(a);
        }
    cos_.buf().upload(c.data(), c.size() * 4);
    sin_.buf().upload(s.data(), s.size() * 4);
}

std::vector<uint16_t> TextEncoder::encode(const std::vector<int32_t> &ids) {
    const int n = int(ids.size());
    if (n <= 0 || n > kMaxTextTokens) fail("text length must be 1..256 tokens");
    std::vector<int32_t> padded(kMaxTextTokens, ids.back());
    std::copy(ids.begin(), ids.end(), padded.begin());
    ids_.buf().upload(padded.data(), padded.size() * 4);
    count_.buf().upload(&n, 4);
    valid_.buf().upload(&n, 4);
    std::vector<int32_t> pos(3 * kMaxTextTokens, 0);
    for (int a = 0; a < 3; ++a)
        for (int i = 0; i < n; ++i) pos[a * kMaxTextTokens + i] = i;
    positions_.buf().upload(pos.data(), pos.size() * 4);

    vocab_->run({{"input_flat", ids_}, {"weight", w_.embed}, {"num_tokens", count_}}, {{"output_flat", hidden_[0]}});
    Buf x = hidden_[0], out = hidden_[1];
    for (auto &layer : w_.layers) {
        Bindings a = {{"x", x}, {"cos", cos_}, {"sin", sin_}, {"position_ids", positions_}, {"paged_kv_cache", paged_},
                      {"indptr", indptr_}, {"indices", indices_}, {"cache_location", cache_loc_},
                      {"valid_tokens", valid_}, {"last_page_len", last_page_len_}};
        for (const char *k : {"x_norm_w", "w_q", "w_scale_q", "w_scale2_q", "w_zeropoint_q", "w_k", "w_scale_k",
                              "w_scale2_k", "w_zeropoint_k", "w_v", "w_scale_v", "w_scale2_v", "w_zeropoint_v",
                              "q_norm_w", "k_norm_w", "w_o", "w_scale_o", "w_scale2_o", "w_zeropoint_o"})
            a[k] = layer->get(k);
        attn_->run(a, {{"output", out}, {"kv_page_out", page_out_}});
        Bindings m = {{"x", out}};
        for (const char *k : {"w_norm", "w1", "w_scale1", "w_scale2_1", "w2", "w_scale2", "w_scale2_2", "w3", "w_scale3",
                              "w_scale2_3", "w_zp_scale1", "w_zp_scale2", "w_zp_scale3", "input_permutation"})
            m[k] = layer->get(k);
        mlp_->run(m, {{"output", x}});
    }
    std::vector<uint16_t> hidden(size_t(n) * kTextDim);
    x.download(hidden.data(), hidden.size() * 2);
    return hidden;
}

}  // namespace axd
