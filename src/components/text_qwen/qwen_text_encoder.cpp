// Qwen3-VL text path, layers 0..49, on the NPU: paged prefill (256-token pages, one attention + one MLP kernel per page and layer).
#include <algorithm>
#include <cmath>

#include "common.hpp"
#include "qwen_text_encoder.hpp"
#include "safetensors.hpp"

namespace axd {

namespace {
constexpr double kRopeTheta = 5000000.0;
// per-page control block (one allocation, 256-byte slots): positions [3,256] i32, indptr [2], cache_location [2],
// valid_tokens [1], num_tokens [1], ids [256]
constexpr size_t kCtrlPositions = 0, kCtrlIndptr = 3 * 256 * 4, kCtrlLoc = kCtrlIndptr + 256, kCtrlValid = kCtrlLoc + 256,
                 kCtrlIds = kCtrlValid + 256, kCtrlBytes = kCtrlIds + 256 * 4;
}  // namespace

TextEncoder::~TextEncoder() = default;

TextEncoder::TextEncoder(const ModelDir &dir, const TextWeights &w) : w_(w) {
    // 256-token pages over a 32-page KV cache (prompts up to 8192 tokens incl. image tokens).
    vocab_ = std::make_unique<Model>(dir.axmodel("text-vocab-embedding-paged"));
    attn_ = std::make_unique<Model>(dir.axmodel("text-attention-paged"));
    mlp_ = std::make_unique<Model>(dir.axmodel("text-mlp-paged"));
    if (dir.has_axmodel("text-attention-paged-deepstack"))
        attn_deepstack_ = std::make_unique<Model>(dir.axmodel("text-attention-paged-deepstack"));
    const size_t page = size_t(kTextPage) * kTextDim * 2;
    pages_x_ = DeviceBuffer(page * kTextPages, "text-pages");
    pages_attn_ = DeviceBuffer(page, "text-attn");
    page_ctrl_ = DeviceBuffer(kCtrlBytes * kTextPages, "text-page-ctrl");
    kv_cache_ = DeviceBuffer(size_t(kTextPages) * 2 * kTextPage * 8 * kTextHeadDim * 2, "text-kv-cache");
    kv_page_out_ = DeviceBuffer(size_t(2) * kTextPage * 8 * kTextHeadDim * 2, "text-kv-page-out");
    indices_ = DeviceBuffer(kTextPages * 4, "text-indices");
    std::vector<int32_t> idx(kTextPages);
    for (int i = 0; i < kTextPages; ++i) idx[i] = i;
    indices_.buf().upload(idx.data(), idx.size() * 4);
    zero_ = DeviceBuffer(4, "text-zero");  // kv_write_page_cursor (unused with cache_location)
    zero_.buf().fill_zero();
    const int rows = kTextPage * kTextPages;
    std::vector<float> c(size_t(rows) * 64), s(size_t(rows) * 64);
    for (int p = 0; p < rows; ++p)
        for (int i = 0; i < 64; ++i) {
            double a = double(p) / std::pow(kRopeTheta, double(2 * i) / kTextHeadDim);
            c[size_t(p) * 64 + i] = float(std::cos(a));
            s[size_t(p) * 64 + i] = float(std::sin(a));
        }
    cos_ = DeviceBuffer(c.size() * 4, "text-cos");
    sin_ = DeviceBuffer(s.size() * 4, "text-sin");
    cos_.buf().upload(c.data(), c.size() * 4);
    sin_.buf().upload(s.data(), s.size() * 4);
}

// Paged prefill: layer-major, page-minor (every layer rewrites each page's KV slot before a later page reads it,
// so one 32-page cache serves all 50 layers). Within the current page attention is causal.
std::vector<uint16_t> TextEncoder::encode(const std::vector<int32_t> &ids,
                                          const std::vector<std::array<int32_t, 3>> *mrope, const TextVision *vision) {
    const int n = int(ids.size());
    if (n <= 0) fail("empty text");
    const int pages = (n + kTextPage - 1) / kTextPage;
    if (pages > kTextPages)
        fail("text of " + std::to_string(n) + " tokens exceeds the encoder capacity of " +
             std::to_string(kTextPage * kTextPages));
    if (mrope && int(mrope->size()) != n) fail("M-RoPE positions do not match the token count");
    if (vision && !attn_deepstack_) fail("this model directory has no deepstack text kernel");
    if (!stream_) stream_ = std::make_unique<Stream>();
    const size_t page_bytes = size_t(kTextPage) * kTextDim * 2;
    Buf X = pages_x_.buf(), C = page_ctrl_.buf();
    auto page_x = [&](int p) { return X.view(size_t(p) * page_bytes, page_bytes); };
    auto ctrl = [&](int p, size_t off, size_t n_) { return C.view(size_t(p) * kCtrlBytes + off, n_); };

    // per-page control inputs
    std::vector<uint8_t> blk(kCtrlBytes * pages, 0);
    int max_pos = 0;
    for (int p = 0; p < pages; ++p) {
        const int start = p * kTextPage, rows = std::min(kTextPage, n - start);
        uint8_t *b = blk.data() + size_t(p) * kCtrlBytes;
        auto *pos = reinterpret_cast<int32_t *>(b + kCtrlPositions);
        for (int r = 0; r < kTextPage; ++r) {
            const int i = start + std::min(r, rows - 1);  // tail rows repeat the last position
            for (int a = 0; a < 3; ++a) {
                int v = mrope ? (*mrope)[i][a] : i;
                pos[a * kTextPage + r] = v;
                max_pos = std::max(max_pos, v);
            }
        }
        auto *indptr = reinterpret_cast<int32_t *>(b + kCtrlIndptr);
        indptr[0] = 0, indptr[1] = p;
        auto *loc = reinterpret_cast<int32_t *>(b + kCtrlLoc);
        loc[0] = p, loc[1] = 0;
        *reinterpret_cast<int32_t *>(b + kCtrlValid) = rows;
        *reinterpret_cast<int32_t *>(b + kCtrlValid + 4) = kTextPage;  // num_tokens for the embedding gather
        auto *pid = reinterpret_cast<int32_t *>(b + kCtrlIds);
        for (int r = 0; r < kTextPage; ++r) pid[r] = ids[start + std::min(r, rows - 1)];  // tail repeats the last id
    }
    if (max_pos >= kTextPage * kTextPages) fail("M-RoPE position out of the RoPE table range");
    C.upload(blk.data(), blk.size());

    // token embeddings (all 256 rows of a page, the tail gets the last token's embedding)
    for (int p = 0; p < pages; ++p) {
        vocab_->submit(*stream_, {{"input_flat", ctrl(p, kCtrlIds, kTextPage * 4)}, {"weight", w_.embed},
                                  {"num_tokens", ctrl(p, kCtrlValid + 4, 4)}},
                       {{"output_flat", page_x(p)}});
    }
    stream_->sync();

    // vision: merged features replace the image-token embeddings; deepstack pages for layers 1..3
    std::vector<int> ds_slot(pages, -1);
    if (vision && !vision->rows.empty()) {
        const auto &rows = vision->rows;
        for (size_t k = 0; k < rows.size(); ++k)
            if (rows[k] < 0 || rows[k] >= n) fail("vision row out of range");
        const size_t row_bytes = size_t(kTextDim) * 2;
        for (size_t k = 0; k < rows.size();) {  // contiguous runs
            size_t e = k + 1;
            while (e < rows.size() && rows[e] == rows[e - 1] + 1) ++e;
            X.view(size_t(rows[k]) * row_bytes, (e - k) * row_bytes).upload(vision->merger + k * kTextDim, (e - k) * row_bytes);
            k = e;
        }
        int used_pages = 0;
        for (int r : rows)
            if (ds_slot[r / kTextPage] < 0) ds_slot[r / kTextPage] = used_pages++;
        const size_t need = size_t(used_pages) * 3;
        if (need > deepstack_pages_) {
            deepstack_ = DeviceBuffer(need * page_bytes, "text-deepstack");
            deepstack_pages_ = need;
        }
        std::vector<uint16_t> buf(size_t(kTextPage) * kTextDim);
        for (int lvl = 0; lvl < 3; ++lvl)
            for (int p = 0; p < pages; ++p) {
                if (ds_slot[p] < 0) continue;
                std::fill(buf.begin(), buf.end(), 0);
                for (size_t k = 0; k < rows.size(); ++k)
                    if (rows[k] / kTextPage == p)
                        std::copy(vision->deepstack[lvl] + k * kTextDim, vision->deepstack[lvl] + (k + 1) * kTextDim,
                                  buf.begin() + size_t(rows[k] % kTextPage) * kTextDim);
                deepstack_.buf()
                    .view((size_t(ds_slot[p]) * 3 + lvl) * page_bytes, page_bytes)
                    .upload(buf.data(), page_bytes);
            }
    }

    static const char *kAttn[] = {"x_norm_w", "w_q", "w_scale_q", "w_scale2_q", "w_zeropoint_q", "w_k", "w_scale_k",
                                  "w_scale2_k", "w_zeropoint_k", "w_v", "w_scale_v", "w_scale2_v", "w_zeropoint_v",
                                  "q_norm_w", "k_norm_w", "w_o", "w_scale_o", "w_scale2_o", "w_zeropoint_o"};
    static const char *kMlp[] = {"w_norm", "w1", "w_scale1", "w_scale2_1", "w2", "w_scale2", "w_scale2_2", "w3",
                                 "w_scale3", "w_scale2_3", "w_zp_scale1", "w_zp_scale2", "w_zp_scale3", "input_permutation"};
    for (int l = 0; l < int(w_.layers.size()); ++l) {
        const auto &layer = *w_.layers[l];
        Bindings a = {{"cos", cos_}, {"sin", sin_}, {"paged_kv_cache", kv_cache_}, {"indices", indices_},
                      {"kv_write_page_cursor", zero_}};
        for (const char *k : kAttn) a[k] = layer.get(k);
        Bindings m;
        for (const char *k : kMlp) m[k] = layer.get(k);
        for (int p = 0; p < pages; ++p) {
            a["x"] = page_x(p);
            a["position_ids"] = ctrl(p, kCtrlPositions, 3 * kTextPage * 4);
            a["indptr"] = ctrl(p, kCtrlIndptr, 8);
            a["cache_location"] = ctrl(p, kCtrlLoc, 8);
            a["valid_tokens"] = ctrl(p, kCtrlValid, 4);
            Model *attn = attn_.get();
            if (l >= 1 && l <= 3 && ds_slot[p] >= 0) {
                attn = attn_deepstack_.get();
                a["post_residual_addition"] =
                    deepstack_.buf().view((size_t(ds_slot[p]) * 3 + (l - 1)) * page_bytes, page_bytes);
            }
            attn->submit(*stream_, a, {{"output", pages_attn_}, {"kv_page_out", kv_page_out_}});
            m["x"] = pages_attn_;
            mlp_->submit(*stream_, m, {{"output", page_x(p)}});
        }
    }
    stream_->sync();
    std::vector<uint16_t> hidden(size_t(n) * kTextDim);
    X.download(hidden.data(), hidden.size() * 2);
    return hidden;
}

std::unique_ptr<TextWeights> load_text_weights(const ModelDir &dir) {
    auto w = std::make_unique<TextWeights>();
    std::vector<std::unique_ptr<SafeTensors>> files;
    std::vector<TensorBank *> banks;
    files.push_back(std::make_unique<SafeTensors>(dir.file("text_encoder/embed_bf16.safetensors")));
    w->embed_bank = std::make_unique<TensorBank>("text-embedding");
    {
        auto &t = files.back()->get("weight");
        w->embed_bank->add_file("weight", files.back()->fd(), files.back()->file_offset(t), t.nbytes);
        banks.push_back(w->embed_bank.get());
    }
    // kernel input name -> tensor name in the layer file
    static const std::pair<const char *, const char *> kMap[] = {
        {"x_norm_w", "attention.w_norm_attn"}, {"w_q", "attention.w_q"},
        {"w_scale_q", "attention.w_scale_q"}, {"w_scale2_q", "attention.w_scale2_q"},
        {"w_zeropoint_q", "attention.w_zeropoint_q"}, {"w_k", "attention.w_k"},
        {"w_scale_k", "attention.w_scale_k"}, {"w_scale2_k", "attention.w_scale2_k"},
        {"w_zeropoint_k", "attention.w_zeropoint_k"}, {"w_v", "attention.w_v"},
        {"w_scale_v", "attention.w_scale_v"}, {"w_scale2_v", "attention.w_scale2_v"},
        {"w_zeropoint_v", "attention.w_zeropoint_v"}, {"q_norm_w", "attention.w_norm_q"},
        {"k_norm_w", "attention.w_norm_k"}, {"w_o", "attention.w_o"},
        {"w_scale_o", "attention.w_scale_o"}, {"w_scale2_o", "attention.w_scale2_o"},
        {"w_zeropoint_o", "attention.w_zeropoint_o"}, {"w_norm", "mlp.w_norm"},
        {"w1", "mlp.w1"}, {"w_scale1", "mlp.w_scale1"}, {"w_scale2_1", "mlp.w_scale2_1"},
        {"w2", "mlp.w2"}, {"w_scale2", "mlp.w_scale2"}, {"w_scale2_2", "mlp.w_scale2_2"},
        {"w3", "mlp.w3"}, {"w_scale3", "mlp.w_scale3"}, {"w_scale2_3", "mlp.w_scale2_3"},
        {"w_zp_scale1", "mlp.w_zp_scale1"}, {"w_zp_scale2", "mlp.w_zp_scale2"},
        {"w_zp_scale3", "mlp.w_zp_scale3"}, {"input_permutation", "mlp.input_permutation"},
    };
    for (int i = 0; i < kTextLayers; ++i) {
        char rel[64];
        std::snprintf(rel, sizeof(rel), "text_encoder/layers/layer_%03d.safetensors", i);
        files.push_back(std::make_unique<SafeTensors>(dir.file(rel)));
        auto bank = std::make_unique<TensorBank>("text-layer-" + std::to_string(i));
        for (auto &kv : kMap) {
            auto &t = files.back()->get(kv.second);
            bank->add_file(kv.first, files.back()->fd(), files.back()->file_offset(t), t.nbytes);
        }
        banks.push_back(bank.get());
        w->layers.push_back(std::move(bank));
    }
    materialize_parallel(banks);
    w->embed = w->embed_bank->get("weight");
    return w;
}


}  // namespace axd
