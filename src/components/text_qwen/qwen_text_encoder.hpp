// Qwen3-VL text tower on the NPU: token embedding + the first N decoder layers, returning the
// un-normalized hidden state (MiniMax-H3 conditions on hidden_states[50]).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "model_dir.hpp"
#include "npu.hpp"

namespace axd {

constexpr int kTextDim = 5120;
constexpr int kTextLayers = 50;
constexpr int kMaxTextTokens = 256;  // single-page kernels (older model directories)
constexpr int kTextHeadDim = 128;
constexpr int kTextPage = 256;
constexpr int kTextPages = 32;       // paged kernels: KV cache capacity (8192 tokens)

struct TextWeights {
    DeviceBuffer embed;                               // [vocab,5120] bf16
    std::vector<std::unique_ptr<TensorBank>> layers;  // kernel input name -> buffer
};
std::unique_ptr<TextWeights> load_text_weights(const ModelDir &dir);

// Vision features injected into the text stream (Qwen3-VL image tokens): rows are global token indices in
// presentation order; merger / deepstack[k] are [rows.size(),5120] bf16. merger replaces the token embedding,
// deepstack[k] is added to the input of decoder layer k+1.
struct TextVision {
    std::vector<int32_t> rows;
    const uint16_t *merger = nullptr;
    const uint16_t *deepstack[3] = {nullptr, nullptr, nullptr};
};

class TextEncoder {
public:
    TextEncoder(const ModelDir &dir, const TextWeights &w);
    ~TextEncoder();
    // ids: token ids; mrope: optional [n][3] Qwen M-RoPE positions (default: i,i,i); vision: optional image rows.
    // Returns hidden_states[50] as [n,5120] bf16 words.
    std::vector<uint16_t> encode(const std::vector<int32_t> &ids, const std::vector<std::array<int32_t, 3>> *mrope = nullptr,
                                 const TextVision *vision = nullptr);
    int max_tokens() const { return paged_kernels_ ? kTextPage * kTextPages : kMaxTextTokens; }
    bool supports_vision() const { return attn_deepstack_ != nullptr; }

private:
    std::vector<uint16_t> encode_single(const std::vector<int32_t> &ids);
    std::vector<uint16_t> encode_paged(const std::vector<int32_t> &ids, const std::vector<std::array<int32_t, 3>> *mrope,
                                       const TextVision *vision);
    const TextWeights &w_;
    bool paged_kernels_ = false;
    std::unique_ptr<Model> vocab_, attn_, mlp_, attn_deepstack_;
    std::unique_ptr<Stream> stream_;
    // paged state: every page's hidden rows, per-page control inputs, the shared KV cache, deepstack pages
    DeviceBuffer pages_x_, pages_attn_, page_ctrl_, kv_cache_, kv_page_out_, deepstack_;
    size_t deepstack_pages_ = 0;
    DeviceBuffer hidden_[2], ids_, count_, positions_, cos_, sin_, paged_, page_out_, indptr_, indices_, cache_loc_,
        valid_, last_page_len_;
};

}  // namespace axd
