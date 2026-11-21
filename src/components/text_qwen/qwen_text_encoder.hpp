// Qwen3-VL text tower on the NPU: token embedding + the first N decoder layers, returning the
// un-normalized hidden state (MiniMax-H3 conditions on hidden_states[50]).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "model_dir.hpp"
#include "npu.hpp"

namespace axd {

constexpr int kTextDim = 5120;
constexpr int kTextLayers = 50;
constexpr int kMaxTextTokens = 256;
constexpr int kTextHeadDim = 128;

struct TextWeights {
    DeviceBuffer embed;                               // [vocab,5120] bf16
    std::vector<std::unique_ptr<TensorBank>> layers;  // kernel input name -> buffer
};
std::unique_ptr<TextWeights> load_text_weights(const ModelDir &dir);

class TextEncoder {
public:
    TextEncoder(const ModelDir &dir, const TextWeights &w);
    // [n,5120] bf16 words.
    std::vector<uint16_t> encode(const std::vector<int32_t> &ids);

private:
    const TextWeights &w_;
    std::unique_ptr<Model> vocab_, attn_, mlp_;
    DeviceBuffer hidden_[2], ids_, count_, positions_, cos_, sin_, paged_, page_out_, indptr_, indices_, cache_loc_,
        valid_, last_page_len_;
};

}  // namespace axd
