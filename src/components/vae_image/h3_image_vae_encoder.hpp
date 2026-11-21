// MiniMax-H3 image VAE encoder (+ quant_conv) on the NPU, one 256x256 pixel tile per launch.
// Tiling, blending and posterior sampling are done by the caller (they follow the official pipeline).
#pragma once

#include <memory>
#include <vector>

#include "model_dir.hpp"
#include "npu.hpp"

namespace axd {

constexpr int kVaeEncTile = 256;      // pixels
constexpr int kVaeEncMoments = 48;    // mean (24) + logvar (24)
constexpr int kVaeEncTileLatent = 16; // kVaeEncTile / 16

class VaeEncoder {
public:
    explicit VaeEncoder(const ModelDir &dir);
    ~VaeEncoder();
    // tiles: each [3,256,256] float32 (ImageNet-normalized) -> each [48,16,16] float32 moments.
    std::vector<std::vector<float>> encode_tiles(const std::vector<std::vector<float>> &tiles);

private:
    std::unique_ptr<Model> model_;
    std::unique_ptr<Stream> stream_;
    std::string in_name_, out_name_;
    DeviceBuffer in_, out_;
    size_t capacity_ = 0;
};

}  // namespace axd
