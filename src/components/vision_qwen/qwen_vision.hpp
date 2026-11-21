// Qwen3-VL vision tower (27 blocks + merger) on the NPU for MiniMax-H3 reference images.
// The tower is compiled as a chain of stage AXModels (manifest purpose "vision-vit", shape "stage K, grid G"):
// stage 0 takes the packed pixels, every stage passes its hidden state on as output 0, and the extra outputs,
// in stage order, are deepstack 0, deepstack 1, merger, deepstack 2 (all [tokens,5120] bf16).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "model_dir.hpp"
#include "npu.hpp"

namespace axd {

struct VisionFeatures {
    int tokens = 0;                         // merged tokens (grid_h/2 * grid_w/2)
    std::vector<uint16_t> merger;           // [tokens,5120] bf16
    std::array<std::vector<uint16_t>, 3> deepstack;
};

class VisionEncoder {
public:
    explicit VisionEncoder(const ModelDir &dir);
    ~VisionEncoder();
    // pixel_values: [patches,1536] float (Qwen packing) for one image with grid (1, grid_h, grid_w).
    VisionFeatures encode(const std::vector<float> &pixel_values, int grid_h, int grid_w);
    int grid() const { return grid_; }  // compiled patch grid (square)

private:
    std::vector<std::unique_ptr<Model>> stages_;
    std::unique_ptr<Stream> stream_;
    int grid_ = 0, patches_ = 0, tokens_ = 0, hidden_ = 0;
    DeviceBuffer pixels_, h_[2], extra_[4];
};

}  // namespace axd
