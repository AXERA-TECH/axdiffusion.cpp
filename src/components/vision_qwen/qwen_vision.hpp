// Qwen3-VL vision tower (27 blocks + merger + 3 deepstack mergers) on the NPU for MiniMax-H3 reference images.
// Six runtime-grid kernels per capacity tier (manifest purposes "vision-vit-embed|pre|attn|post|merge|merge-ds",
// shape "... cap N"): the host loops the layers, the learned weights are device tensors bound per launch, and the
// position table and the 2D rotary table are computed here for the request's patch grid. Rows are in the official
// 2x2 merge-block order, so merged row k is image token k.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
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
    static bool available(const ModelDir &dir);
    // pixel_values: [patches,1536] float (Qwen packing, merge-block order) for one image with grid (1, grid_h, grid_w).
    VisionFeatures encode(const std::vector<float> &pixel_values, int grid_h, int grid_w);
    // Largest patch count (grid_h * grid_w) the kernels take.
    int max_patches() const;

private:
    struct Tier {
        std::map<std::string, std::string> paths;                 // kernel -> file
        std::map<std::string, std::unique_ptr<Model>> models;     // loaded on first use
    };
    Model &kernel(Tier &t, const std::string &name);
    void workspace(int cap);

    std::map<int, Tier> tiers_;  // by capacity (rows)
    std::unique_ptr<TensorBank> bank_;
    std::vector<uint16_t> patch_weight_;  // [1152,768] bf16 (temporal kernels summed); bank source
    std::vector<float> pos_table_;        // [48*48,1152]
    std::unique_ptr<Stream> stream_;
    int cap_ = 0;  // capacity the buffers below are sized for
    DeviceBuffer bounds_, px_, pos_, cos_, sin_, x_[2], q_, k_, v_, attn_, out_[4];
};

}  // namespace axd
