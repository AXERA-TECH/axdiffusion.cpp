// MiniMax-H3 DiT on the NPU (shared by the image and video models): host text/image projections,
// 2-layer token refiner, Euler steps over 50 blocks with precomputed AdaLN, PDD/final head, optional LoRA branch.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model_dir.hpp"
#include "npu.hpp"
#include "task.hpp"

namespace axd {

constexpr int kDitHidden = 5376;
constexpr int kDitFfn = 14336;
constexpr int kDitHeads = 56;
constexpr int kHeadDim = 128;
constexpr int kDitLayers = 50;
constexpr int kPatchDim = 96;  // 24 latent channels x 2 x 2
constexpr int kLatentChannels = 24;
constexpr int kDitTextDim = 5120;  // width of the text-encoder hidden state the DiT consumes
constexpr int kSteps = 6;
extern const float kSigmas[kSteps + 1];

struct DitWeights {
    std::unique_ptr<TensorBank> shared, adaln;
    std::vector<std::unique_ptr<TensorBank>> blocks;
    std::vector<float> proj_in_w, proj_in_b;  // [5376,96], [5376] (host projection)
    std::vector<float> context_w, context_b;  // [5376,5120], [5376]
    DeviceBuffer adaln_blocks;                // [6,50,6,3,5376] bf16
    Buf adaln_slot(int step, int layer, int slot) const;
    Buf adaln_input(int step) const;
    Buf final_shift(int step) const;
    Buf final_scale(int step) const;
};
std::unique_ptr<DitWeights> load_dit_weights(const ModelDir &dir);

struct LoraSet;
struct DitShape;
struct RefinerWs;

class Dit {
public:
    Dit(const ModelDir &dir, const DitWeights &w);
    ~Dit();
    struct Timing {
        double prepare = 0, steps = 0;
        std::vector<double> step_seconds;
        std::vector<double> block_seconds;  // all blocks of all steps
    };
    // Run the sampler. `sample` is [image_tokens,96] float32 noise (patchified) and is updated in place to the
    // final latent rows. `progress` (optional) receives one report per step and is checked for cancellation.
    void run(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
             std::vector<float> &sample, int steps, int blocks, Timing &t, const LoraSet *lora = nullptr,
             Progress *progress = nullptr);
    bool has_lora_kernel(int padded, int rank) const {
        auto it = lora_block_.find(rank);
        return it != lora_block_.end() && it->second.count(padded) != 0;
    }
    std::vector<int> lora_ranks() const;  // rank tiers that have LoRA block kernels
    int used() const { return used_; }
    int padded() const { return padded_; }

private:
    DitShape &shape(int padded);
    RefinerWs &refiner_ws(int rows);
    void prepare(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
                 const LoraSet *lora);
    void step(std::vector<float> &sample, int step_index, int blocks, Timing &t, const LoraSet *lora);

    const DitWeights &w_;
    std::map<int, std::unique_ptr<Model>> block_, final_, refiner_, rmsnorm_;
    std::map<int, std::map<int, std::unique_ptr<Model>>> lora_block_;  // [rank][padded]
    int fast_tier_ = 0;  // static (no runtime bounds) 1024x1024 short-prompt tier, if present
    std::map<int, std::unique_ptr<DitShape>> shapes_;
    std::map<int, std::unique_ptr<RefinerWs>> refiners_;
    Buf refined_text_;
    int text_rows_ = 0, used_ = 0, padded_ = 0, image_tokens_ = 0;
};

// [24,H,W] <-> rows [(H/2)*(W/2), 96] with column c*4 + ph*2 + pw
std::vector<float> patchify(const std::vector<float> &latent, int h, int w);
std::vector<float> unpatchify(const std::vector<float> &rows, int h, int w);

}  // namespace axd
