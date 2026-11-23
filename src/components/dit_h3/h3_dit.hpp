// MiniMax-H3 DiT on the NPU (shared by the image and video models): host text/image projections,
// 2-layer token refiner, Euler steps over 50 blocks with precomputed AdaLN, PDD/final head, optional LoRA branch.
#pragma once

#include <array>
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
    // adaln: blocks [6,50,6,3,5376], input [6,2688], final [6,5376] (bf16).
    // adaln_edit (reference editing, M=2: [t_i, 0.999] per step), optional: blocks [6,50,6,6,5376],
    // input [6,2,2688], final [6,2,5376].
    std::unique_ptr<TensorBank> adaln_edit;
    Buf adaln_slot(int step, int layer, int slot, bool edit = false) const;
    Buf adaln_input(int step, bool edit = false) const;
    Buf final_shift(int step, bool edit = false) const;
    Buf final_scale(int step, bool edit = false) const;
};
std::unique_ptr<DitWeights> load_dit_weights(const ModelDir &dir);

struct LoraSet;
struct DitShape;

// Reference-image conditioning: the packed sequence is [text | references | generated image]. Reference rows are
// fixed (noised to t=0.999 once) and read AdaLN row 3; text rows read their tag (1 text, 0 vision block).
struct DitEdit {
    std::vector<int32_t> text_tags;                // [text_rows]
    std::vector<float> reference_rows;             // [reference_rows,96] patchified, noised references
    std::vector<std::array<double, 3>> positions;  // [used] packed (t, h, w) position ids
};
struct RefinerWs;

class Dit {
public:
    Dit(const ModelDir &dir, const DitWeights &w);
    ~Dit();
    struct Timing {
        double prepare = 0, steps = 0;
        std::vector<double> step_seconds;
        std::vector<double> block_seconds;  // all blocks of all steps (host bind+submit time when async)
        double sync_seconds = 0;            // waiting for the queued launches (async)
    };
    // Run the sampler. `sample` is [image_tokens,96] float32 noise (patchified) and is updated in place to the
    // final latent rows. `progress` (optional) receives one report per step and is checked for cancellation.
    void run(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
             std::vector<float> &sample, int steps, int blocks, Timing &t, const LoraSet *lora = nullptr,
             Progress *progress = nullptr, const DitEdit *edit = nullptr);
    bool supports_edit() const { return !edit_block_.empty() && w_.adaln_edit != nullptr; }
    int max_edit_rows() const { return edit_block_.empty() ? 0 : edit_block_.rbegin()->first; }
    bool has_lora_kernel(int padded, int rank) const {
        auto it = lora_block_.find(rank);
        return it != lora_block_.end() && it->second.count(padded) != 0;
    }
    std::vector<int> lora_ranks() const;  // rank tiers that have LoRA block kernels
    // Longest text (rows) any runtime tier accepts next to `image_rows` image rows (LoRA rank tier if rank > 0).
    int max_text_rows(int image_rows, int lora_rank = 0) const;
    // Smallest tier whose kernel accepts `used` rows (0 = none). lora_rank 0 = base kernels.
    int pick_tier(int used, int lora_rank, bool edit) const;
    int used() const { return used_; }
    int padded() const { return padded_; }

private:
    DitShape &shape(int padded);
    RefinerWs &refiner_ws(int rows);
    void prepare(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
                 const LoraSet *lora, const DitEdit *edit);
    void step(std::vector<float> &sample, int step_index, int blocks, Timing &t, const LoraSet *lora);

    const DitWeights &w_;
    // A tier kernel: loaded on first use. Block/refiner kernels take runtime sequence bounds and accept used rows
    // in (padded - granule, padded]; the granule is part of the manifest shape ("granule N").
    struct Kernel {
        std::string path, what;
        int granule = 0;
        std::unique_ptr<Model> model;
        Model &get();
    };
    using Tiers = std::map<int, Kernel>;
    Tiers block_, final_, refiner_, rmsnorm_, edit_block_, edit_final_;
    Kernel context_proj_;  // text projection, 256-row pages
    DeviceBuffer ctx_in_, ctx_out_;
    bool edit_ = false;                 // the prepared request is an edit
    int reference_rows_ = 0;
    std::vector<uint16_t> reference_proj_;  // [reference_rows,5376] bf16, projected once per request
    std::map<int, Tiers> lora_block_;       // [rank][padded]
    std::map<int, Tiers> edit_lora_block_;  // [rank][padded], reference editing with a LoRA
    std::unique_ptr<Stream> stream_;  // block + final launches of a step are queued here
    std::map<int, std::unique_ptr<DitShape>> shapes_;
    std::map<int, std::unique_ptr<RefinerWs>> refiners_;
    Buf refined_text_;
    int text_rows_ = 0, used_ = 0, padded_ = 0, image_tokens_ = 0;
};

// [24,H,W] <-> rows [(H/2)*(W/2), 96] with column c*4 + ph*2 + pw
std::vector<float> patchify(const std::vector<float> &latent, int h, int w);
std::vector<float> unpatchify(const std::vector<float> &rows, int h, int w);

}  // namespace axd
