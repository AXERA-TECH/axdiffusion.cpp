// MiniMax-H3-Image text-to-image pipeline on AX8860:
//   tokenizer -> text encoder (50 layers, NPU) -> DiT (6 steps x 50 blocks, NPU) -> VAE (tiles, NPU) -> PNG
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "lora.hpp"
#include "npu.hpp"
#include "safetensors.hpp"
#include "tokenizer.hpp"

namespace axd {

// ------------------------------------------------------------------ model directory
class ModelDir {
public:
    explicit ModelDir(const std::string &root);
    std::string axmodel(const std::string &purpose, const std::string &shape_prefix = "") const;
    bool has_axmodel(const std::string &purpose, const std::string &shape_prefix = "") const;
    std::string file(const std::string &rel) const { return root_ + "/" + rel; }
    const std::string &root() const { return root_; }

private:
    struct Entry {
        std::string purpose, shape, file;
    };
    std::string root_;
    std::vector<Entry> models_;
};

// ------------------------------------------------------------------ constants
constexpr int kDitHidden = 5376;
constexpr int kDitFfn = 14336;
constexpr int kDitHeads = 56;
constexpr int kHeadDim = 128;
constexpr int kDitLayers = 50;
constexpr int kPatchDim = 96;   // 24 latent channels x 2 x 2
constexpr int kTextDim = 5120;
constexpr int kTextLayers = 50;
constexpr int kMaxTextTokens = 256;
constexpr int kLatentChannels = 24;
constexpr int kSteps = 6;
extern const float kSigmas[kSteps + 1];

// ------------------------------------------------------------------ resident weights
struct DitWeights {
    std::unique_ptr<TensorBank> shared, adaln;
    std::vector<std::unique_ptr<TensorBank>> blocks;
    std::vector<float> proj_in_w, proj_in_b;    // [5376,96], [5376] (host projection)
    std::vector<float> context_w, context_b;    // [5376,5120], [5376]
    DeviceBuffer adaln_blocks;                  // [6,50,6,3,5376] bf16
    Buf adaln_slot(int step, int layer, int slot) const;
    Buf adaln_input(int step) const;
    Buf final_shift(int step) const;
    Buf final_scale(int step) const;
};

struct TextWeights {
    DeviceBuffer embed;                                  // [151936,5120] bf16
    std::vector<std::unique_ptr<TensorBank>> layers;     // kernel input name -> buffer
};

struct VaeWeights {
    std::unique_ptr<TensorBank> bank;
};

// ------------------------------------------------------------------ stages
class TextEncoder {
public:
    TextEncoder(const ModelDir &dir, const TextWeights &w);
    // Returns the un-normalized hidden state after layer 50, [n,5120] bf16 words.
    std::vector<uint16_t> encode(const std::vector<int32_t> &ids);

private:
    const TextWeights &w_;
    std::unique_ptr<Model> vocab_, attn_, mlp_;
    DeviceBuffer hidden_[2], ids_, count_, positions_, cos_, sin_, paged_, page_out_, indptr_, indices_, cache_loc_,
        valid_, last_page_len_;
};

struct DitShape;  // per padded-length workspace
struct RefinerWs;

class Dit {
public:
    Dit(const ModelDir &dir, const DitWeights &w);
    ~Dit();
    // Run the full sampler. `sample` is [image_tokens,96] float32 noise (patchified) and is
    // updated in place to the final latent rows.
    struct Timing {
        double prepare = 0, steps = 0;
        std::vector<double> step_seconds;
        std::vector<double> block_seconds;  // all blocks of all steps
    };
    void run(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
             std::vector<float> &sample, int steps, int blocks, Timing &t, const LoraSet *lora = nullptr);
    bool has_lora_kernel(int padded) const { return lora_block_.count(padded) != 0; }
    int used() const { return used_; }
    int padded() const { return padded_; }

private:
    DitShape &shape(int padded);
    RefinerWs &refiner_ws(int rows);
    void prepare(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
                 const LoraSet *lora);
    void step(std::vector<float> &sample, int step_index, int blocks, Timing &t, const LoraSet *lora);

    const DitWeights &w_;
    std::map<int, std::unique_ptr<Model>> block_, lora_block_, final_, refiner_, rmsnorm_;
    std::map<int, std::unique_ptr<DitShape>> shapes_;
    std::map<int, std::unique_ptr<RefinerWs>> refiners_;
    Buf refined_text_;
    int text_rows_ = 0, used_ = 0, padded_ = 0, image_tokens_ = 0;
};

class Vae {
public:
    Vae(const ModelDir &dir, const VaeWeights &w);
    ~Vae();
    // latents [24,H,W] float32 (normalized latent space) -> RGB8 image [H*16,W*16,3].
    std::vector<uint8_t> decode(const std::vector<float> &latents, int h, int w, double &seconds);

private:
    struct Batch;
    Batch &batch(int n);
    void decode_tiles(const std::vector<uint16_t> &tokens, int n, std::vector<float> &pixels);
    const VaeWeights &w_;
    std::map<std::string, std::map<int, std::unique_ptr<Model>>> models_;
    std::map<int, std::unique_ptr<Batch>> batches_;
    DeviceBuffer cos_, sin_, dummy_bf16_, dummy_f32_;
};

// ------------------------------------------------------------------ pipeline
struct GenerateRequest {
    std::string prompt;
    uint64_t seed = 42;
    int size = 1024;  // 1024 or 2048 (square)
    int steps = kSteps;
    int blocks = kDitLayers;
    std::string out_png;     // optional
    std::string dump_dir;    // optional: write raw tensors for verification
    std::vector<LoraSpec> loras;  // optional dynamic LoRAs
    std::vector<Image> references;  // optional reference images: switches to reference-image editing
};

struct GenerateResult {
    std::vector<int32_t> ids;
    bool truncated = false;
    int used = 0, padded = 0;
    double lora_s = 0, tokenize_s = 0, text_s = 0, dit_prepare_s = 0, dit_steps_s = 0, vae_s = 0, png_s = 0, total_s = 0;
    std::vector<double> step_seconds;
    double block_mean_s = 0;
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
};

class Pipeline {
public:
    explicit Pipeline(const std::string &model_dir, const std::string &lora_dir = "", int lora_cache = 4);
    std::vector<std::string> loras() const { return lora_ ? lora_->available() : std::vector<std::string>{}; }
    ~Pipeline();
    GenerateResult generate(const GenerateRequest &req);
    double load_seconds() const { return load_s_; }
    void cmm(size_t &used, size_t &total) const { rt_->cmm(used, total); }

private:
    std::unique_ptr<Runtime> rt_;
    std::unique_ptr<ModelDir> dir_;
    std::unique_ptr<Tokenizer> tok_;
    std::unique_ptr<DitWeights> dit_w_;
    std::unique_ptr<TextWeights> text_w_;
    std::unique_ptr<VaeWeights> vae_w_;
    std::unique_ptr<TextEncoder> text_;
    std::unique_ptr<Dit> dit_;
    std::unique_ptr<Vae> vae_;
    std::unique_ptr<LoraManager> lora_;
    double load_s_ = 0;
};

// host helpers (pipeline_host.cpp)
std::vector<float> torch_randn(uint64_t seed, size_t count);
std::vector<float> patchify(const std::vector<float> &noise, int h, int w);    // [24,H,W] -> [H/2*W/2,96]
std::vector<float> unpatchify(const std::vector<float> &rows, int h, int w);   // inverse
bool write_png(const std::string &path, const std::vector<uint8_t> &rgb, int w, int h);
std::string encode_png(const std::vector<uint8_t> &rgb, int w, int h);  // in-memory PNG bytes

std::unique_ptr<DitWeights> load_dit_weights(const ModelDir &dir);
std::unique_ptr<TextWeights> load_text_weights(const ModelDir &dir);
std::unique_ptr<VaeWeights> load_vae_weights(const ModelDir &dir);

}  // namespace axd
