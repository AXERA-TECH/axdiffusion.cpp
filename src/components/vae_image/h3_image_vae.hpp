// MiniMax-H3-Image VAE decoder (2D ViT, 24 layers) on the NPU, decoded in 256 px tiles blended on the host.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model_dir.hpp"
#include "npu.hpp"

namespace axd {

struct VaeWeights {
    std::unique_ptr<TensorBank> bank;
};
std::unique_ptr<VaeWeights> load_vae_weights(const ModelDir &dir);

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

}  // namespace axd
