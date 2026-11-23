// Host-side preparation for reference-image editing (no NPU): reference resizing, VAE pixel tiles,
// posterior sampling, Qwen3-VL pixel packing, the edit text presentation, M-RoPE positions, the packed
// DiT layout and the per-step two-timestep plan. Mirrors the official MiniMax-H3-Image pipeline.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "image_io.hpp"

namespace axd {
namespace edit {

constexpr int kImageTokenId = 151655, kVisionStartId = 151652, kVisionEndId = 151653;
constexpr int kVitPatch = 16, kVitMerge = 2, kVitTemporal = 2;
constexpr float kReferenceNoiseT = 0.999f;

// Official bound_short_edge: returns {width, height}.
std::array<int, 2> bound_short_edge(int width, int height, int lo, int hi, int multiple = 32);

// Pillow-compatible LANCZOS resize of an 8-bit RGB image (Image.resize(..., LANCZOS, reducing_gap=None)).
Image resize_lanczos(const Image &in, int out_w, int out_h);

Image resize_for_vae(const Image &in);  // short edge into [1024, 2048], multiple of 32
Image resize_for_vit(const Image &in);  // short edge 1024, multiple of 32
// {width, height} the two resizes above produce for a source of this size (no pixels touched).
std::array<int, 2> vae_size(int width, int height);
std::array<int, 2> vit_size(int width, int height);

// uint8 RGB -> ImageNet-normalized CHW float, cropped to a multiple of 16.
std::vector<float> vae_pixels(const Image &img, int &h, int &w);

// diffusers tile layout along one axis (in pixels or latent units with the given ratio).
void split_tiles(int length, int tile, int min_overlap, int ratio, std::vector<int> &starts, std::vector<int> &overlaps);

// Blend + crop [C,th,tw] tiles (row-major tile grid) exactly like diffusers' _stitch_tiles.
std::vector<float> stitch_tiles(const std::vector<std::vector<float>> &tiles, int channels, int tile_h, int tile_w,
                                const std::vector<int> &y_overlaps, const std::vector<int> &x_overlaps, int out_h,
                                int out_w);

// moments [48,H,W] + eps [24,H,W] -> latent [24,H,W]: mean + exp(0.5*clamp(logvar)) * eps, rounded through fp16,
// then normalized with the VAE latent mean/std.
std::vector<float> posterior_latent(const std::vector<float> &moments, const std::vector<float> &eps, int h, int w,
                                    const float *latent_mean, const float *latent_std);

// Qwen3-VL pixel packing: [grid_h*grid_w, 1536] float and grid (1, grid_h, grid_w) in 16-px patch units.
std::vector<float> vit_pixel_values(const Image &img, std::array<int, 3> &grid_thw);

struct Presentation {
    std::vector<int32_t> ids;
    std::vector<int32_t> token_tags;  // H3 tags: 1 text, 0 vision block
    std::vector<int32_t> mm_types;    // Qwen: 1 image pad, 0 otherwise
};
Presentation make_presentation(const std::vector<std::vector<int32_t>> &label_ids,
                               const std::vector<int32_t> &prompt_ids, const std::vector<std::array<int, 3>> &grids);

// Qwen3-VL get_rope_index for one sample: [S][3] (t, h, w).
std::vector<std::array<int32_t, 3>> mrope_positions(const Presentation &p, const std::vector<std::array<int, 3>> &grids);

struct PackedLayout {
    int text_rows = 0, reference_rows = 0, target_rows = 0, used = 0;
    std::vector<std::array<double, 3>> position_ids;  // [used]
    std::vector<int32_t> token_tags;                  // [used]
};
// reference_grids / target are latent sizes {h, w}.
PackedLayout packed_layout(const std::vector<int32_t> &text_tags, const std::vector<std::array<int, 2>> &reference_grids,
                           int latent_h, int latent_w);

// Reference rectified-flow noising at t = 0.999: t * latent + (1 - t) * noise (float32).
void noise_reference(std::vector<float> &latent, const std::vector<float> &noise);

}  // namespace edit
}  // namespace axd
