// Pipeline assembly and host-side helpers (noise, patchify, PNG).
#include "pipeline.hpp"

#include <cmath>
#include <fstream>

#include "common.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace axd {


// ------------------------------------------------------------------ noise
// Reproduces torch.randn((1,24,1,H,W)) on CPU (MT19937 + 16-wide Box-Muller), so a seed
// gives the same image as the reference pipeline.
namespace {
struct MT19937 {
    uint32_t s[624];
    int idx = 624;
    explicit MT19937(uint32_t seed) {
        s[0] = seed;
        for (int i = 1; i < 624; ++i) s[i] = 1812433253u * (s[i - 1] ^ (s[i - 1] >> 30)) + uint32_t(i);
    }
    uint32_t next() {
        if (idx >= 624) {
            for (int i = 0; i < 624; ++i) {
                uint32_t y = (s[i] & 0x80000000u) | (s[(i + 1) % 624] & 0x7FFFFFFFu);
                s[i] = s[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908B0DFu : 0u);
            }
            idx = 0;
        }
        uint32_t y = s[idx++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680u;
        y ^= (y << 15) & 0xEFC60000u;
        y ^= y >> 18;
        return y;
    }
    float uniform() { return float(next() & 0xFFFFFFu) * (1.0f / float(1u << 24)); }
};

void normal_fill16(float *d) {
    for (int j = 0; j < 8; ++j) {
        float u1 = 1.0f - d[j], u2 = d[j + 8];
        float radius = sqrtf(-2.0f * logf(u1));
        float theta = float(2.0 * M_PI * double(u2));
        d[j] = radius * cosf(theta);
        d[j + 8] = radius * sinf(theta);
    }
}
}  // namespace

std::vector<float> torch_randn(uint64_t seed, size_t count) {
    if (count < 16) fail("noise tensor too small");
    MT19937 mt{uint32_t(seed)};
    std::vector<float> out(count);
    for (auto &v : out) v = mt.uniform();
    for (size_t start = 0; start + 16 <= count; start += 16) normal_fill16(&out[start]);
    if (count % 16) {
        size_t start = count - 16;
        for (size_t i = start; i < count; ++i) out[i] = mt.uniform();
        normal_fill16(&out[start]);
    }
    return out;
}

// [24,H,W] -> rows [(H/2)*(W/2), 96]; column = c*4 + ph*2 + pw
std::vector<float> patchify(const std::vector<float> &noise, int h, int w) {
    std::vector<float> rows(noise.size());
    const int h2 = h / 2, w2 = w / 2;
    for (int c = 0; c < kLatentChannels; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t r = size_t(y / 2) * w2 + x / 2;
                rows[r * kPatchDim + c * 4 + (y % 2) * 2 + (x % 2)] = noise[(size_t(c) * h + y) * w + x];
            }
    (void)h2;
    return rows;
}

std::vector<float> unpatchify(const std::vector<float> &rows, int h, int w) {
    std::vector<float> out(rows.size());
    const int w2 = w / 2;
    for (int c = 0; c < kLatentChannels; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t r = size_t(y / 2) * w2 + x / 2;
                out[(size_t(c) * h + y) * w + x] = rows[r * kPatchDim + c * 4 + (y % 2) * 2 + (x % 2)];
            }
    return out;
}

bool write_png(const std::string &path, const std::vector<uint8_t> &rgb, int w, int h) {
    stbi_write_png_compression_level = 1;
    std::string tmp = path + ".tmp";
    if (!stbi_write_png(tmp.c_str(), w, h, 3, rgb.data(), w * 3)) return false;
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

std::string encode_png(const std::vector<uint8_t> &rgb, int w, int h) {
    stbi_write_png_compression_level = 1;
    std::string out;
    auto sink = [](void *ctx, void *data, int size) {
        static_cast<std::string *>(ctx)->append(static_cast<const char *>(data), size_t(size));
    };
    if (!stbi_write_png_to_func(sink, &out, w, h, 3, rgb.data(), w * 3)) fail("PNG encoding failed");
    return out;
}

// ------------------------------------------------------------------ Pipeline
Pipeline::Pipeline(const std::string &model_dir, const std::string &lora_dir, int lora_cache) {
    double t0 = now_s();
    rt_ = std::make_unique<Runtime>(0);
    dir_ = std::make_unique<ModelDir>(model_dir);
    tok_ = std::make_unique<Tokenizer>(dir_->file("text_encoder/tokenizer/tokenizer.json"));
    AXD_LOG("loading DiT weights");
    dit_w_ = load_dit_weights(*dir_);
    AXD_LOG("loading text encoder weights");
    text_w_ = load_text_weights(*dir_);
    AXD_LOG("loading VAE weights");
    vae_w_ = load_vae_weights(*dir_);
    AXD_LOG("loading AXModels");
    text_ = std::make_unique<TextEncoder>(*dir_, *text_w_);
    dit_ = std::make_unique<Dit>(*dir_, *dit_w_);
    vae_ = std::make_unique<Vae>(*dir_, *vae_w_);
    if (!lora_dir.empty())
        lora_ = std::make_unique<LoraManager>(lora_dir, dir_->file("dit/shared.safetensors"), lora_cache);
    load_s_ = now_s() - t0;
    size_t used = 0, total = 0;
    rt_->cmm(used, total);
    AXD_LOG("ready in %.1f s, CMM %.2f / %.2f GiB", load_s_, used / 1073741824.0, total / 1073741824.0);
}

Pipeline::~Pipeline() {
    // Release NPU objects before the runtime.
    lora_.reset();
    vae_.reset();
    dit_.reset();
    text_.reset();
    vae_w_.reset();
    text_w_.reset();
    dit_w_.reset();
    rt_.reset();
}

namespace {
template <typename T>
void dump(const std::string &dir, const std::string &name, const std::vector<T> &v) {
    if (dir.empty()) return;
    std::ofstream f(dir + "/" + name, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
}
}  // namespace

GenerateResult Pipeline::generate(const GenerateRequest &req) {
    if (req.size != 1024 && req.size != 2048) fail("size must be 1024 or 2048");
    if (req.prompt.empty()) fail("prompt is empty");
    rt_->bind_thread();
    GenerateResult r;
    double t_all = now_s();
    const LoraSet *lora = nullptr;
    if (!req.loras.empty()) {
        if (!lora_) fail("LoRA requested but the server was started without a LoRA directory");
        double tl = now_s();
        lora = &lora_->get(req.loras);
        r.lora_s = now_s() - tl;
    }

    double t0 = now_s();
    r.ids = tok_->encode(req.prompt);
    if (r.ids.empty()) fail("prompt produced no tokens");
    if (r.ids.size() > size_t(kMaxTextTokens)) {
        r.ids.resize(kMaxTextTokens);
        r.truncated = true;
    }
    r.tokenize_s = now_s() - t0;

    t0 = now_s();
    auto hidden = text_->encode(r.ids);
    r.text_s = now_s() - t0;
    dump(req.dump_dir, "text_hidden.bf16", hidden);

    const int lat = req.size / 16;
    auto noise = torch_randn(req.seed, size_t(kLatentChannels) * lat * lat);
    auto sample = patchify(noise, lat, lat);

    Dit::Timing dt;
    dit_->run(hidden, int(r.ids.size()), lat, lat, sample, req.steps, req.blocks, dt, lora);
    r.dit_prepare_s = dt.prepare;
    r.dit_steps_s = dt.steps;
    r.step_seconds = dt.step_seconds;
    double sum = 0;
    for (double b : dt.block_seconds) sum += b;
    r.block_mean_s = dt.block_seconds.empty() ? 0 : sum / dt.block_seconds.size();
    r.used = dit_->used();
    r.padded = dit_->padded();
    auto latents = unpatchify(sample, lat, lat);
    dump(req.dump_dir, "latents.f32", latents);

    r.rgb = vae_->decode(latents, lat, lat, r.vae_s);
    r.width = r.height = req.size;

    if (!req.out_png.empty()) {
        t0 = now_s();
        if (!write_png(req.out_png, r.rgb, r.width, r.height)) fail("cannot write " + req.out_png);
        r.png_s = now_s() - t0;
    }
    r.total_s = now_s() - t_all;
    return r;
}

}  // namespace axd
