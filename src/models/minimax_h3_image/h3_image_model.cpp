// MiniMax-H3-Image plugin: tokenizer -> Qwen3-VL text encoder (NPU) -> H3 DiT (NPU) -> image VAE (NPU).
#include <algorithm>
#include <cstdlib>
#include <fstream>

#include "common.hpp"
#include "edit_host.hpp"
#include "h3_dit.hpp"
#include "h3_image_vae.hpp"
#include "h3_image_vae_encoder.hpp"
#include "json.hpp"
#include "lora.hpp"
#include "model.hpp"
#include "model_dir.hpp"
#include "qwen_text_encoder.hpp"
#include "qwen_vision.hpp"
#include "random.hpp"
#include "tokenizer.hpp"

namespace axd {

namespace {

// Raw little-endian file -> vector (used for debug inputs such as precomputed vision features).
template <typename T>
std::vector<T> read_raw(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) fail("cannot read " + path);
    std::vector<T> v(size_t(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}

template <typename T>
void dump(const std::string &dir, const std::string &name, const std::vector<T> &v) {
    if (dir.empty()) return;
    std::ofstream f(dir + "/" + name, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
}

class H3ImageModel : public ModelPlugin {
public:
    H3ImageModel(const std::string &model_dir, const ModelOptions &opt) {
        double t0 = now_s();
        rt_ = std::make_unique<Runtime>(opt.device);
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
        if (dir_->has_axmodel("vision-vit")) vit_ = std::make_unique<VisionEncoder>(*dir_);
        if (dir_->has_axmodel("image-vae-encode")) {
            vae_enc_ = std::make_unique<VaeEncoder>(*dir_);
            std::ifstream cf(dir_->file("vae/config.json"));
            auto cfg = nlohmann::json::parse(cf);
            latents_mean_ = cfg.at("latents_mean").get<std::vector<float>>();
            latents_std_ = cfg.at("latents_std").get<std::vector<float>>();
            if (latents_mean_.size() != size_t(kLatentChannels) || latents_std_.size() != size_t(kLatentChannels))
                fail("vae/config.json: latents_mean/std must have 24 entries");
        }
        if (!opt.lora_dir.empty())
            lora_ = std::make_unique<LoraManager>(opt.lora_dir, dir_->file("dit/shared.safetensors"), opt.lora_cache,
                                                  dit_->lora_ranks());
        load_s_ = now_s() - t0;
        size_t used = 0, total = 0;
        rt_->cmm(used, total);
        AXD_LOG("ready in %.1f s, CMM %.2f / %.2f GiB", load_s_, used / 1073741824.0, total / 1073741824.0);
    }

    ~H3ImageModel() override {
        // NPU objects go before the runtime.
        lora_.reset();
        vit_.reset();
        vae_enc_.reset();
        vae_.reset();
        dit_.reset();
        text_.reset();
        vae_w_.reset();
        text_w_.reset();
        dit_w_.reset();
        rt_.reset();
    }

    ModelInfo info() const override {
        ModelInfo i;
        i.arch = "minimax-h3-image";
        i.name = "minimax-h3-image";
        i.tasks = {Task::txt2img};
        if (edit_ready()) i.tasks.push_back(Task::edit);
        for (int sz : supported_sizes()) i.sizes.push_back({sz, sz});
        i.max_prompt_tokens = std::min(text_->max_tokens(), dit_->max_text_rows(64 * 64 / 4));
        i.max_references = edit_ready() ? 3 : 0;
        i.lora = lora_ != nullptr;
        return i;
    }

    // Square sizes whose image rows fit at least one DiT tier (the manifest decides: 512/1024/2048/4096 ...).
    std::vector<int> supported_sizes() const {
        std::vector<int> out;
        for (int sz : {512, 768, 1024, 1536, 2048, 3072, 4096}) {
            const int lat = sz / 16;
            if (dit_->max_text_rows(lat * lat / 4) > 0) out.push_back(sz);
        }
        return out;
    }
    void check_size(const Request &req) const {
        auto sizes = supported_sizes();
        if (req.width == req.height && std::find(sizes.begin(), sizes.end(), req.width) != sizes.end()) return;
        std::string have;
        for (int sz : sizes) have += " " + std::to_string(sz) + "x" + std::to_string(sz);
        fail("unsupported size " + std::to_string(req.width) + "x" + std::to_string(req.height) + " (supported:" + have + ")");
    }

    std::vector<std::string> loras() const override { return lora_ ? lora_->available() : std::vector<std::string>{}; }
    double load_seconds() const override { return load_s_; }
    Runtime &runtime() override { return *rt_; }

    // Edit needs the edit DiT kernels + AdaLN table, the paged deepstack text kernels, the VAE encoder and vision
    // features (the ViT kernel, or AXD_EDIT_VISION=<dir> with precomputed features for testing).
    bool edit_ready() const {
        return dit_->supports_edit() && text_->supports_vision() && vae_enc_ &&
               (vit_ || std::getenv("AXD_EDIT_VISION"));
    }

    Result run(const Request &req, Progress &progress) override {
        if (req.task == Task::edit || !req.images.empty()) return run_edit(req, progress);
        if (req.task != Task::txt2img) fail(std::string("task ") + task_name(req.task) + " is not supported by this model");
        check_size(req);
        if (req.prompt.empty()) fail("prompt is empty");
        const int steps = req.steps ? req.steps : kSteps;
        const int blocks = req.blocks ? req.blocks : kDitLayers;
        rt_->bind_thread();
        Result r;
        double t_all = now_s();

        const LoraSet *lora = nullptr;
        if (!req.loras.empty()) {
            if (!lora_) fail("LoRA requested but no LoRA directory was configured");
            progress("lora");
            double tl = now_s();
            lora = &lora_->get(req.loras);
            r.seconds["lora_build"] = now_s() - tl;
            r.metrics["lora_rank"] = lora->rank;
        }

        progress("text_encoder");
        double t0 = now_s();
        r.token_ids = tok_->encode(req.prompt);
        if (r.token_ids.empty()) fail("prompt produced no tokens");
        const int lat0 = req.width / 16;
        const int max_text = std::min(text_->max_tokens(), dit_->max_text_rows(lat0 * lat0 / 4, lora ? lora->rank : 0));
        if (max_text <= 0) fail("no DiT kernel tier for this resolution");
        if (r.token_ids.size() > size_t(max_text)) {
            r.token_ids.resize(max_text);
            r.truncated = true;
        }
        r.seconds["tokenize"] = now_s() - t0;
        t0 = now_s();
        auto hidden = text_->encode(r.token_ids);
        r.seconds["text_encoder"] = now_s() - t0;
        dump(req.dump_dir, "text_hidden.bf16", hidden);

        const int lat = req.width / 16;
        TorchGenerator gen(req.seed);
        auto noise = gen.randn(size_t(kLatentChannels) * lat * lat);
        auto sample = patchify(noise, lat, lat);

        Dit::Timing dt;
        dit_->run(hidden, int(r.token_ids.size()), lat, lat, sample, steps, blocks, dt, lora, &progress);
        r.seconds["dit"] = dt.prepare + dt.steps;
        // wall time per block (with async launches the per-launch host times only cover binding/submission)
        r.metrics["block_mean_s"] = dt.block_seconds.empty() ? 0 : dt.steps / double(dt.block_seconds.size());
        r.metrics["tokens"] = double(r.token_ids.size());
        r.metrics["sequence"] = dit_->used();
        r.metrics["padded"] = dit_->padded();
        auto latents = unpatchify(sample, lat, lat);
        dump(req.dump_dir, "latents.f32", latents);

        progress("vae");
        Image img;
        double vae_s = 0;
        img.rgb = vae_->decode(latents, lat, lat, vae_s);
        img.width = img.height = req.width;
        r.seconds["vae"] = vae_s;
        r.images.push_back(std::move(img));
        r.seconds["total"] = now_s() - t_all;
        return r;
    }

    Result run_edit(const Request &req, Progress &progress) {
        if (!edit_ready()) fail("reference-image editing needs edit kernels that this model directory does not have");
        if (req.images.empty()) fail("edit needs at least one reference image");
        if (req.images.size() > 9) fail("at most 9 reference images");
        if (!req.loras.empty()) fail("LoRA is not supported together with reference editing yet");
        check_size(req);
        if (req.prompt.empty()) fail("prompt is empty");
        const int steps = req.steps ? req.steps : kSteps;
        const int blocks = req.blocks ? req.blocks : kDitLayers;
        rt_->bind_thread();
        Result r;
        double t_all = now_s();
        const int n_ref = int(req.images.size());
        TorchGenerator gen(req.seed);

        // 1) references -> VAE latents (posterior sample, fp16 rounding, normalization)
        progress("vae_encode");
        double t0 = now_s();
        std::vector<std::vector<float>> ref_latents(n_ref);
        std::vector<std::array<int, 2>> ref_grids(n_ref);
        std::vector<Image> vit_copies(n_ref);
        for (int k = 0; k < n_ref; ++k) {
            const Image &src = req.images[k];
            if (std::min(src.width, src.height) < 32) fail("reference image is too small");
            Image vimg = edit::resize_for_vae(src);
            vit_copies[k] = edit::resize_for_vit(src);
            int h = 0, w = 0;
            auto px = edit::vae_pixels(vimg, h, w);  // [3,h,w]
            std::vector<int> ys, yo, xs, xo;
            edit::split_tiles(h, kVaeEncTile, 64, 16, ys, yo);
            edit::split_tiles(w, kVaeEncTile, 64, 16, xs, xo);
            std::vector<std::vector<float>> tiles;
            for (int ty : ys)
                for (int tx : xs) {
                    if (ty + kVaeEncTile > h || tx + kVaeEncTile > w) fail("reference smaller than one VAE tile");
                    std::vector<float> t(size_t(3) * kVaeEncTile * kVaeEncTile);
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < kVaeEncTile; ++y)
                            std::copy_n(&px[(size_t(c) * h + ty + y) * w + tx], kVaeEncTile,
                                        &t[(size_t(c) * kVaeEncTile + y) * kVaeEncTile]);
                    tiles.push_back(std::move(t));
                }
            auto moments_tiles = vae_enc_->encode_tiles(tiles);
            for (auto &v : yo) v /= 16;
            for (auto &v : xo) v /= 16;
            const int lh = h / 16, lw = w / 16;
            auto moments = edit::stitch_tiles(moments_tiles, kVaeEncMoments, kVaeEncTileLatent, kVaeEncTileLatent, yo,
                                              xo, lh, lw);
            auto eps = gen.randn(size_t(kLatentChannels) * lh * lw);
            ref_latents[k] = edit::posterior_latent(moments, eps, lh, lw, latents_mean_.data(), latents_std_.data());
            ref_grids[k] = {lh, lw};
            dump(req.dump_dir, "reference_latent_" + std::to_string(k) + ".f32", ref_latents[k]);
        }
        r.seconds["vae_encode"] = now_s() - t0;

        // 2) generator order after the posterior draws: target noise, then one noise per reference
        const int lat = req.width / 16;
        auto noise = gen.randn(size_t(kLatentChannels) * lat * lat);
        DitEdit de;
        for (int k = 0; k < n_ref; ++k) {
            auto rn = gen.randn(ref_latents[k].size());
            edit::noise_reference(ref_latents[k], rn);
            auto rows = patchify(ref_latents[k], ref_grids[k][0], ref_grids[k][1]);
            de.reference_rows.insert(de.reference_rows.end(), rows.begin(), rows.end());
        }

        // 3) vision features [n_ref*tokens, 5120] x 4 (merger, deepstack 0..2)
        progress("vision");
        t0 = now_s();
        std::vector<std::array<int, 3>> grids(n_ref);
        std::vector<std::vector<float>> pixel_values(n_ref);
        for (int k = 0; k < n_ref; ++k) pixel_values[k] = edit::vit_pixel_values(vit_copies[k], grids[k]);
        std::vector<uint16_t> merger, ds[3];
        if (const char *vd = std::getenv("AXD_EDIT_VISION")) {  // testing: precomputed features for these inputs
            merger = read_raw<uint16_t>(std::string(vd) + "/vision_merger.bf16");
            for (int i = 0; i < 3; ++i) ds[i] = read_raw<uint16_t>(std::string(vd) + "/vision_deepstack_" + std::to_string(i) + ".bf16");
        } else {
            for (int k = 0; k < n_ref; ++k) {  // one image per launch chain; features concatenated in reference order
                auto f = vit_->encode(pixel_values[k], grids[k][1], grids[k][2]);
                merger.insert(merger.end(), f.merger.begin(), f.merger.end());
                for (int i = 0; i < 3; ++i) ds[i].insert(ds[i].end(), f.deepstack[i].begin(), f.deepstack[i].end());
                if (!req.dump_dir.empty()) {
                    dump(req.dump_dir, "vision_merger_" + std::to_string(k) + ".bf16", f.merger);
                    for (int i = 0; i < 3; ++i)
                        dump(req.dump_dir, "vision_deepstack" + std::to_string(i) + "_" + std::to_string(k) + ".bf16", f.deepstack[i]);
                }
            }
        }
        r.seconds["vision"] = now_s() - t0;

        // 4) presentation + paged text encoder with vision rows and deepstack
        progress("text_encoder");
        t0 = now_s();
        std::vector<std::vector<int32_t>> labels(n_ref);
        for (int k = 0; k < n_ref; ++k) labels[k] = tok_->encode("Picture " + std::to_string(k + 1) + ": ");
        auto prompt_ids = tok_->encode(req.prompt);
        auto pres = edit::make_presentation(labels, prompt_ids, grids);
        auto mrope = edit::mrope_positions(pres, grids);
        TextVision tv;
        for (int i = 0; i < int(pres.mm_types.size()); ++i)
            if (pres.mm_types[i] == 1) tv.rows.push_back(i);
        if (merger.size() != tv.rows.size() * size_t(kDitTextDim))
            fail("vision features do not match the image tokens (" + std::to_string(merger.size() / kDitTextDim) +
                 " vs " + std::to_string(tv.rows.size()) + ")");
        tv.merger = merger.data();
        for (int i = 0; i < 3; ++i) tv.deepstack[i] = ds[i].data();
        if (int(pres.ids.size()) > text_->max_tokens()) fail("prompt is too long for editing");
        r.token_ids = pres.ids;
        auto hidden = text_->encode(pres.ids, &mrope, &tv);
        r.seconds["text_encoder"] = now_s() - t0;
        dump(req.dump_dir, "text_hidden.bf16", hidden);

        // 5) DiT over [text | references | target]
        const int text_rows = int(pres.ids.size());
        de.text_tags = pres.token_tags;
        auto layout = edit::packed_layout(pres.token_tags, ref_grids, lat, lat);
        de.positions = layout.position_ids;
        auto sample = patchify(noise, lat, lat);
        Dit::Timing dt;
        dit_->run(hidden, text_rows, lat, lat, sample, steps, blocks, dt, nullptr, &progress, &de);
        r.seconds["dit"] = dt.prepare + dt.steps;
        r.metrics["tokens"] = double(text_rows);
        r.metrics["references"] = double(n_ref);
        r.metrics["sequence"] = dit_->used();
        r.metrics["padded"] = dit_->padded();
        auto latents = unpatchify(sample, lat, lat);
        dump(req.dump_dir, "latents.f32", latents);

        // 6) decode
        progress("vae");
        Image img;
        double vae_s = 0;
        img.rgb = vae_->decode(latents, lat, lat, vae_s);
        img.width = img.height = req.width;
        r.seconds["vae"] = vae_s;
        r.images.push_back(std::move(img));
        r.seconds["total"] = now_s() - t_all;
        return r;
    }

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
    std::unique_ptr<VaeEncoder> vae_enc_;
    std::unique_ptr<VisionEncoder> vit_;
    std::vector<float> latents_mean_, latents_std_;
    std::unique_ptr<LoraManager> lora_;
    double load_s_ = 0;
};

}  // namespace

std::unique_ptr<ModelPlugin> create_minimax_h3_image(const std::string &dir, const ModelOptions &opt) {
    return std::make_unique<H3ImageModel>(dir, opt);
}

}  // namespace axd
