// MiniMax-H3-Image plugin: tokenizer -> Qwen3-VL text encoder (NPU) -> H3 DiT (NPU) -> image VAE (NPU).
#include <algorithm>
#include <cmath>
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
        double tp = now_s();
        auto phase = [&](const char *what) {
            AXD_LOG("%s: %.1f s", what, now_s() - tp);
            tp = now_s();
        };
        dit_w_ = load_dit_weights(*dir_);
        phase("DiT weights");
        text_w_ = load_text_weights(*dir_);
        phase("text encoder weights");
        vae_w_ = load_vae_weights(*dir_);
        phase("VAE weights");
        text_ = std::make_unique<TextEncoder>(*dir_, *text_w_);
        dit_ = std::make_unique<Dit>(*dir_, *dit_w_);
        vae_ = std::make_unique<Vae>(*dir_, *vae_w_);
        phase("text/DiT/VAE kernels");
        has_edit_kernels_ = VisionEncoder::available(*dir_) && dir_->has_axmodel("image-vae-encode");
        if (has_edit_kernels_) {
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
        for (auto &wh : preset_sizes()) i.sizes.push_back({wh[0], wh[1]});
        i.max_prompt_tokens = std::min(text_->max_tokens(), dit_->max_text_rows(image_rows(1024, 1024)));
        i.max_references = edit_ready() ? 3 : 0;
        i.lora = lora_ != nullptr;
        if (refine_ready()) {
            i.refine_steps = {1, 2};
            i.refine_min_pixels = kRefineMinPixels;
        }
        return i;
    }

    // Tokens the prompt may use. Editing spends part of the text encoder's context on the reference labels and
    // vision tokens, and the DiT sequence also carries the reference and target rows.
    int edit_prompt_limit(int overhead, int reference_rows, int width, int height) const {
        return std::max(0, std::min(text_->max_tokens() - overhead,
                                    dit_->max_edit_rows() - overhead - reference_rows - image_rows(width, height)));
    }

    PromptBudget prompt_budget(const std::string &prompt, int width, int height,
                               const std::vector<std::array<int, 2>> &reference_sizes) const override {
        PromptBudget b;
        if (reference_sizes.empty()) {
            b.limit = std::max(0, std::min(text_->max_tokens(), dit_->max_text_rows(image_rows(width, height))));
        } else {
            int overhead = 0, reference_rows = 0;
            for (size_t k = 0; k < reference_sizes.size(); ++k) {
                auto vit = edit::vit_size(reference_sizes[k][0], reference_sizes[k][1]);
                auto vae = edit::vae_size(reference_sizes[k][0], reference_sizes[k][1]);
                overhead += int(tok_->encode("Picture " + std::to_string(k + 1) + ": ").size()) + 2 +
                            (vit[0] / edit::kVitPatch) * (vit[1] / edit::kVitPatch) / (edit::kVitMerge * edit::kVitMerge);
                reference_rows += (vae[0] / 32) * (vae[1] / 32);
            }
            b.limit = edit_prompt_limit(overhead, reference_rows, width, height);
        }
        auto count = [&](const std::string &s) { return int(tok_->encode(s).size()); };
        b.tokens = count(prompt);
        if (b.tokens > b.limit) {
            // Longest prefix, cut on a code point, that still fits: token count grows with the prefix.
            std::vector<size_t> starts;
            for (size_t i = 0; i < prompt.size(); ++i)
                if ((static_cast<unsigned char>(prompt[i]) & 0xC0) != 0x80) starts.push_back(i);
            size_t lo = 0, hi = starts.size();  // code points kept
            while (lo < hi) {
                size_t mid = (lo + hi + 1) / 2;
                size_t bytes = mid < starts.size() ? starts[mid] : prompt.size();
                if (count(prompt.substr(0, bytes)) <= b.limit) lo = mid;
                else hi = mid - 1;
            }
            b.cut_chars = int(lo);
        }
        return b;
    }

    // Image rows of a W x H output (patch 2x2 on the 16x latent).
    static int image_rows(int w, int h) { return (w / 32) * (h / 32); }

    // Common sizes (aspect ratios x pixel budgets) that the installed DiT tiers cover with a typical
    // (64-token) prompt; any other W x H (multiples of 32) is accepted when its sequence fits a tier.
    std::vector<std::array<int, 2>> preset_sizes() const {
        static const int ratios[][2] = {{1, 1}, {4, 3}, {3, 4}, {3, 2}, {2, 3}, {16, 9}, {9, 16}, {21, 9}, {9, 21}};
        static const double mpix[] = {0.25, 1.0, 2.0, 4.0, 16.0};
        std::vector<std::array<int, 2>> out;
        for (double mp : mpix)
            for (auto &r : ratios) {
                const double area = mp * 1024 * 1024;
                int w = int(std::lround(std::sqrt(area * r[0] / r[1]) / 32.0)) * 32;
                int h = int(std::lround(std::sqrt(area * r[1] / r[0]) / 32.0)) * 32;
                if (w < 256 || h < 256) continue;
                if (dit_->pick_tier(64 + image_rows(w, h), 0, false)) out.push_back({w, h});
            }
        return out;
    }

    void check_size(const Request &req) const {
        if (req.width % 32 || req.height % 32 || req.width < 256 || req.height < 256)
            fail("width and height must be multiples of 32 and at least 256 (got " + std::to_string(req.width) + "x" +
                 std::to_string(req.height) + ")");
        if (dit_->max_text_rows(image_rows(req.width, req.height)) <= 0)
            fail("no DiT kernel tier covers " + std::to_string(req.width) + "x" + std::to_string(req.height) +
                 " (see GET /health sizes for covered presets)");
    }

    std::vector<std::string> loras() const override { return lora_ ? lora_->available() : std::vector<std::string>{}; }
    double load_seconds() const override { return load_s_; }
    Runtime &runtime() override { return *rt_; }

    // Edit needs the edit DiT kernels + AdaLN table, the paged deepstack text kernels, the VAE encoder and vision
    // features (the ViT kernels). Those two are large files the engine takes ~13 s to load, and it cannot load them
    // while other threads copy to the device, so they are loaded by the first edit request.
    bool edit_ready() const {
        return dit_->supports_edit() && text_->supports_vision() && has_edit_kernels_;
    }

    // Image (multiple of 32 pixels) -> VAE posterior moments [48, h/16, w/16] from 256 px tiles. Reference images
    // use the official minimum overlap (64 px, blended). Refine re-noises the result, where tiles without overlap
    // leave no visible seam and need about half the launches.
    std::vector<float> encode_moments(const Image &vimg, int &lh, int &lw, int min_overlap = 64) {
        int h = 0, w = 0;
        auto px = edit::vae_pixels(vimg, h, w);  // [3,h,w]
        std::vector<int> ys, yo, xs, xo;
        edit::split_tiles(h, kVaeEncTile, min_overlap, 16, ys, yo);
        edit::split_tiles(w, kVaeEncTile, min_overlap, 16, xs, xo);
        std::vector<std::vector<float>> tiles;
        for (int ty : ys)
            for (int tx : xs) {
                if (ty + kVaeEncTile > h || tx + kVaeEncTile > w) fail("image smaller than one VAE tile");
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
        lh = h / 16;
        lw = w / 16;
        return edit::stitch_tiles(moments_tiles, kVaeEncMoments, kVaeEncTileLatent, kVaeEncTileLatent, yo, xo, lh, lw);
    }

    // Two-pass refine: the base image is half the requested size (rounded to 32 px).
    static constexpr long kRefineMinPixels = 3L * 1024 * 1024;
    static int half32(int v) { return std::max(256, (v / 2 + 16) / 32 * 32); }
    bool refine_ready() const { return has_edit_kernels_; }  // needs the VAE encoder

    Result run(const Request &req, Progress &progress) override {
        if (req.refine_steps) {
            if (req.task != Task::txt2img || !req.images.empty()) fail("refine_steps is for text-to-image only");
            if (!refine_ready()) fail("refine needs the image-vae-encode kernel, which this model directory does not have");
            if (req.refine_steps != 1 && req.refine_steps != 2) fail("refine_steps must be 0, 1 or 2");
            check_size(req);
            if (long(req.width) * req.height < kRefineMinPixels)
                fail("refine_steps needs an output of at least 3 megapixels (for example 2048x1536); smaller images are "
                     "generated directly");
            Request base = req;
            base.refine_steps = 0;
            base.width = half32(req.width);
            base.height = half32(req.height);
            Result rb = run_txt2img(base, progress, Pass::base, nullptr, 0);
            Request second = req;
            second.refine_steps = 0;
            second.steps = 0;  // the refine pass always ends on the last step
            second.seed = req.seed + 1;  // fresh noise: the base image is already a function of the first draw
            Result r = run_txt2img(second, progress, Pass::refine, &rb.images[0], req.refine_steps);
            for (auto &kv : rb.seconds) r.seconds["base_" + kv.first] = kv.second;
            r.seconds["total"] += rb.seconds["total"];
            r.metrics["base_width"] = base.width;
            r.metrics["base_height"] = base.height;
            r.metrics["refine_steps"] = req.refine_steps;
            return r;
        }
        if (req.task == Task::edit || !req.images.empty()) return run_edit(req, progress);
        return run_txt2img(req, progress, Pass::single, nullptr, 0);
    }

    // Progress stages: a single pass reports text_encoder / denoise / vae. In a two-pass request the base pass
    // reports text_encoder / denoise / base_vae and the refine pass refine_encode / refine / vae, so observers
    // see every stage once and in order.
    enum class Pass { single, base, refine };

    // `init` (refine pass): the image to start from; it is resized to the request size, encoded, mixed with noise
    // at the sigma of step 6 - refine_n, and only the last refine_n steps run.
    Result run_txt2img(const Request &req, Progress &progress, Pass pass, const Image *init, int refine_n) {
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
            if (pass != Pass::refine) progress("lora");
            double tl = now_s();
            lora = &lora_->get(req.loras);
            r.seconds["lora_build"] = now_s() - tl;
            r.metrics["lora_rank"] = lora->rank;
        }

        if (pass != Pass::refine) progress("text_encoder");
        double t0 = now_s();
        r.token_ids = tok_->encode(req.prompt);
        if (r.token_ids.empty()) fail("prompt produced no tokens");
        const int max_text =
            std::min(text_->max_tokens(), dit_->max_text_rows(image_rows(req.width, req.height), lora ? lora->rank : 0));
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

        const int lat_h = req.height / 16, lat_w = req.width / 16;
        TorchGenerator gen(req.seed);
        auto noise = gen.randn(size_t(kLatentChannels) * lat_h * lat_w);
        int first_step = 0;
        if (init) {  // refine: x = (1 - sigma) * x0 + sigma * noise
            first_step = kSteps - refine_n;
            progress("refine_encode");
            t0 = now_s();
            if (!vae_enc_) vae_enc_ = std::make_unique<VaeEncoder>(*dir_);
            Image src = *init;
            if (src.width != req.width || src.height != req.height)
                src = edit::resize_lanczos(src, req.width, req.height);
            int lh = 0, lw = 0;
            auto moments = encode_moments(src, lh, lw, 0);
            std::vector<float> zero(size_t(kLatentChannels) * lh * lw, 0.f);  // posterior mean
            auto x0 = edit::posterior_latent(moments, zero, lh, lw, latents_mean_.data(), latents_std_.data());
            r.seconds["vae_encode"] = now_s() - t0;
            if (x0.size() != noise.size()) fail("init latent size mismatch");
            const float sg = kSigmas[first_step];
            for (size_t i = 0; i < noise.size(); ++i) noise[i] = (1.0f - sg) * x0[i] + sg * noise[i];
        }
        auto sample = patchify(noise, lat_h, lat_w);

        Dit::Timing dt;
        const char *stage = pass == Pass::refine ? "refine" : "denoise";
        progress(stage, 0, steps - first_step);  // the stage is visible while its first step runs
        dit_->run(hidden, int(r.token_ids.size()), lat_h, lat_w, sample, steps, blocks, dt, lora, &progress, nullptr,
                  first_step, stage);
        r.seconds["dit"] = dt.prepare + dt.steps;
        r.seconds["dit_prepare"] = dt.prepare;
        // wall time per block (with async launches the per-launch host times only cover binding/submission)
        r.metrics["block_mean_s"] = dt.block_seconds.empty() ? 0 : dt.steps / double(dt.block_seconds.size());
        r.metrics["tokens"] = double(r.token_ids.size());
        r.metrics["sequence"] = dit_->used();
        r.metrics["padded"] = dit_->padded();
        auto latents = unpatchify(sample, lat_h, lat_w);
        dump(req.dump_dir, "latents.f32", latents);

        progress(pass == Pass::base ? "base_vae" : "vae");
        Image img;
        double vae_s = 0;
        img.rgb = vae_->decode(latents, lat_h, lat_w, vae_s);
        img.width = req.width;
        img.height = req.height;
        r.seconds["vae"] = vae_s;
        r.images.push_back(std::move(img));
        r.seconds["total"] = now_s() - t_all;
        return r;
    }

    Result run_edit(const Request &req, Progress &progress) {
        if (!edit_ready()) fail("reference-image editing needs edit kernels that this model directory does not have");
        if (req.images.empty()) fail("edit needs at least one reference image");
        if (req.images.size() > 9) fail("at most 9 reference images");
        check_size(req);
        if (req.prompt.empty()) fail("prompt is empty");
        const int steps = req.steps ? req.steps : kSteps;
        const int blocks = req.blocks ? req.blocks : kDitLayers;
        rt_->bind_thread();
        Result r;
        double t_all = now_s();
        if (!vit_) {
            progress("load_edit");
            vit_ = std::make_unique<VisionEncoder>(*dir_);
            if (!vae_enc_) vae_enc_ = std::make_unique<VaeEncoder>(*dir_);
            r.seconds["edit_load"] = now_s() - t_all;
            AXD_LOG("edit kernels (ViT, VAE encoder): %.1f s", r.seconds["edit_load"]);
        }
        const int n_ref = int(req.images.size());
        TorchGenerator gen(req.seed);
        const LoraSet *lora = nullptr;
        if (!req.loras.empty()) {
            if (!lora_) fail("LoRA requested but no LoRA directory was configured");
            progress("lora");
            double tl = now_s();
            lora = &lora_->get(req.loras);
            r.seconds["lora_build"] = now_s() - tl;
            r.metrics["lora_rank"] = lora->rank;
        }

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
            int lh = 0, lw = 0;
            auto moments = encode_moments(vimg, lh, lw);
            auto eps = gen.randn(size_t(kLatentChannels) * lh * lw);
            ref_latents[k] = edit::posterior_latent(moments, eps, lh, lw, latents_mean_.data(), latents_std_.data());
            ref_grids[k] = {lh, lw};
            dump(req.dump_dir, "reference_latent_" + std::to_string(k) + ".f32", ref_latents[k]);
        }
        r.seconds["vae_encode"] = now_s() - t0;

        // 2) generator order after the posterior draws: target noise, then one noise per reference
        const int lat_h = req.height / 16, lat_w = req.width / 16;
        auto noise = gen.randn(size_t(kLatentChannels) * lat_h * lat_w);
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
        r.seconds["vision"] = now_s() - t0;

        // 4) presentation + paged text encoder with vision rows and deepstack
        progress("text_encoder");
        t0 = now_s();
        std::vector<std::vector<int32_t>> labels(n_ref);
        for (int k = 0; k < n_ref; ++k) labels[k] = tok_->encode("Picture " + std::to_string(k + 1) + ": ");
        auto prompt_ids = tok_->encode(req.prompt);
        {
            const int overhead = int(edit::make_presentation(labels, {}, grids).ids.size());
            const int limit = edit_prompt_limit(overhead, int(de.reference_rows.size() / kPatchDim), req.width, req.height);
            if (limit <= 0) fail("the reference images leave no room for a prompt at this output size");
            if (int(prompt_ids.size()) > limit) {
                prompt_ids.resize(limit);
                r.truncated = true;
            }
        }
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
        auto layout = edit::packed_layout(pres.token_tags, ref_grids, lat_h, lat_w);
        de.positions = layout.position_ids;
        auto sample = patchify(noise, lat_h, lat_w);
        Dit::Timing dt;
        dit_->run(hidden, text_rows, lat_h, lat_w, sample, steps, blocks, dt, lora, &progress, &de);
        r.seconds["dit"] = dt.prepare + dt.steps;
        r.seconds["dit_prepare"] = dt.prepare;
        r.metrics["tokens"] = double(text_rows);
        r.metrics["references"] = double(n_ref);
        r.metrics["sequence"] = dit_->used();
        r.metrics["padded"] = dit_->padded();
        auto latents = unpatchify(sample, lat_h, lat_w);
        dump(req.dump_dir, "latents.f32", latents);

        // 6) decode
        progress("vae");
        Image img;
        double vae_s = 0;
        img.rgb = vae_->decode(latents, lat_h, lat_w, vae_s);
        img.width = req.width;
        img.height = req.height;
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
    bool has_edit_kernels_ = false;
    std::vector<float> latents_mean_, latents_std_;
    std::unique_ptr<LoraManager> lora_;
    double load_s_ = 0;
};

}  // namespace

std::unique_ptr<ModelPlugin> create_minimax_h3_image(const std::string &dir, const ModelOptions &opt) {
    return std::make_unique<H3ImageModel>(dir, opt);
}

}  // namespace axd
