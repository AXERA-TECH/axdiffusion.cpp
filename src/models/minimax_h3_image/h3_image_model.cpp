// MiniMax-H3-Image plugin: tokenizer -> Qwen3-VL text encoder (NPU) -> H3 DiT (NPU) -> image VAE (NPU).
#include <fstream>

#include "common.hpp"
#include "edit_host.hpp"
#include "h3_dit.hpp"
#include "h3_image_vae.hpp"
#include "lora.hpp"
#include "model.hpp"
#include "model_dir.hpp"
#include "qwen_text_encoder.hpp"
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
        i.sizes = {{1024, 1024}, {2048, 2048}};
        i.max_prompt_tokens = kMaxTextTokens;
        i.max_references = 0;
        i.lora = lora_ != nullptr;
        return i;
    }

    std::vector<std::string> loras() const override { return lora_ ? lora_->available() : std::vector<std::string>{}; }
    double load_seconds() const override { return load_s_; }
    Runtime &runtime() override { return *rt_; }

    Result run(const Request &req, Progress &progress) override {
        if (req.task == Task::edit || !req.images.empty())
            fail("reference-image editing is not available with this model directory yet (edit kernels missing)");
        if (req.task != Task::txt2img) fail(std::string("task ") + task_name(req.task) + " is not supported by this model");
        if (req.width != req.height || (req.width != 1024 && req.width != 2048))
            fail("supported sizes: 1024x1024, 2048x2048");
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
        if (r.token_ids.size() > size_t(kMaxTextTokens)) {
            r.token_ids.resize(kMaxTextTokens);
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
        double sum = 0;
        for (double b : dt.block_seconds) sum += b;
        r.metrics["block_mean_s"] = dt.block_seconds.empty() ? 0 : sum / dt.block_seconds.size();
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

}  // namespace

std::unique_ptr<ModelPlugin> create_minimax_h3_image(const std::string &dir, const ModelOptions &opt) {
    return std::make_unique<H3ImageModel>(dir, opt);
}

}  // namespace axd
