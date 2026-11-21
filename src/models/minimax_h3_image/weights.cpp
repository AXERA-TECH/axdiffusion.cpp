// Model directory lookup and one-time loading of all weights into device memory.
#include <cstring>
#include <fstream>

#include "common.hpp"
#include "json.hpp"
#include "pipeline.hpp"

namespace axd {

const float kSigmas[kSteps + 1] = {1.0f, 0.9675f, 0.8875f, 0.75f, 0.5328f, 0.2320f, 0.0f};

// ------------------------------------------------------------------ ModelDir
ModelDir::ModelDir(const std::string &root) : root_(root) {
    std::ifstream f(root + "/axmodels/manifest.json");
    if (!f) fail("not a MiniMax-H3-Image-AX8860 directory (axmodels/manifest.json missing): " + root);
    auto j = nlohmann::json::parse(f);
    for (auto &m : j.at("models"))
        models_.push_back({m.at("purpose").get<std::string>(), m.at("shape").get<std::string>(),
                           m.at("file").get<std::string>()});
}

std::string ModelDir::axmodel(const std::string &purpose, const std::string &shape_prefix) const {
    std::vector<const Entry *> hits, narrowed;
    for (auto &m : models_)
        if (m.purpose == purpose) hits.push_back(&m);
    if (!shape_prefix.empty()) {
        for (auto *m : hits)
            if (m->shape.rfind(shape_prefix, 0) == 0) narrowed.push_back(m);
        if (!narrowed.empty()) hits = narrowed;
    }
    if (hits.size() != 1) fail("expected one AXModel for " + purpose + " / " + shape_prefix);
    return root_ + "/axmodels/" + hits[0]->file;
}

bool ModelDir::has_axmodel(const std::string &purpose, const std::string &shape_prefix) const {
    for (auto &m : models_)
        if (m.purpose == purpose && m.shape.rfind(shape_prefix, 0) == 0) return true;
    return false;
}

// ------------------------------------------------------------------ helpers
namespace {

std::vector<float> to_f32(const TensorView &t) {
    std::vector<float> out(size_t(t.numel()));
    if (t.dtype == "F32") {
        std::memcpy(out.data(), t.data, out.size() * 4);
    } else if (t.dtype == "BF16") {
        auto *p = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t i = 0; i < out.size(); ++i) out[i] = bf16_to_f32(p[i]);
    } else {
        fail("expected a float tensor, got " + t.dtype);
    }
    return out;
}

std::vector<uint16_t> to_bf16(const std::vector<float> &v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = f32_to_bf16(v[i]);
    return out;
}

}  // namespace

// ------------------------------------------------------------------ DiT
Buf DitWeights::adaln_slot(int step, int layer, int slot) const {
    const size_t slot_bytes = 3 * kDitHidden * 2;
    size_t index = (size_t(step) * kDitLayers + layer) * 6 + slot;
    return adaln_blocks.buf().view(index * slot_bytes, slot_bytes);
}
Buf DitWeights::adaln_input(int step) const {
    return adaln->get("adaln_input").view(size_t(step) * 2688 * 2, 2688 * 2);
}
Buf DitWeights::final_shift(int step) const {
    return adaln->get("final_shift").view(size_t(step) * kDitHidden * 2, kDitHidden * 2);
}
Buf DitWeights::final_scale(int step) const {
    return adaln->get("final_scale").view(size_t(step) * kDitHidden * 2, kDitHidden * 2);
}

std::unique_ptr<DitWeights> load_dit_weights(const ModelDir &dir) {
    auto w = std::make_unique<DitWeights>();
    {
        SafeTensors st(dir.file("dit/shared.safetensors"));
        w->shared = std::make_unique<TensorBank>("dit-shared");
        for (auto &name : st.names()) {
            auto &t = st.get(name);
            w->shared->add(name, t.data, t.nbytes);
        }
        w->shared->materialize();
        w->proj_in_w = to_f32(st.get("proj_in.weight"));
        w->proj_in_b = to_f32(st.get("proj_in.bias"));
        w->context_w = to_f32(st.get("context.weight"));
        w->context_b = to_f32(st.get("context.bias"));
    }
    {
        SafeTensors st(dir.file("dit/adaln.safetensors"));
        w->adaln = std::make_unique<TensorBank>("dit-adaln");
        for (const char *name : {"adaln_input", "final_shift", "final_scale"}) {
            auto &t = st.get(name);
            w->adaln->add(name, t.data, t.nbytes);
        }
        w->adaln->materialize();
        auto &blocks = st.get("blocks");  // [6,50,6,3,5376] bf16
        if (blocks.nbytes != size_t(kSteps) * kDitLayers * 6 * 3 * kDitHidden * 2) fail("unexpected adaln table size");
        w->adaln_blocks = DeviceBuffer(blocks.nbytes, "dit-adaln-blocks");
        w->adaln_blocks.buf().upload(blocks.data, blocks.nbytes);
    }
    for (int i = 0; i < kDitLayers; ++i) {
        char rel[64];
        std::snprintf(rel, sizeof(rel), "dit/blocks/block%02d.safetensors", i);
        SafeTensors st(dir.file(rel));
        auto bank = std::make_unique<TensorBank>("dit-block-" + std::to_string(i));
        for (auto &name : st.names()) {
            auto &t = st.get(name);
            bank->add(name, t.data, t.nbytes);
        }
        bank->materialize();
        w->blocks.push_back(std::move(bank));
        if (i % 10 == 9) AXD_LOG("DiT blocks loaded: %d/50", i + 1);
    }
    return w;
}

// ------------------------------------------------------------------ text encoder
std::unique_ptr<TextWeights> load_text_weights(const ModelDir &dir) {
    auto w = std::make_unique<TextWeights>();
    {
        SafeTensors st(dir.file("text_encoder/embed_bf16.safetensors"));
        auto &t = st.get("weight");
        w->embed = DeviceBuffer(t.nbytes, "text-embedding");
        w->embed.buf().upload(t.data, t.nbytes);
    }
    // kernel input name -> tensor name in the layer file
    static const std::pair<const char *, const char *> kMap[] = {
        {"x_norm_w", "attention.w_norm_attn"}, {"w_q", "attention.w_q"},
        {"w_scale_q", "attention.w_scale_q"}, {"w_scale2_q", "attention.w_scale2_q"},
        {"w_zeropoint_q", "attention.w_zeropoint_q"}, {"w_k", "attention.w_k"},
        {"w_scale_k", "attention.w_scale_k"}, {"w_scale2_k", "attention.w_scale2_k"},
        {"w_zeropoint_k", "attention.w_zeropoint_k"}, {"w_v", "attention.w_v"},
        {"w_scale_v", "attention.w_scale_v"}, {"w_scale2_v", "attention.w_scale2_v"},
        {"w_zeropoint_v", "attention.w_zeropoint_v"}, {"q_norm_w", "attention.w_norm_q"},
        {"k_norm_w", "attention.w_norm_k"}, {"w_o", "attention.w_o"},
        {"w_scale_o", "attention.w_scale_o"}, {"w_scale2_o", "attention.w_scale2_o"},
        {"w_zeropoint_o", "attention.w_zeropoint_o"}, {"w_norm", "mlp.w_norm"},
        {"w1", "mlp.w1"}, {"w_scale1", "mlp.w_scale1"}, {"w_scale2_1", "mlp.w_scale2_1"},
        {"w2", "mlp.w2"}, {"w_scale2", "mlp.w_scale2"}, {"w_scale2_2", "mlp.w_scale2_2"},
        {"w3", "mlp.w3"}, {"w_scale3", "mlp.w_scale3"}, {"w_scale2_3", "mlp.w_scale2_3"},
        {"w_zp_scale1", "mlp.w_zp_scale1"}, {"w_zp_scale2", "mlp.w_zp_scale2"},
        {"w_zp_scale3", "mlp.w_zp_scale3"}, {"input_permutation", "mlp.input_permutation"},
    };
    for (int i = 0; i < kTextLayers; ++i) {
        char rel[64];
        std::snprintf(rel, sizeof(rel), "text_encoder/layers/layer_%03d.safetensors", i);
        SafeTensors st(dir.file(rel));
        auto bank = std::make_unique<TensorBank>("text-layer-" + std::to_string(i));
        for (auto &kv : kMap) {
            auto &t = st.get(kv.second);
            bank->add(kv.first, t.data, t.nbytes);
        }
        bank->materialize();
        w->layers.push_back(std::move(bank));
        if (i % 10 == 9) AXD_LOG("text layers loaded: %d/50", i + 1);
    }
    return w;
}

// ------------------------------------------------------------------ VAE
std::unique_ptr<VaeWeights> load_vae_weights(const ModelDir &dir) {
    constexpr int C = kLatentChannels, H = 1024, HEADS = 16, HD = 64, LAYERS = 24;
    SafeTensors st(dir.file("vae/vae_bf16.safetensors"));
    auto w = std::make_unique<VaeWeights>();
    w->bank = std::make_unique<TensorBank>("vae");
    // Prepared tensors must outlive materialize().
    std::vector<std::vector<uint8_t>> keep;
    auto put_bf16 = [&](const std::string &name, const std::vector<float> &v) {
        auto b = to_bf16(v);
        keep.emplace_back(reinterpret_cast<uint8_t *>(b.data()), reinterpret_cast<uint8_t *>(b.data()) + b.size() * 2);
        w->bank->add(name, keep.back().data(), keep.back().size());
    };
    auto put_f32 = [&](const std::string &name, const std::vector<float> &v) {
        keep.emplace_back(reinterpret_cast<const uint8_t *>(v.data()),
                          reinterpret_cast<const uint8_t *>(v.data()) + v.size() * 4);
        w->bank->add(name, keep.back().data(), keep.back().size());
    };
    auto f = [&](const std::string &name) { return to_f32(st.get(name)); };
    keep.reserve(16 + LAYERS * 12);

    put_bf16("post_quant_weight", f("post_quant_conv.weight"));  // [24,24,1,1] -> [24,24]
    put_bf16("post_quant_bias", f("post_quant_conv.bias"));
    put_bf16("embed_weight", f("decoder.proj_in.weight"));
    put_bf16("embed_bias", f("decoder.proj_in.bias"));
    put_bf16("register_tokens", f("decoder.register_tokens"));   // [1,4,1024] -> [4,1024]
    put_f32("norm_weight", f("decoder.norm_out.weight"));
    put_f32("norm_bias", f("decoder.norm_out.bias"));
    put_bf16("proj_weight", f("decoder.proj_out.weight"));
    put_bf16("proj_bias", f("decoder.proj_out.bias"));

    for (int l = 0; l < LAYERS; ++l) {
        std::string p = "decoder.transformer_blocks." + std::to_string(l);
        std::string k = std::to_string(l) + ".";
        // q/k/v rows interleaved per head: [h0 q, h0 k, h0 v, h1 q, ...]
        auto q = f(p + ".attn.to_q.weight"), kk = f(p + ".attn.to_k.weight"), v = f(p + ".attn.to_v.weight");
        auto qb = f(p + ".attn.to_q.bias"), kb = f(p + ".attn.to_k.bias"), vb = f(p + ".attn.to_v.bias");
        std::vector<float> qkv(size_t(3) * H * H), qkvb(3 * H);
        for (int h = 0; h < HEADS; ++h)
            for (int part = 0; part < 3; ++part) {
                const auto &src = part == 0 ? q : part == 1 ? kk : v;
                const auto &srcb = part == 0 ? qb : part == 1 ? kb : vb;
                size_t dst_row = size_t(h) * 3 * HD + size_t(part) * HD;
                std::memcpy(&qkv[dst_row * H], &src[size_t(h) * HD * H], size_t(HD) * H * 4);
                std::memcpy(&qkvb[dst_row], &srcb[size_t(h) * HD], HD * 4);
            }
        // gated FFN: swap the two halves of the first projection
        auto w1 = f(p + ".ff.net.0.proj.weight"), b1 = f(p + ".ff.net.0.proj.bias");
        size_t half_rows = w1.size() / H / 2;
        std::vector<float> w1s(w1.size()), b1s(b1.size());
        std::memcpy(w1s.data(), &w1[half_rows * H], half_rows * H * 4);
        std::memcpy(&w1s[half_rows * H], w1.data(), half_rows * H * 4);
        std::memcpy(b1s.data(), &b1[half_rows], half_rows * 4);
        std::memcpy(&b1s[half_rows], b1.data(), half_rows * 4);

        put_f32(k + "norm1", f(p + ".norm1.weight"));
        put_bf16(k + "qkv_weight", qkv);
        put_bf16(k + "qkv_bias", qkvb);
        put_bf16(k + "out_weight", f(p + ".attn.to_out.0.weight"));
        put_bf16(k + "out_bias", f(p + ".attn.to_out.0.bias"));
        put_f32(k + "scale1", f(p + ".scale1"));
        put_f32(k + "norm2", f(p + ".norm2.weight"));
        put_bf16(k + "w1_weight", w1s);
        put_bf16(k + "w1_bias", b1s);
        put_bf16(k + "w2_weight", f(p + ".ff.net.2.weight"));
        put_bf16(k + "w2_bias", f(p + ".ff.net.2.bias"));
        put_f32(k + "scale2", f(p + ".scale2"));
    }
    (void)C;
    w->bank->materialize();
    return w;
}

}  // namespace axd
