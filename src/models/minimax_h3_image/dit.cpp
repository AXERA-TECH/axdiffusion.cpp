// DiT: host text/image projections, 2-layer token refiner, 6 Euler steps x 50 blocks, PDD head.
#include <cmath>

#include "common.hpp"
#include "pipeline.hpp"

namespace axd {

namespace {

constexpr int kWorkspaceHeads = 14;  // 56 heads / 4 cores
constexpr int kAdalnRows = 3;

int padded_length(int used) { return std::max((used + 63) / 64 * 64, (used + 255) / 256 * 256); }
int runtime_padded_length(int used) {
    int base = padded_length(used);
    return (base + 511) / 512 * 512;
}

// out[rows, N] (bf16) = x[rows, K] @ W[N, K]^T + b, FP32 accumulation, threaded over N.
void host_linear(const float *x, int rows, int K, const std::vector<float> &W, const std::vector<float> &b, int N,
                 uint16_t *out) {
    parallel_for(size_t(N), [&](size_t n0, size_t n1) {
        for (size_t n = n0; n < n1; ++n) {
            const float *w = &W[n * size_t(K)];
            for (int r = 0; r < rows; ++r) {
                const float *xr = x + size_t(r) * K;
                float acc = 0.f;
#pragma omp simd reduction(+ : acc)
                for (int k = 0; k < K; ++k) acc += xr[k] * w[k];
                out[size_t(r) * N + n] = f32_to_bf16(acc + b[n]);
            }
        }
    });
}

std::string shape_key(int padded) {
    switch (padded) {
        case 1280: return "1024x1024 short";
        case 1536: return "1024x1024 arbitrary";
        case 4608: return "2048x2048";
    }
    fail("no compiled DiT kernel for padded length " + std::to_string(padded));
}

}  // namespace

struct DitShape {
    int padded;
    DeviceBuffer x[2], attention_output, residual, ffn, adaln_ws, mlp_ws, k_ws, v_ws, attn_ws, q_ws, video, inverse,
        final_ws, combined, rope, positions, bounds;
};

struct RefinerWs {
    int rows;
    DeviceBuffer x[2], attention_output, residual, ffn, mlp, k_ws, v_ws, attn_ws, q_ws, normalized, bounds;
};

Dit::Dit(const ModelDir &dir, const DitWeights &w) : w_(w) {
    for (int p : {1280, 1536, 4608}) {
        block_[p] = std::make_unique<Model>(dir.axmodel("dit-block", shape_key(p)));
        final_[p] = std::make_unique<Model>(dir.axmodel("dit-final", shape_key(p)));
        // optional block kernel with the LoRA low-rank branch
        if (dir.has_axmodel("dit-block-lora", shape_key(p)))
            lora_block_[p] = std::make_unique<Model>(dir.axmodel("dit-block-lora", shape_key(p)));
    }
    for (int r : {64, 512}) {
        std::string key = r == 64 ? "64-row" : "512-row";
        refiner_[r] = std::make_unique<Model>(dir.axmodel("dit-refiner", key));
        rmsnorm_[r] = std::make_unique<Model>(dir.axmodel("dit-refiner-norm", key));
    }
    // Allocate every workspace up front so requests never touch the allocator.
    for (int p : {1280, 1536, 4608}) shape(p);
    for (int r : {64, 512}) refiner_ws(r);
}

Dit::~Dit() = default;

DitShape &Dit::shape(int padded) {
    auto &slot = shapes_[padded];
    if (slot) return *slot;
    auto s = std::make_unique<DitShape>();
    s->padded = padded;
    const size_t row = size_t(kDitHidden) * 2;
    const size_t kv = size_t(4) * kWorkspaceHeads * padded * kHeadDim * 2;
    for (auto &x : s->x) x = DeviceBuffer(padded * row, "dit-x");
    s->attention_output = DeviceBuffer(padded * row, "dit-attn-out");
    s->residual = DeviceBuffer(padded * row, "dit-residual");
    s->ffn = DeviceBuffer(size_t(padded) * kDitFfn * 2, "dit-ffn");
    s->adaln_ws = DeviceBuffer(size_t(3) * 6 * kDitHidden * 2, "dit-adaln-ws");
    s->mlp_ws = DeviceBuffer(padded * row, "dit-mlp-ws");
    s->k_ws = DeviceBuffer(kv, "dit-k-ws");
    s->v_ws = DeviceBuffer(kv, "dit-v-ws");
    s->attn_ws = DeviceBuffer(kv, "dit-attn-ws");
    s->q_ws = DeviceBuffer(kv, "dit-q-ws");
    s->video = DeviceBuffer(size_t(padded) * kPatchDim * 2, "dit-video");
    s->inverse = DeviceBuffer(size_t(padded) * 4, "dit-inverse");
    s->final_ws = DeviceBuffer(size_t(2) * kDitHidden * 2, "dit-final-ws");
    s->combined = DeviceBuffer(size_t(padded) * 4, "dit-combined");
    s->rope = DeviceBuffer(size_t(padded) * 96 * 2, "dit-rope");
    s->positions = DeviceBuffer(size_t(padded) * 8 * 4, "dit-positions");
    // runtime sequence bounds; the 1280 base kernel ignores them, its LoRA variant reads them
    s->bounds = DeviceBuffer(3 * 4, "dit-bounds");
    s->inverse.buf().fill_zero();
    // positions: arange(padded) repeated 8 times per row
    std::vector<int32_t> pos(size_t(padded) * 8);
    for (int i = 0; i < padded; ++i)
        for (int j = 0; j < 8; ++j) pos[size_t(i) * 8 + j] = i;
    s->positions.buf().upload(pos.data(), pos.size() * 4);
    slot = std::move(s);
    return *slot;
}

RefinerWs &Dit::refiner_ws(int rows) {
    auto &slot = refiners_[rows];
    if (slot) return *slot;
    auto r = std::make_unique<RefinerWs>();
    r->rows = rows;
    const size_t row = size_t(kDitHidden) * 2;
    const size_t kv = size_t(4) * kWorkspaceHeads * rows * kHeadDim * 2;
    for (auto &x : r->x) x = DeviceBuffer(rows * row, "ref-x");
    r->attention_output = DeviceBuffer(rows * row, "ref-attn");
    r->residual = DeviceBuffer(rows * row, "ref-residual");
    r->ffn = DeviceBuffer(size_t(rows) * kDitFfn * 2, "ref-ffn");
    r->mlp = DeviceBuffer(rows * row, "ref-mlp");
    r->k_ws = DeviceBuffer(kv, "ref-k");
    r->v_ws = DeviceBuffer(kv, "ref-v");
    r->attn_ws = DeviceBuffer(kv, "ref-attn-ws");
    r->q_ws = DeviceBuffer(kv, "ref-q");
    r->normalized = DeviceBuffer(rows * row, "ref-normalized");
    if (rows == 512) r->bounds = DeviceBuffer(12, "ref-bounds");
    slot = std::move(r);
    return *slot;
}

void Dit::prepare(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
                  const LoraSet *lora) {
    image_tokens_ = (latent_h / 2) * (latent_w / 2);
    used_ = text_rows + image_tokens_;
    const bool fast = latent_h == 64 && latent_w == 64 && text_rows == 51;
    padded_ = fast ? 1280 : runtime_padded_length(used_);
    if (!fast && !(padded_ - 512 < used_ && used_ <= padded_)) fail("runtime sequence window violated");
    if (!block_.count(padded_)) fail("no compiled DiT kernel for padded length " + std::to_string(padded_));
    if (lora && !lora_block_.count(padded_))
        fail("LoRA is not available for this shape (no LoRA block kernel for padded length " + std::to_string(padded_) + ")");
    text_rows_ = text_rows;

    // 1) text projection 5120 -> 5376 on the host (any prompt length)
    std::vector<float> xt(size_t(text_rows) * kTextDim);
    for (size_t i = 0; i < xt.size(); ++i) xt[i] = bf16_to_f32(text_hidden[i]);
    std::vector<uint16_t> proj(size_t(text_rows) * kDitHidden);
    host_linear(xt.data(), text_rows, kTextDim, w_.context_w, w_.context_b, kDitHidden, proj.data());

    // 2) token refiner (2 layers) + RMSNorm on the NPU
    const int rows = fast ? 64 : 512;
    auto &r = refiner_ws(rows);
    r.x[0].buf().fill_zero();
    r.x[0].buf().upload(proj.data(), proj.size() * 2);
    if (r.bounds.size()) {
        int32_t b[3] = {0, text_rows, rows};
        r.bounds.buf().upload(b, sizeof(b));
    }
    Buf cur = r.x[0], tgt = r.x[1];
    for (int i = 0; i < 2; ++i) {
        std::string p = "refiner." + std::to_string(i) + ".";
        // LoRA-merged refiner weights when the active set touches this projection
        auto rw = [&](const std::string &n) {
            if (lora && lora->refiner && lora->refiner->has(p + n)) return lora->refiner->get(p + n);
            return w_.shared->get(p + n);
        };
        Bindings in = {{"x", cur},
                       {"norm1_weight", w_.shared->get(p + "norm1")},
                       {"norm2_weight", w_.shared->get(p + "norm2")},
                       {"qkv_weight", rw("qkv")},
                       {"q_norm_weight", w_.shared->get(p + "q_norm")},
                       {"k_norm_weight", w_.shared->get(p + "k_norm")},
                       {"out_weight", rw("out")},
                       {"fc1_weight", rw("fc1")},
                       {"fc2_weight", rw("fc2")},
                       {"k_workspace", r.k_ws},
                       {"v_workspace", r.v_ws},
                       {"attention_workspace", r.attn_ws},
                       {"mlp_workspace", r.mlp},
                       {"q_workspace", r.q_ws}};
        if (r.bounds.size()) in["sequence_bounds"] = r.bounds;
        refiner_[rows]->run(in, {{"output", tgt},
                                 {"attention_output", r.attention_output},
                                 {"ffn_workspace", r.ffn},
                                 {"residual_workspace", r.residual}});
        std::swap(cur, tgt);
    }
    rmsnorm_[rows]->run({{"x", cur}, {"weight", w_.shared->get("refiner.final_norm")}}, {{"output", r.normalized}});
    refined_text_ = r.normalized.buf().view(0, size_t(text_rows) * kDitHidden * 2);

    // 3) per-request geometry: 3D RoPE table, combined indices, sequence bounds
    auto &s = shape(padded_);
    const double sqrt_area = std::sqrt(double(latent_h) * latent_w);
    auto grid = [&](int dim, int k) {  // linspace(left, left+ratio, dim/2, endpoint=False) * 32
        double ratio = dim / sqrt_area, left = (1.0 - ratio) / 2.0;
        return (left + ratio * k / (dim / 2)) * 32.0;
    };
    float inv_freq[16];
    for (int i = 0; i < 16; ++i) inv_freq[i] = 1.0f / std::pow(10000.0f, float(2 * i) / 32.0f);
    std::vector<uint16_t> rope(size_t(padded_) * 96, 0);
    for (int t = 0; t < used_; ++t) {
        float p[3];
        if (t < text_rows) {
            p[0] = float(t), p[1] = 0.f, p[2] = 0.f;
        } else {
            int i = t - text_rows, hh = i / (latent_w / 2), ww = i % (latent_w / 2);
            p[0] = float(text_rows), p[1] = float(grid(latent_h, hh)), p[2] = float(grid(latent_w, ww));
        }
        for (int a = 0; a < 3; ++a)
            for (int i = 0; i < 16; ++i) {
                float f = p[a] * inv_freq[i];
                rope[size_t(t) * 96 + a * 16 + i] = f32_to_bf16(std::cos(f));
                rope[size_t(t) * 96 + 48 + a * 16 + i] = f32_to_bf16(std::sin(f));
            }
    }
    s.rope.buf().upload(rope.data(), rope.size() * 2);
    std::vector<uint32_t> combined(padded_, 0);
    for (int i = 0; i < text_rows; ++i) combined[i] = 1;
    s.combined.buf().upload(combined.data(), combined.size() * 4);
    if (s.bounds.size()) {
        int32_t b[3] = {0, used_, padded_};
        s.bounds.buf().upload(b, sizeof(b));
    }
    s.x[0].buf().fill_zero();
    s.x[1].buf().fill_zero();
}

void Dit::step(std::vector<float> &sample, int si, int blocks, Timing &t, const LoraSet *lora) {
    auto &s = shape(padded_);
    const size_t row = size_t(kDitHidden) * 2;
    // image rows: patch projection 96 -> 5376 on the host
    std::vector<uint16_t> image(size_t(image_tokens_) * kDitHidden);
    host_linear(sample.data(), image_tokens_, kPatchDim, w_.proj_in_w, w_.proj_in_b, kDitHidden, image.data());
    Buf x0 = s.x[0];
    x0.copy_from(refined_text_, refined_text_.size, 0, 0);
    x0.upload(image.data(), image.size() * 2, size_t(text_rows_) * row);
    if (size_t(used_) * row < x0.size) x0.view(size_t(used_) * row, x0.size - size_t(used_) * row).fill_zero();

    auto &bank_shared = *w_.shared;
    int cur = 0;
    for (int l = 0; l < blocks; ++l) {
        double t0 = now_s();
        auto &b = *w_.blocks[l];
        Bindings in = {{"x", s.x[cur]},
                       {"adaln_input", w_.adaln_input(si)},
                       {"combined_indices", s.combined},
                       {"norm1_weight", b.get("norm1")},
                       {"norm2_weight", b.get("norm2")},
                       {"shift_msa_input", w_.adaln_slot(si, l, 0)},
                       {"scale_msa_input", w_.adaln_slot(si, l, 1)},
                       {"gate_msa_input", w_.adaln_slot(si, l, 2)},
                       {"shift_mlp_input", w_.adaln_slot(si, l, 3)},
                       {"scale_mlp_input", w_.adaln_slot(si, l, 4)},
                       {"gate_mlp_input", w_.adaln_slot(si, l, 5)},
                       {"qkv_weight", b.get("qkv.payload")},
                       {"qkv_scale", b.get("qkv.scale")},
                       {"qkv_scale2", b.get("qkv.scale2")},
                       {"q_norm_weight", b.get("q_norm")},
                       {"k_norm_weight", b.get("k_norm")},
                       {"rope_cache", s.rope},
                       {"position_ids", s.positions},
                       {"out_weight", b.get("out.payload")},
                       {"out_scale", b.get("out.scale")},
                       {"out_scale2", b.get("out.scale2")},
                       {"fc1_weight", b.get("fc1.payload")},
                       {"fc1_scale", b.get("fc1.scale")},
                       {"fc1_scale2", b.get("fc1.scale2")},
                       {"fc2_weight", b.get("fc2.payload")},
                       {"fc2_scale", b.get("fc2.scale")},
                       {"fc2_scale2", b.get("fc2.scale2")},
                       {"adaln_workspace", s.adaln_ws},
                       {"residual_workspace", s.residual},
                       {"attention_output", s.attention_output},
                       {"mlp_workspace", s.mlp_ws},
                       {"ffn_workspace", s.ffn},
                       {"k_workspace", s.k_ws},
                       {"v_workspace", s.v_ws},
                       {"attention_workspace", s.attn_ws},
                       {"q_workspace", s.q_ws}};
        if (s.bounds.size()) in["sequence_bounds"] = s.bounds;
        Model *model = block_[padded_].get();
        if (lora) {
            model = lora_block_[padded_].get();
            const auto &ab = lora->block_ab[l];
            for (int k = 0; k < 8; ++k) in[kLoraInputs[k]] = ab[k];
        }
        model->run(in, {{"output", s.x[1 - cur]},
                                  {"attention_output", s.attention_output},
                                  {"residual_workspace", s.residual},
                                  {"ffn_workspace", s.ffn}});
        cur = 1 - cur;
        t.block_seconds.push_back(now_s() - t0);
    }
    final_[padded_]->run({{"x", s.x[cur]},
                          {"inverse_indices", s.inverse},
                          {"adaln_input", w_.adaln_input(si)},
                          {"norm_weight", bank_shared.get("final.norm")},
                          {"shift_input", w_.final_shift(si)},
                          {"scale_input", w_.final_scale(si)},
                          {"final_adaln_workspace", s.final_ws},
                          {"video_weight", bank_shared.get("pdd." + std::to_string(si) + ".weight")},
                          {"video_bias", bank_shared.get("pdd." + std::to_string(si) + ".bias")}},
                         {{"video_output", s.video}});
    std::vector<uint16_t> vel(size_t(image_tokens_) * kPatchDim);
    s.video.buf().download(vel.data(), vel.size() * 2, size_t(text_rows_) * kPatchDim * 2);
    const float dt = kSigmas[si] - kSigmas[si + 1];
    for (size_t i = 0; i < vel.size(); ++i) sample[i] += dt * bf16_to_f32(vel[i]);
}

void Dit::run(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
              std::vector<float> &sample, int steps, int blocks, Timing &t, const LoraSet *lora) {
    if (steps < 1 || steps > kSteps) fail("steps must be 1..6");
    if (blocks < 1 || blocks > kDitLayers) fail("blocks must be 1..50");
    double t0 = now_s();
    prepare(text_hidden, text_rows, latent_h, latent_w, lora);
    t.prepare = now_s() - t0;
    t0 = now_s();
    for (int i = 0; i < steps; ++i) {
        double s0 = now_s();
        step(sample, i, blocks, t, lora);
        t.step_seconds.push_back(now_s() - s0);
        AXD_LOG("step %d/%d %.2f s", i + 1, steps, t.step_seconds.back());
    }
    t.steps = now_s() - t0;
}

}  // namespace axd
