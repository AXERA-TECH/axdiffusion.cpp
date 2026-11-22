// DiT: host text/image projections, 2-layer token refiner, 6 Euler steps x 50 blocks, PDD head.
#include <cmath>

#include "common.hpp"
#include "h3_dit.hpp"
#include "lora.hpp"
#include <cstdlib>
#include "safetensors.hpp"

namespace axd {

const float kSigmas[kSteps + 1] = {1.0f, 0.9675f, 0.8875f, 0.75f, 0.5328f, 0.2320f, 0.0f};

namespace {

constexpr int kWorkspaceHeads = 14;  // 56 heads / 4 cores
constexpr int kAdalnRows = 3;


// out[rows, N] (bf16) = x[rows, K] @ W[N, K]^T + b, FP32 accumulation, threaded over N.
void host_linear(const float *x, int rows, int K, const std::vector<float> &W, const std::vector<float> &b, int N,
                 uint16_t *out) {
    // Threaded over output columns; each weight row stays in L1 while all rows stream past it. On the board's
    // Cortex-A55 cores this simple order beat row- and column-blocked variants.
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
    // Sequence tiers come from the model directory's manifest ("... padded N ..."), so new tiers
    // (longer prompts, edit, other aspect ratios) only need new AXModels, not code.
    // Only the paths are recorded here: a tier's kernel is loaded, and its workspaces allocated, on first use.
    auto load = [&](const std::string &purpose, Tiers &into, bool need_granule) {
        auto shapes = dir.tier_shapes(purpose, "padded ");
        for (auto &kv : dir.tiers(purpose, "padded ")) {
            Kernel k;
            k.path = kv.second;
            k.what = purpose + " tier " + std::to_string(kv.first);
            if (need_granule) {
                k.granule = ModelDir::shape_int(shapes[kv.first], "granule ", 0);
                if (k.granule <= 0) fail(k.what + ": manifest shape must state 'granule N'");
            }
            into[kv.first] = std::move(k);
        }
    };
    load("dit-block", block_, true);
    load("dit-block-edit", edit_block_, true);
    load("dit-final-edit", edit_final_, false);
    load("dit-final", final_, false);
    // LoRA block kernels per rank tier: "dit-block-lora" is rank 64, "dit-block-lora-rN" is rank N.
    load("dit-block-lora", lora_block_[64], true);
    for (int r : {32, 96, 128, 192, 256}) load("dit-block-lora-r" + std::to_string(r), lora_block_[r], true);
    for (auto it = lora_block_.begin(); it != lora_block_.end();)
        it = it->second.empty() ? lora_block_.erase(it) : std::next(it);
    for (auto &kv : dir.tiers("dit-refiner", "")) {
        refiner_[kv.first].path = kv.second;
        refiner_[kv.first].granule = 512;
        refiner_[kv.first].what = "dit-refiner " + std::to_string(kv.first) + "-row";
    }
    context_proj_.path = dir.axmodel("dit-context-proj");
    context_proj_.what = "dit-context-proj";
    for (auto &kv : dir.tiers("dit-refiner-norm", "")) rmsnorm_[kv.first].path = kv.second;
    for (auto &kv : block_)
        if (!final_.count(kv.first)) fail("DiT tier " + std::to_string(kv.first) + " has a block but no final kernel");
    for (auto &kv : refiner_)
        if (!rmsnorm_.count(kv.first)) fail("refiner tier " + std::to_string(kv.first) + " has no RMSNorm kernel");
    if (block_.empty() || refiner_.empty()) fail("model directory has no DiT / refiner kernels");
}

Dit::~Dit() = default;

Model &Dit::Kernel::get() {
    if (!model) {
        model = std::make_unique<Model>(path);
        if (granule && !model->has_input("sequence_bounds")) fail(what + " kernel has no runtime sequence bounds");
    }
    return *model;
}

int Dit::pick_tier(int used, int lora_rank, bool edit) const {
    const Tiers *cands = &block_;
    if (edit) {
        cands = &edit_block_;
    } else if (lora_rank) {
        auto it = lora_block_.find(lora_rank);
        if (it == lora_block_.end()) return 0;
        cands = &it->second;
    }
    const auto &finals = edit ? edit_final_ : final_;
    for (auto &kv : *cands) {
        const int P = kv.first;
        if (P < used || !finals.count(P)) continue;
        if (P - kv.second.granule < used) return P;
    }
    return 0;
}

int Dit::max_text_rows(int image_rows, int lora_rank) const {
    const Tiers *cands = &block_;
    if (lora_rank) {
        auto it = lora_block_.find(lora_rank);
        if (it == lora_block_.end()) return 0;
        cands = &it->second;
    }
    int best = 0;
    for (auto &kv : *cands)
        if (final_.count(kv.first)) best = std::max(best, kv.first - image_rows);
    return best;
}

std::vector<int> Dit::lora_ranks() const {
    std::vector<int> out;
    for (auto &kv : lora_block_)
        if (!kv.second.empty()) out.push_back(kv.first);
    return out;
}

DitShape &Dit::shape(int padded) {
    // One tier's workspaces are kept (they are up to a few GiB); another tier replaces them.
    for (auto it = shapes_.begin(); it != shapes_.end();) it = it->first == padded ? std::next(it) : shapes_.erase(it);
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
    s->adaln_ws = DeviceBuffer(size_t(2) * 3 * 6 * kDitHidden * 2, "dit-adaln-ws");  // M=2 edit size
    s->mlp_ws = DeviceBuffer(padded * row, "dit-mlp-ws");
    s->k_ws = DeviceBuffer(kv, "dit-k-ws");
    s->v_ws = DeviceBuffer(kv, "dit-v-ws");
    s->attn_ws = DeviceBuffer(kv, "dit-attn-ws");
    s->q_ws = DeviceBuffer(kv, "dit-q-ws");
    s->video = DeviceBuffer(size_t(padded) * kPatchDim * 2, "dit-video");
    s->inverse = DeviceBuffer(size_t(padded) * 4, "dit-inverse");
    s->final_ws = DeviceBuffer(size_t(2) * 2 * kDitHidden * 2, "dit-final-ws");  // M=2 edit size
    s->combined = DeviceBuffer(size_t(padded) * 4, "dit-combined");
    s->rope = DeviceBuffer(size_t(padded) * 96 * 2, "dit-rope");
    s->positions = DeviceBuffer(size_t(padded) * 8 * 4, "dit-positions");
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
    for (auto it = refiners_.begin(); it != refiners_.end();) it = it->first == rows ? std::next(it) : refiners_.erase(it);
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
    r->bounds = DeviceBuffer(12, "ref-bounds");
    slot = std::move(r);
    return *slot;
}

void Dit::prepare(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
                  const LoraSet *lora, const DitEdit *edit) {
    edit_ = edit != nullptr;
    image_tokens_ = (latent_h / 2) * (latent_w / 2);
    reference_rows_ = edit ? int(edit->reference_rows.size() / kPatchDim) : 0;
    used_ = text_rows + reference_rows_ + image_tokens_;
    if (edit) {
        if (!supports_edit()) fail("this model directory has no reference-edit DiT kernels");
        if (lora) fail("LoRA is not supported together with reference editing yet");
        if (int(edit->text_tags.size()) != text_rows || int(edit->positions.size()) != used_)
            fail("edit layout does not match the sequence");
    }
    if (lora && !lora_block_.count(lora->rank))
        fail("no rank-" + std::to_string(lora->rank) + " LoRA block kernels in this model directory");
    padded_ = pick_tier(used_, lora ? lora->rank : 0, edit != nullptr);
    const auto &cands = edit ? edit_block_ : lora ? lora_block_.at(lora->rank) : block_;
    if (!padded_) {
        std::string have;
        for (auto &kv : cands) have += " " + std::to_string(kv.first);
        fail("no DiT kernel tier for a " + std::to_string(used_) + "-token sequence" +
             (lora ? " with a rank-" + std::to_string(lora->rank) + " LoRA" : std::string()) + " (available:" + have +
             "); a runtime tier accepts (padded - granule, padded] rows: pick a size/prompt that lands in one");
    }
    text_rows_ = text_rows;

    // 1) text projection 5120 -> 5376 on the NPU, one 256-row page per launch (any prompt length)
    std::vector<uint16_t> proj(size_t(text_rows) * kDitHidden);
    {
        const double tl = now_s();
        const int page = 256;
        if (!ctx_in_.size()) {
            ctx_in_ = DeviceBuffer(size_t(page) * kDitTextDim * 2, "dit-context-in");
            ctx_out_ = DeviceBuffer(size_t(page) * kDitHidden * 2, "dit-context-out");
        }
        Model &m = context_proj_.get();
        for (int r0 = 0; r0 < text_rows; r0 += page) {
            const int n = std::min(page, text_rows - r0);
            if (n < page) ctx_in_.buf().fill_zero();
            ctx_in_.buf().upload(text_hidden.data() + size_t(r0) * kDitTextDim, size_t(n) * kDitTextDim * 2);
            m.run({{"x", ctx_in_}, {"weight", w_.shared->get("context.weight")}, {"bias", w_.shared->get("context.bias")}},
                  {{"output", ctx_out_}});
            ctx_out_.buf().view(0, size_t(n) * kDitHidden * 2).download(proj.data() + size_t(r0) * kDitHidden, size_t(n) * kDitHidden * 2);
        }
        AXD_LOG("text projection %d rows: %.2f s", text_rows, now_s() - tl);
    }

    // 2) token refiner (2 layers) + RMSNorm on the NPU
    int rows = 0;
    // runtime refiner kernels accept (rows - 512, rows] valid text rows
    for (auto it = refiner_.begin(); !rows && it != refiner_.end(); ++it)
        if (it->first >= text_rows && it->first - 512 < text_rows) rows = it->first;
    if (!rows) fail("no token refiner tier for " + std::to_string(text_rows) + " text rows");
    auto &r = refiner_ws(rows);
    r.x[0].buf().fill_zero();
    r.x[0].buf().upload(proj.data(), proj.size() * 2);
    {
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
        in["sequence_bounds"] = r.bounds;
        refiner_.at(rows).get().run(in, {{"output", tgt},
                                 {"attention_output", r.attention_output},
                                 {"ffn_workspace", r.ffn},
                                 {"residual_workspace", r.residual}});
        std::swap(cur, tgt);
    }
    rmsnorm_.at(rows).get().run({{"x", cur}, {"weight", w_.shared->get("refiner.final_norm")}}, {{"output", r.normalized}});
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
        if (edit) {
            for (int a = 0; a < 3; ++a) p[a] = float(edit->positions[t][a]);
        } else if (t < text_rows) {
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
    // AdaLN row per token: timestep_index * 3 + tag (text 1 / vision-block 0 / image 0; references use 0.999 = index 1)
    std::vector<uint32_t> combined(padded_, 0);
    std::vector<int32_t> inverse(padded_, 0);
    for (int i = 0; i < text_rows; ++i) combined[i] = edit ? uint32_t(edit->text_tags[i]) : 1u;
    for (int i = 0; i < reference_rows_; ++i) {
        combined[text_rows + i] = 3;
        inverse[text_rows + i] = 1;
    }
    s.combined.buf().upload(combined.data(), combined.size() * 4);
    s.inverse.buf().upload(inverse.data(), inverse.size() * 4);
    reference_proj_.clear();
    if (reference_rows_) {
        reference_proj_.resize(size_t(reference_rows_) * kDitHidden);
        host_linear(edit->reference_rows.data(), reference_rows_, kPatchDim, w_.proj_in_w, w_.proj_in_b, kDitHidden,
                    reference_proj_.data());
    }
    {
        int32_t b[3] = {0, used_, padded_};
        s.bounds.buf().upload(b, sizeof(b));
    }
    s.x[0].buf().fill_zero();
    s.x[1].buf().fill_zero();
}

void Dit::step(std::vector<float> &sample, int si, int blocks, Timing &t, const LoraSet *lora) {
    auto &s = shape(padded_);
    if (!stream_) stream_ = std::make_unique<Stream>();  // created on the NPU worker thread
    const size_t row = size_t(kDitHidden) * 2;
    // image rows: patch projection 96 -> 5376 on the host
    std::vector<uint16_t> image(size_t(image_tokens_) * kDitHidden);
    host_linear(sample.data(), image_tokens_, kPatchDim, w_.proj_in_w, w_.proj_in_b, kDitHidden, image.data());
    Buf x0 = s.x[0];
    x0.copy_from(refined_text_, refined_text_.size, 0, 0);
    if (reference_rows_) x0.upload(reference_proj_.data(), reference_proj_.size() * 2, size_t(text_rows_) * row);
    x0.upload(image.data(), image.size() * 2, size_t(text_rows_ + reference_rows_) * row);
    if (size_t(used_) * row < x0.size) x0.view(size_t(used_) * row, x0.size - size_t(used_) * row).fill_zero();

    auto &bank_shared = *w_.shared;
    int cur = 0;
    for (int l = 0; l < blocks; ++l) {
        double t0 = now_s();
        auto &b = *w_.blocks[l];
        Bindings in = {{"x", s.x[cur]},
                       {"adaln_input", w_.adaln_input(si, edit_)},
                       {"combined_indices", s.combined},
                       {"norm1_weight", b.get("norm1")},
                       {"norm2_weight", b.get("norm2")},
                       {"shift_msa_input", w_.adaln_slot(si, l, 0, edit_)},
                       {"scale_msa_input", w_.adaln_slot(si, l, 1, edit_)},
                       {"gate_msa_input", w_.adaln_slot(si, l, 2, edit_)},
                       {"shift_mlp_input", w_.adaln_slot(si, l, 3, edit_)},
                       {"scale_mlp_input", w_.adaln_slot(si, l, 4, edit_)},
                       {"gate_mlp_input", w_.adaln_slot(si, l, 5, edit_)},
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
        in["sequence_bounds"] = s.bounds;
        Model *model = edit_ ? &edit_block_.at(padded_).get() : &block_.at(padded_).get();
        if (lora) {
            model = &lora_block_.at(lora->rank).at(padded_).get();
            const auto &ab = lora->block_ab[l];
            for (int k = 0; k < 8; ++k) in[kLoraInputs[k]] = ab[k];
        }
        const Bindings outs = {{"output", s.x[1 - cur]},
                               {"attention_output", s.attention_output},
                               {"residual_workspace", s.residual},
                               {"ffn_workspace", s.ffn}};
        model->submit(*stream_, in, outs);
        cur = 1 - cur;
        t.block_seconds.push_back(now_s() - t0);
    }
    auto final_run = [&](const Bindings &in, const Bindings &out) {
        (edit_ ? edit_final_ : final_).at(padded_).get().submit(*stream_, in, out);
        double t0 = now_s();
        stream_->sync();  // the velocity is read on the host next
        t.sync_seconds += now_s() - t0;
    };
    final_run({{"x", s.x[cur]},
                          {"inverse_indices", s.inverse},
                          {"adaln_input", w_.adaln_input(si, edit_)},
                          {"norm_weight", bank_shared.get("final.norm")},
                          {"shift_input", w_.final_shift(si, edit_)},
                          {"scale_input", w_.final_scale(si, edit_)},
                          {"final_adaln_workspace", s.final_ws},
                          {"video_weight", bank_shared.get("pdd." + std::to_string(si) + ".weight")},
                          {"video_bias", bank_shared.get("pdd." + std::to_string(si) + ".bias")}},
                         {{"video_output", s.video}});
    std::vector<uint16_t> vel(size_t(image_tokens_) * kPatchDim);
    s.video.buf().download(vel.data(), vel.size() * 2, size_t(text_rows_ + reference_rows_) * kPatchDim * 2);
    const float dt = kSigmas[si] - kSigmas[si + 1];
    for (size_t i = 0; i < vel.size(); ++i) sample[i] += dt * bf16_to_f32(vel[i]);
}

void Dit::run(const std::vector<uint16_t> &text_hidden, int text_rows, int latent_h, int latent_w,
              std::vector<float> &sample, int steps, int blocks, Timing &t, const LoraSet *lora, Progress *progress,
              const DitEdit *edit) {
    if (steps < 1 || steps > kSteps) fail("steps must be 1..6");
    if (blocks < 1 || blocks > kDitLayers) fail("blocks must be 1..50");
    double t0 = now_s();
    prepare(text_hidden, text_rows, latent_h, latent_w, lora, edit);
    t.prepare = now_s() - t0;
    t0 = now_s();
    for (int i = 0; i < steps; ++i) {
        double s0 = now_s();
        if (progress && progress->cancel) throw Cancelled();
        step(sample, i, blocks, t, lora);
        if (progress) (*progress)("denoise", i + 1, steps);
        t.step_seconds.push_back(now_s() - s0);
        AXD_LOG("step %d/%d %.2f s", i + 1, steps, t.step_seconds.back());
    }
    t.steps = now_s() - t0;
}

Buf DitWeights::adaln_slot(int step, int layer, int slot, bool edit) const {
    const int rows = edit ? 6 : 3;
    const size_t slot_bytes = size_t(rows) * kDitHidden * 2;
    size_t index = (size_t(step) * kDitLayers + layer) * 6 + slot;
    return (edit ? *adaln_edit : *adaln).get("blocks").view(index * slot_bytes, slot_bytes);
}
Buf DitWeights::adaln_input(int step, bool edit) const {
    const size_t n = size_t(edit ? 2 : 1) * 2688 * 2;
    return (edit ? *adaln_edit : *adaln).get("adaln_input").view(size_t(step) * n, n);
}
Buf DitWeights::final_shift(int step, bool edit) const {
    const size_t n = size_t(edit ? 2 : 1) * kDitHidden * 2;
    return (edit ? *adaln_edit : *adaln).get("final_shift").view(size_t(step) * n, n);
}
Buf DitWeights::final_scale(int step, bool edit) const {
    const size_t n = size_t(edit ? 2 : 1) * kDitHidden * 2;
    return (edit ? *adaln_edit : *adaln).get("final_scale").view(size_t(step) * n, n);
}

std::unique_ptr<DitWeights> load_dit_weights(const ModelDir &dir) {
    auto w = std::make_unique<DitWeights>();
    // Every device tensor (shared layers, AdaLN tables, 50 blocks) is read from its file range by parallel
    // workers in one pass; the files stay open until then.
    std::vector<std::unique_ptr<SafeTensors>> files;
    std::vector<TensorBank *> banks;
    auto add_all = [&](TensorBank &bank, SafeTensors &st, const std::vector<std::string> &names) {
        for (auto &name : names) {
            auto &t = st.get(name);
            bank.add_file(name, st.fd(), st.file_offset(t), t.nbytes);
        }
        banks.push_back(&bank);
    };
    files.push_back(std::make_unique<SafeTensors>(dir.file("dit/shared.safetensors")));
    {
        auto &st = *files.back();
        w->shared = std::make_unique<TensorBank>("dit-shared");
        add_all(*w->shared, st, st.names());
        w->proj_in_w = to_f32(st.get("proj_in.weight"));
        w->proj_in_b = to_f32(st.get("proj_in.bias"));
        if (st.get("context.weight").dtype != "BF16") fail("dit/shared.safetensors context.weight must be bf16");
    }
    files.push_back(std::make_unique<SafeTensors>(dir.file("dit/adaln.safetensors")));
    {
        auto &st = *files.back();
        if (st.get("blocks").nbytes != size_t(kSteps) * kDitLayers * 6 * 3 * kDitHidden * 2)
            fail("unexpected adaln table size");  // [6,50,6,3,5376] bf16
        w->adaln = std::make_unique<TensorBank>("dit-adaln");
        add_all(*w->adaln, st, {"adaln_input", "final_shift", "final_scale", "blocks"});
    }
    if (dir.exists("dit/adaln_edit.safetensors")) {
        files.push_back(std::make_unique<SafeTensors>(dir.file("dit/adaln_edit.safetensors")));
        auto &st = *files.back();
        for (const char *name : {"adaln_input", "final_shift", "final_scale", "blocks"})
            if (st.get(name).dtype != "BF16") fail(std::string("adaln_edit ") + name + " must be bf16");
        if (st.get("blocks").nbytes != size_t(kSteps) * kDitLayers * 6 * 6 * kDitHidden * 2)
            fail("unexpected adaln_edit size");  // [6,50,6,6,5376] bf16
        w->adaln_edit = std::make_unique<TensorBank>("dit-adaln-edit");
        add_all(*w->adaln_edit, st, {"adaln_input", "final_shift", "final_scale", "blocks"});
    }
    // 50 block files: map them all, then copy with several workers (disk reads scale with parallel streams)
    for (int i = 0; i < kDitLayers; ++i) {
        char rel[64];
        std::snprintf(rel, sizeof(rel), "dit/blocks/block%02d.safetensors", i);
        files.push_back(std::make_unique<SafeTensors>(dir.file(rel)));
        auto bank = std::make_unique<TensorBank>("dit-block-" + std::to_string(i));
        for (auto &name : files.back()->names()) {
            auto &t = files.back()->get(name);
            bank->add_file(name, files.back()->fd(), files.back()->file_offset(t), t.nbytes);
        }
        banks.push_back(bank.get());
        w->blocks.push_back(std::move(bank));
    }
    materialize_parallel(banks);
    return w;
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

}  // namespace axd
