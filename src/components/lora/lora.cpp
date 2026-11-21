#include "lora.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <cmath>
#include <regex>
#include <set>
#include <sstream>

#include "common.hpp"
#include "h3_dit.hpp"
#include "safetensors.hpp"

namespace axd {

const char *const kLoraInputs[8] = {"lora_a_qkv", "lora_b_qkv", "lora_a_out", "lora_b_out",
                                    "lora_a_fc1", "lora_b_fc1", "lora_a_fc2", "lora_b_fc2"};

namespace {

enum Proj { QKV = 0, OUT = 1, FC1 = 2, FC2 = 3 };
const int kIn[4] = {kDitHidden, 7168, kDitHidden, kDitFfn};
const int kOut[4] = {3 * 7168, kDitHidden, 2 * kDitFfn, kDitHidden};
const char *kProjName[4] = {"qkv", "out", "fc1", "fc2"};

// One LoRA module mapped onto a (possibly partial) row range of a fused projection.
struct Piece {
    int rank = 0;
    std::vector<float> A;  // [rank, in]
    std::vector<float> B;  // [rows, rank]
    int row_offset = 0;    // destination row of B row 0 inside the fused projection
    bool swap_halves = false;
    float scale = 1.f;
};

struct Target {
    bool refiner;
    int index;
    Proj proj;
    bool operator<(const Target &o) const {
        return std::tie(refiner, index, proj) < std::tie(o.refiner, o.index, o.proj);
    }
};

// Strip framework prefixes so both diffusers and ComfyUI key styles reduce to a module path.
std::string normalize(std::string k) {
    for (bool changed = true; changed;) {
        changed = false;
        for (const char *p : {"base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.",
                              "model_0.transformer.", "model."}) {
            size_t n = std::strlen(p);
            if (k.compare(0, n, p) == 0) {
                k = k.substr(n);
                changed = true;
            }
        }
    }
    size_t d = k.find(".default");
    if (d != std::string::npos) k.erase(d, 8);
    return k;
}

// kohya flat names ("lora_unet_blocks_3_attn_qkv_proj", "lora_transformer_transformer_blocks_3_attn_to_q")
// -> dotted module paths understood by map_module().
std::string kohya_to_dotted(const std::string &m) {
    static const std::regex re(
        R"(^lora_(?:unet|transformer)_(token_refiner_refiner_blocks|token_refiner_blocks|transformer_blocks|blocks)_(\d+)_(.+)$)");
    std::smatch s;
    if (!std::regex_match(m, s, re)) return m;
    std::string kind = s[1].str(), op = s[3].str();
    if (kind == "token_refiner_refiner_blocks") kind = "token_refiner.refiner_blocks";
    else if (kind == "token_refiner_blocks") kind = "token_refiner.blocks";
    static const std::pair<const char *, const char *> ops[] = {
        {"attn_qkv_proj", "attn.qkv_proj"}, {"attn_out_proj", "attn.out_proj"}, {"mlp_fc1", "mlp.fc1"},
        {"mlp_fc2", "mlp.fc2"},           {"attn_to_q", "attn.to_q"},         {"attn_to_k", "attn.to_k"},
        {"attn_to_v", "attn.to_v"},       {"attn_to_out_0", "attn.to_out.0"}, {"ff_net_0_proj", "ff.net.0.proj"},
        {"ff_net_2", "ff.net.2"}};
    for (auto &o : ops)
        if (op == o.first) return kind + "." + s[2].str() + "." + o.second;
    return m;
}

// module path -> target; returns false for unknown modules.
bool map_module(const std::string &m, Target &t, int &row_offset, int &rows, bool &swap) {
    static const std::regex re_diff(
        R"(^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)$)");
    static const std::regex re_comfy(
        R"(^(blocks|token_refiner\.blocks)\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)$)");
    std::smatch s;
    std::string kind, op;
    if (std::regex_match(m, s, re_diff) || std::regex_match(m, s, re_comfy)) {
        kind = s[1].str();
        op = s[3].str();
        t.index = std::stoi(s[2].str());
    } else {
        return false;
    }
    t.refiner = kind.rfind("token_refiner", 0) == 0;
    if (t.index < 0 || t.index >= (t.refiner ? 2 : kDitLayers)) return false;
    row_offset = 0;
    swap = false;
    if (op == "attn.to_q" || op == "attn.to_k" || op == "attn.to_v") {
        t.proj = QKV;
        rows = 7168;
        row_offset = op == "attn.to_q" ? 0 : op == "attn.to_k" ? 7168 : 14336;
    } else if (op == "attn.qkv_proj") {
        t.proj = QKV, rows = kOut[QKV];
    } else if (op == "attn.to_out.0" || op == "attn.out_proj") {
        t.proj = OUT, rows = kOut[OUT];
    } else if (op == "ff.net.0.proj") {
        // diffusers order is [x | gate]; the kernel's fc1 rows are [silu-gate | up]
        t.proj = FC1, rows = kOut[FC1], swap = true;
    } else if (op == "mlp.fc1") {
        // ComfyUI/original checkpoint order already matches the kernel
        t.proj = FC1, rows = kOut[FC1], swap = false;
    } else {
        t.proj = FC2, rows = kOut[FC2];
    }
    return true;
}

// Upper Cholesky factor R of a symmetric positive (semi)definite n x n matrix G (G = R^T R); a tiny ridge keeps
// rank-deficient inputs (e.g. a LoRA that only adapts q of a fused qkv) well defined.
std::vector<double> cholesky_upper(std::vector<double> G, int n) {
    double tr = 0;
    for (int i = 0; i < n; ++i) tr += G[size_t(i) * n + i];
    for (int i = 0; i < n; ++i) G[size_t(i) * n + i] += 1e-12 * (tr > 0 ? tr / n : 1.0);
    std::vector<double> R(size_t(n) * n, 0.0);
    for (int j = 0; j < n; ++j) {
        double d = G[size_t(j) * n + j];
        for (int k = 0; k < j; ++k) d -= R[size_t(k) * n + j] * R[size_t(k) * n + j];
        double rjj = std::sqrt(std::max(d, 1e-300));
        R[size_t(j) * n + j] = rjj;
        for (int i = j + 1; i < n; ++i) {
            double v = G[size_t(j) * n + i];
            for (int k = 0; k < j; ++k) v -= R[size_t(k) * n + j] * R[size_t(k) * n + i];
            R[size_t(j) * n + i] = v / rjj;
        }
    }
    return R;
}

// X = R^{-1} Y for upper-triangular R [n,n], Y [n,m] (in place).
void solve_upper(const std::vector<double> &R, int n, std::vector<double> &Y, int m) {
    for (int c = 0; c < m; ++c)
        for (int i = n - 1; i >= 0; --i) {
            double v = Y[size_t(i) * m + c];
            for (int k = i + 1; k < n; ++k) v -= R[size_t(i) * n + k] * Y[size_t(k) * m + c];
            Y[size_t(i) * m + c] = v / R[size_t(i) * n + i];
        }
}

// Replace B [out,n] * A [n,in] by its best rank-k approximation, in place: on return B is [out,k] and A is [k,in].
// Uses only n x n Gram matrices plus row-streaming passes over B and A:
//   B = Qb Rb, A^T = Qa Ra (Rb, Ra from Cholesky of B^T B and A A^T),  C = Rb Ra^T = U S V^T (one-sided Jacobi),
//   B' = B Rb^{-1} U_k S_k,  A' = V_k^T Ra^{-T} A.
// Returns the fraction of the squared Frobenius norm of B*A that is kept.
double truncate_rank(std::vector<float> &B, std::vector<float> &A, int out, int in, int n, int k) {
    std::vector<double> GB(size_t(n) * n, 0.0), GA(size_t(n) * n, 0.0);
    for (int o = 0; o < out; ++o) {
        const float *b = &B[size_t(o) * n];
        for (int i = 0; i < n; ++i) {
            if (b[i] == 0.f) continue;
            double bi = b[i];
            for (int j = i; j < n; ++j) GB[size_t(i) * n + j] += bi * b[j];
        }
    }
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
            const float *ai = &A[size_t(i) * in], *aj = &A[size_t(j) * in];
            double d = 0;
            for (int c = 0; c < in; ++c) d += double(ai[c]) * aj[c];
            GA[size_t(i) * n + j] = d;
        }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j) GB[size_t(i) * n + j] = GB[size_t(j) * n + i], GA[size_t(i) * n + j] = GA[size_t(j) * n + i];
    auto Rb = cholesky_upper(GB, n), Ra = cholesky_upper(GA, n);
    std::vector<double> C(size_t(n) * n, 0.0), V(size_t(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double d = 0;
            for (int t = 0; t < n; ++t) d += Rb[size_t(i) * n + t] * Ra[size_t(j) * n + t];
            C[size_t(i) * n + j] = d;
        }
    for (int i = 0; i < n; ++i) V[size_t(i) * n + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                double a = 0, b = 0, g = 0;
                for (int i = 0; i < n; ++i) {
                    double cp = C[size_t(i) * n + p], cq = C[size_t(i) * n + q];
                    a += cp * cp, b += cq * cq, g += cp * cq;
                }
                if (g == 0.0 || std::fabs(g) <= 1e-15 * std::sqrt(a * b)) continue;
                off = std::max(off, std::fabs(g) / std::sqrt(a * b));
                double zeta = (b - a) / (2 * g);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
                double c = 1 / std::sqrt(1 + t * t), s = c * t;
                for (int i = 0; i < n; ++i) {
                    double cp = C[size_t(i) * n + p], cq = C[size_t(i) * n + q];
                    C[size_t(i) * n + p] = c * cp - s * cq;
                    C[size_t(i) * n + q] = s * cp + c * cq;
                    double vp = V[size_t(i) * n + p], vq = V[size_t(i) * n + q];
                    V[size_t(i) * n + p] = c * vp - s * vq;
                    V[size_t(i) * n + q] = s * vp + c * vq;
                }
            }
        if (off < 1e-12) break;
    }
    std::vector<double> sig(n);
    std::vector<int> order(n);
    double total = 0, kept = 0;
    for (int j = 0; j < n; ++j) {
        double s2 = 0;
        for (int i = 0; i < n; ++i) s2 += C[size_t(i) * n + j] * C[size_t(i) * n + j];
        sig[j] = s2, total += s2, order[j] = j;
    }
    std::sort(order.begin(), order.end(), [&](int x, int y) { return sig[x] > sig[y]; });
    for (int j = 0; j < k; ++j) kept += sig[order[j]];
    // TB = Rb^{-1} (U_k S_k) = Rb^{-1} C[:, top-k]   [n,k];   TA = Ra^{-1} V[:, top-k]   [n,k]
    std::vector<double> TB(size_t(n) * k), TA(size_t(n) * k);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) TB[size_t(i) * k + j] = C[size_t(i) * n + order[j]], TA[size_t(i) * k + j] = V[size_t(i) * n + order[j]];
    solve_upper(Rb, n, TB, k);
    solve_upper(Ra, n, TA, k);
    std::vector<float> Bn(size_t(out) * k), An(size_t(k) * in, 0.f);
    for (int o = 0; o < out; ++o) {
        const float *b = &B[size_t(o) * n];
        for (int j = 0; j < k; ++j) {
            double d = 0;
            for (int t = 0; t < n; ++t) d += b[t] * TB[size_t(t) * k + j];
            Bn[size_t(o) * k + j] = float(d);
        }
    }
    // A' [k,in] = TA^T A
    std::vector<double> acc(in);
    for (int j = 0; j < k; ++j) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int t = 0; t < n; ++t) {
            double w = TA[size_t(t) * k + j];
            if (w == 0.0) continue;
            const float *a = &A[size_t(t) * in];
            for (int c = 0; c < in; ++c) acc[c] += w * a[c];
        }
        for (int c = 0; c < in; ++c) An[size_t(j) * in + c] = float(acc[c]);
    }
    B.swap(Bn);
    A.swap(An);
    return total > 0 ? kept / total : 1.0;
}

std::string fmt_scale(float s) {
    std::ostringstream o;
    o << s;
    return o.str();
}

}  // namespace

LoraManager::LoraManager(std::string lora_dir, const std::string &shared_safetensors, int cache_size)
    : dir_(std::move(lora_dir)), shared_path_(shared_safetensors), cache_size_(std::max(1, cache_size)) {
    // Zero A/B used by blocks a LoRA does not touch.
    size_t total = 0;
    std::array<size_t, 8> sz{};
    for (int p = 0; p < 4; ++p) {
        sz[2 * p] = size_t(kLoraRank) * kIn[p] * 2;
        sz[2 * p + 1] = size_t(kOut[p]) * kLoraRank * 2;
    }
    std::array<size_t, 8> off{};
    for (int i = 0; i < 8; ++i) {
        off[i] = total;
        total += (sz[i] + 255) / 256 * 256;
    }
    zero_ = DeviceBuffer(total, "lora-zero");
    zero_.buf().fill_zero();
    for (int i = 0; i < 8; ++i) zero_ab_[i] = zero_.buf().view(off[i], sz[i]);
}

LoraManager::~LoraManager() = default;

std::vector<std::string> LoraManager::available() const {
    std::vector<std::string> out;
    if (DIR *d = opendir(dir_.c_str())) {
        while (auto *e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) out.push_back(n.substr(0, n.size() - 12));
        }
        closedir(d);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string LoraManager::resolve(const std::string &name) const {
    struct stat st {};
    if (name.find('/') != std::string::npos) {
        if (stat(name.c_str(), &st) == 0) return name;
        fail("LoRA file not found: " + name);
    }
    std::string p = dir_ + "/" + name;
    if (stat(p.c_str(), &st) == 0) return p;
    p += ".safetensors";
    if (stat(p.c_str(), &st) == 0) return p;
    fail("LoRA '" + name + "' not found in " + dir_);
}

const LoraSet &LoraManager::get(const std::vector<LoraSpec> &specs) {
    std::vector<LoraSpec> sorted = specs;
    std::sort(sorted.begin(), sorted.end(), [](auto &a, auto &b) { return a.name < b.name; });
    std::string key;
    for (auto &s : sorted) key += s.name + "@" + fmt_scale(s.scale) + ";";
    for (auto it = cache_.begin(); it != cache_.end(); ++it)
        if ((*it)->key == key) {
            cache_.splice(cache_.begin(), cache_, it);
            return *cache_.front();
        }
    double t0 = now_s();
    auto set = build(sorted, key);
    AXD_LOG("LoRA set [%s] built in %.2f s (%.1f MiB)", key.c_str(), now_s() - t0, set->bytes / 1048576.0);
    cache_.push_front(std::move(set));
    while (int(cache_.size()) > cache_size_) cache_.pop_back();
    return *cache_.front();
}

std::unique_ptr<LoraSet> LoraManager::build(const std::vector<LoraSpec> &specs, const std::string &key) {
    std::map<Target, std::vector<Piece>> pieces;
    for (auto &spec : specs) {
        SafeTensors st(resolve(spec.name));
        // group tensors by module
        struct Mod {
            const TensorView *A = nullptr, *B = nullptr;
            float alpha = -1;
        };
        std::map<std::string, Mod> mods;
        std::vector<std::string> unknown;
        for (auto &k : st.names()) {
            std::string n = normalize(k);
            auto ends = [&](const char *suf) {
                size_t l = std::strlen(suf);
                return n.size() > l && n.compare(n.size() - l, l, suf) == 0;
            };
            std::string mod;
            if (ends(".lora_A.weight") || ends(".lora_down.weight")) {
                mod = n.substr(0, n.rfind(".lora_"));
                mods[mod].A = &st.get(k);
            } else if (ends(".lora_B.weight") || ends(".lora_up.weight")) {
                mod = n.substr(0, n.rfind(".lora_"));
                mods[mod].B = &st.get(k);
            } else if (ends(".alpha")) {
                mod = n.substr(0, n.size() - 6);
                mods[mod].alpha = to_f32(st.get(k))[0];
            } else {
                unknown.push_back(k);
            }
        }
        for (auto &kv : mods) {
            Target t{};
            int off = 0, rows = 0;
            bool swap = false;
            if (!map_module(kohya_to_dotted(kv.first), t, off, rows, swap)) {
                unknown.push_back(kv.first);
                continue;
            }
            auto &m = kv.second;
            if (!m.A || !m.B) fail(spec.name + ": module " + kv.first + " is missing lora_A or lora_B");
            int r = int(m.A->shape.at(0));
            int in = int(m.A->shape.at(1));
            if (in != kIn[t.proj] || m.B->shape.at(1) != r || m.B->shape.at(0) != rows)
                fail(spec.name + ": " + kv.first + " has shape A[" + std::to_string(r) + "," + std::to_string(in) +
                     "] B[" + std::to_string(m.B->shape.at(0)) + "," + std::to_string(m.B->shape.at(1)) +
                     "], expected A[r," + std::to_string(kIn[t.proj]) + "] B[" + std::to_string(rows) + ",r]");
            Piece p;
            p.rank = r;
            p.A = to_f32(*m.A);
            p.B = to_f32(*m.B);
            p.row_offset = off;
            p.swap_halves = swap;
            p.scale = spec.scale * (m.alpha > 0 ? m.alpha / float(r) : 1.f);
            pieces[t].push_back(std::move(p));
        }
        if (!unknown.empty()) {
            std::string msg = spec.name + ": unsupported LoRA keys (" + std::to_string(unknown.size()) + "):";
            for (size_t i = 0; i < unknown.size() && i < 12; ++i) msg += " " + unknown[i];
            fail(msg);
        }
    }

    auto set = std::make_unique<LoraSet>();
    set->key = key;
    set->block_ab.assign(kDitLayers, zero_ab_);
    set->block_active.assign(kDitLayers, false);

    // Build padded bf16 A [64,in] and B [out,64] for one target. If the combined rank exceeds the kernel's
    // rank, B*A is replaced by its best rank-64 approximation (truncated SVD).
    auto pack = [&](const std::vector<Piece> &ps, Proj proj, std::vector<uint16_t> &A, std::vector<uint16_t> &B) {
        const int in = kIn[proj], out = kOut[proj];
        int total = 0;
        for (auto &p : ps) total += p.rank;
        // dense combined factors: Bt [out,total] (scale, row offset and half swap applied), At [total,in]
        std::vector<float> At(size_t(total) * in), Bt(size_t(out) * total, 0.f);
        int r0 = 0;
        for (auto &p : ps) {
            std::copy(p.A.begin(), p.A.end(), At.begin() + long(r0) * in);
            const int rows = int(p.B.size() / p.rank), half = out / 2;
            for (int o = 0; o < rows; ++o) {
                int dst = p.row_offset + o;
                if (p.swap_halves) dst = dst < half ? dst + half : dst - half;
                for (int r = 0; r < p.rank; ++r) Bt[size_t(dst) * total + r0 + r] = p.scale * p.B[size_t(o) * p.rank + r];
            }
            r0 += p.rank;
        }
        int rank = total;
        if (total > kLoraRank) {
            double kept = truncate_rank(Bt, At, out, in, total, kLoraRank);
            AXD_LOG("LoRA rank %d > %d on %s: truncated SVD keeps %.2f%% of the update energy", total, kLoraRank,
                    kProjName[proj], kept * 100.0);
            rank = kLoraRank;
        }
        A.assign(size_t(kLoraRank) * in, 0);
        B.assign(size_t(out) * kLoraRank, 0);
        for (int r = 0; r < rank; ++r)
            for (int k = 0; k < in; ++k) A[size_t(r) * in + k] = f32_to_bf16(At[size_t(r) * in + k]);
        for (int o = 0; o < out; ++o)
            for (int r = 0; r < rank; ++r) B[size_t(o) * kLoraRank + r] = f32_to_bf16(Bt[size_t(o) * rank + r]);
    };

    // ---- DiT blocks: one allocation for all active blocks
    std::vector<int> active;
    for (int i = 0; i < kDitLayers; ++i)
        for (int p = 0; p < 4; ++p)
            if (pieces.count({false, i, Proj(p)})) {
                active.push_back(i);
                break;
            }
    std::array<size_t, 8> sz{};
    size_t per_block = 0;
    for (int i = 0; i < 8; ++i) {
        sz[i] = zero_ab_[i].size;
        per_block += (sz[i] + 255) / 256 * 256;
    }
    if (!active.empty()) {
        set->blocks = DeviceBuffer(per_block * active.size(), "lora-blocks");
        set->bytes += set->blocks.size();
        // pack all (block, projection) targets in parallel (SVD truncation can be heavy), then upload
        std::vector<std::vector<uint16_t>> packed(active.size() * 8);
        parallel_for(active.size() * 4, [&](size_t t0, size_t t1) {
            for (size_t t = t0; t < t1; ++t) {
                int b = active[t / 4], p = int(t % 4);
                auto it = pieces.find({false, b, Proj(p)});
                pack(it != pieces.end() ? it->second : std::vector<Piece>{}, Proj(p), packed[t * 2], packed[t * 2 + 1]);
            }
        });
        for (size_t ai = 0; ai < active.size(); ++ai) {
            int b = active[ai];
            size_t off = per_block * ai;
            std::array<Buf, 8> ab;
            for (int p = 0; p < 4; ++p) {
                auto &A = packed[(ai * 4 + p) * 2];
                auto &B = packed[(ai * 4 + p) * 2 + 1];
                ab[2 * p] = set->blocks.buf().view(off, sz[2 * p]);
                off += (sz[2 * p] + 255) / 256 * 256;
                ab[2 * p + 1] = set->blocks.buf().view(off, sz[2 * p + 1]);
                off += (sz[2 * p + 1] + 255) / 256 * 256;
                ab[2 * p].upload(A.data(), A.size() * 2);
                ab[2 * p + 1].upload(B.data(), B.size() * 2);
                std::vector<uint16_t>().swap(A);
                std::vector<uint16_t>().swap(B);
            }
            set->block_ab[b] = ab;
            set->block_active[b] = true;
        }
    }

    // ---- token refiner: merge W' = W + B A on the host (bf16 weights, 2 layers)
    bool refiner_touched = false;
    for (auto &kv : pieces) refiner_touched |= kv.first.refiner;
    if (refiner_touched) {
        SafeTensors shared(shared_path_);
        set->refiner = std::make_unique<TensorBank>("lora-refiner");
        std::vector<std::vector<uint16_t>> keep;
        keep.reserve(8);
        for (auto &kv : pieces) {
            if (!kv.first.refiner) continue;
            const Proj proj = kv.first.proj;
            const int in = kIn[proj], out = kOut[proj];
            std::vector<uint16_t> A, B;
            pack(kv.second, proj, A, B);
            std::string name = "refiner." + std::to_string(kv.first.index) + "." + kProjName[proj];
            auto &w = shared.get(name);
            auto *wp = reinterpret_cast<const uint16_t *>(w.data);
            std::vector<float> Af(A.size());
            for (size_t i = 0; i < A.size(); ++i) Af[i] = bf16_to_f32(A[i]);
            keep.emplace_back(size_t(out) * in);
            auto &merged = keep.back();
            parallel_for(size_t(out), [&](size_t o0, size_t o1) {
                std::vector<float> row(in);
                for (size_t o = o0; o < o1; ++o) {
                    for (int k = 0; k < in; ++k) row[k] = bf16_to_f32(wp[o * in + k]);
                    for (int r = 0; r < kLoraRank; ++r) {
                        float b = bf16_to_f32(B[o * kLoraRank + r]);
                        if (b == 0.f) continue;
                        const float *a = &Af[size_t(r) * in];
#pragma omp simd
                        for (int k = 0; k < in; ++k) row[k] += b * a[k];
                    }
                    for (int k = 0; k < in; ++k) merged[o * in + k] = f32_to_bf16(row[k]);
                }
            });
            set->refiner->add(name, merged.data(), merged.size() * 2);
        }
        set->refiner->materialize();
        set->bytes += set->refiner->bytes();
    }
    return set;
}

}  // namespace axd
