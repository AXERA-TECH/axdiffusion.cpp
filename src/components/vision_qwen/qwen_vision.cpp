#include "qwen_vision.hpp"

#include <cmath>
#include <cstdlib>

#include "common.hpp"
#include "safetensors.hpp"

namespace axd {

namespace {
constexpr int kPatchDim = 1536;  // 3 * 2 (temporal) * 16 * 16
constexpr int kFrameDim = 768;   // 3 * 16 * 16
constexpr int kHidden = 1152, kHeads = 16, kHeadDim = 72, kOutDim = 5120, kDepth = 27;
constexpr int kPosSide = 48, kRotary = kHeadDim / 2, kSinCosWidth = 64;
constexpr int kRowTile = 128, kQueryBlock = 512;
const char *const kKernels[] = {"embed", "pre", "attn", "post", "merge", "merge-ds"};
const char *const kWeights = "vit/visual.safetensors";
const std::string kPrefix = "model.visual.";

// Raster index of each patch in merge-block order.
std::vector<int> merge_order(int gh, int gw) {
    std::vector<int> o;
    o.reserve(size_t(gh) * gw);
    for (int bh = 0; bh < gh / 2; ++bh)
        for (int bw = 0; bw < gw / 2; ++bw)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) o.push_back((bh * 2 + i) * gw + bw * 2 + j);
    return o;
}
}  // namespace

bool VisionEncoder::available(const ModelDir &dir) {
    for (auto k : kKernels)
        if (dir.tiers(std::string("vision-vit-") + k, "cap ").empty()) return false;
    return dir.exists(kWeights);
}

VisionEncoder::VisionEncoder(const ModelDir &dir) {
    for (auto k : kKernels)
        for (auto &kv : dir.tiers(std::string("vision-vit-") + k, "cap ")) tiers_[kv.first].paths[k] = kv.second;
    if (tiers_.empty()) fail("model directory has no vision-vit kernels");
    for (auto &t : tiers_)
        if (t.second.paths.size() != std::size(kKernels))
            fail("vision-vit capacity " + std::to_string(t.first) + " is missing kernels");

    SafeTensors st(dir.file(kWeights));
    bank_ = std::make_unique<TensorBank>("vit-weights");
    auto add = [&](const std::string &name) {
        auto &t = st.get(kPrefix + name);
        if (t.dtype != "BF16") fail(std::string(kWeights) + " " + name + " must be bf16");
        bank_->add_file(name, st.fd(), st.file_offset(t), t.nbytes);
    };
    for (int l = 0; l < kDepth; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".";
        for (auto n : {"norm1", "attn.qkv", "attn.proj", "norm2", "mlp.linear_fc1", "mlp.linear_fc2"})
            for (auto s : {".weight", ".bias"}) add(p + n + s);
    }
    for (std::string p : {"merger.", "deepstack_merger_list.0.", "deepstack_merger_list.1.", "deepstack_merger_list.2."})
        for (auto n : {"norm", "linear_fc1", "linear_fc2"})
            for (auto s : {".weight", ".bias"}) add(p + n + s);
    add("patch_embed.proj.bias");
    {  // a still image repeats its frame, so the two temporal kernels of the Conv3d add: [1152,3,2,16,16] -> [1152,768]
        auto w = to_f32(st.get(kPrefix + "patch_embed.proj.weight"));
        if (w.size() != size_t(kHidden) * kPatchDim) fail("unexpected vision patch_embed shape");
        patch_weight_.resize(size_t(kHidden) * kFrameDim);
        for (int o = 0; o < kHidden; ++o)
            for (int c = 0; c < 3; ++c)
                for (int p = 0; p < 256; ++p) {
                    const float *src = &w[(size_t(o) * 3 + c) * 512];
                    patch_weight_[size_t(o) * kFrameDim + c * 256 + p] = f32_to_bf16(src[p] + src[256 + p]);
                }
        bank_->add("patch_embed.proj.weight", patch_weight_.data(), patch_weight_.size() * 2);
    }
    pos_table_ = to_f32(st.get(kPrefix + "pos_embed.weight"));
    if (pos_table_.size() != size_t(kPosSide) * kPosSide * kHidden) fail("unexpected vision pos_embed shape");
    materialize_parallel({bank_.get()});
    bounds_ = DeviceBuffer(12, "vit-bounds");
}

VisionEncoder::~VisionEncoder() = default;

int VisionEncoder::max_patches() const {
    // every core writes one flush block past the last query block
    return (tiers_.rbegin()->first / kQueryBlock - 1) * kQueryBlock;
}

Model &VisionEncoder::kernel(Tier &t, const std::string &name) {
    auto &m = t.models[name];
    if (!m) m = std::make_unique<Model>(t.paths.at(name));
    return *m;
}

void VisionEncoder::workspace(int cap) {
    if (cap == cap_) return;
    const size_t rows = size_t(cap);
    px_ = DeviceBuffer(rows * kFrameDim * 2, "vit-px");
    pos_ = DeviceBuffer(rows * kHidden * 2, "vit-pos");
    cos_ = DeviceBuffer(rows * kSinCosWidth * 2, "vit-cos");
    sin_ = DeviceBuffer(rows * kSinCosWidth * 2, "vit-sin");
    for (auto &x : x_) x = DeviceBuffer(rows * kHidden * 2, "vit-hidden");
    for (auto *b : {&q_, &k_, &v_, &attn_}) *b = DeviceBuffer(rows * kHidden * 2, "vit-attn");
    for (auto &o : out_) o = DeviceBuffer(rows / 4 * kOutDim * 2, "vit-out");
    cap_ = cap;
}

VisionFeatures VisionEncoder::encode(const std::vector<float> &pixel_values, int gh, int gw) {
    const int used = gh * gw;
    if (gh % 2 || gw % 2 || used % kRowTile)
        fail("vision grid " + std::to_string(gh) + "x" + std::to_string(gw) + " is not supported");
    if (pixel_values.size() != size_t(used) * kPatchDim) fail("vision pixel_values size mismatch");
    const int need = (used + kQueryBlock - 1) / kQueryBlock * kQueryBlock + kQueryBlock;
    auto it = tiers_.lower_bound(need);
    if (it == tiers_.end())
        fail("the reference image is too elongated for the vision kernels (" + std::to_string(gw * 16) + "x" +
             std::to_string(gh * 16) + " after resizing; at most " + std::to_string(max_patches()) + " patches)");
    const int cap = it->first;
    Tier &tier = it->second;
    workspace(cap);

    // host tables for this grid: first-frame pixels, interpolated positions, rotary cos/sin (zero in the padded tail)
    const auto order = merge_order(gh, gw);
    {
        std::vector<uint16_t> px(size_t(cap) * kFrameDim, 0);
        for (int r = 0; r < used; ++r)
            for (int c = 0; c < 3; ++c) {
                const float *src = &pixel_values[size_t(r) * kPatchDim + c * 512];
                uint16_t *dst = &px[size_t(r) * kFrameDim + c * 256];
                for (int p = 0; p < 256; ++p) dst[p] = f32_to_bf16(src[p]);
            }
        px_.buf().upload(px.data(), px.size() * 2);
    }
    {
        std::vector<int> h0(gh), h1(gh), w0(gw), w1(gw);
        std::vector<float> hf(gh), wf(gw);
        auto axis = [](int n, std::vector<int> &lo, std::vector<int> &hi, std::vector<float> &frac) {
            for (int i = 0; i < n; ++i) {
                float g = n > 1 ? float(double(i) * (kPosSide - 1) / (n - 1)) : 0.0f;
                lo[i] = int(g);
                hi[i] = std::min(lo[i] + 1, kPosSide - 1);
                frac[i] = g - float(lo[i]);
            }
        };
        axis(gh, h0, h1, hf);
        axis(gw, w0, w1, wf);
        std::vector<uint16_t> pos(size_t(cap) * kHidden, 0);
        for (int r = 0; r < used; ++r) {
            const int y = order[r] / gw, x = order[r] % gw;
            const float *a = &pos_table_[size_t(h0[y] * kPosSide + w0[x]) * kHidden];
            const float *b = &pos_table_[size_t(h0[y] * kPosSide + w1[x]) * kHidden];
            const float *c = &pos_table_[size_t(h1[y] * kPosSide + w0[x]) * kHidden];
            const float *d = &pos_table_[size_t(h1[y] * kPosSide + w1[x]) * kHidden];
            const float wa = (1 - hf[y]) * (1 - wf[x]), wb = (1 - hf[y]) * wf[x], wc = hf[y] * (1 - wf[x]),
                        wd = hf[y] * wf[x];
            uint16_t *o = &pos[size_t(r) * kHidden];
            for (int i = 0; i < kHidden; ++i) o[i] = f32_to_bf16(a[i] * wa + b[i] * wb + c[i] * wc + d[i] * wd);
        }
        pos_.buf().upload(pos.data(), pos.size() * 2);
    }
    {
        float inv[kRotary / 2];
        for (int j = 0; j < kRotary / 2; ++j) inv[j] = 1.0f / std::pow(10000.0f, float(2 * j) / kRotary);
        std::vector<uint16_t> cs(size_t(cap) * kSinCosWidth, 0), sn(cs.size(), 0);
        for (int r = 0; r < used; ++r) {
            const int y = order[r] / gw, x = order[r] % gw;
            for (int j = 0; j < kRotary / 2; ++j) {
                const float ay = float(y) * inv[j], ax = float(x) * inv[j];
                cs[size_t(r) * kSinCosWidth + j] = f32_to_bf16(std::cos(ay));
                sn[size_t(r) * kSinCosWidth + j] = f32_to_bf16(std::sin(ay));
                cs[size_t(r) * kSinCosWidth + kRotary / 2 + j] = f32_to_bf16(std::cos(ax));
                sn[size_t(r) * kSinCosWidth + kRotary / 2 + j] = f32_to_bf16(std::sin(ax));
            }
        }
        cos_.buf().upload(cs.data(), cs.size() * 2);
        sin_.buf().upload(sn.data(), sn.size() * 2);
    }
    const int32_t b[3] = {0, used, cap};
    bounds_.buf().upload(b, sizeof(b));

    if (!stream_) stream_ = std::make_unique<Stream>();
    auto w = [&](const std::string &name) { return bank_->get(name); };
    const Buf bounds = bounds_.buf();
    Model &embed = kernel(tier, "embed"), &pre = kernel(tier, "pre"), &attn = kernel(tier, "attn"),
          &post = kernel(tier, "post"), &merge = kernel(tier, "merge"), &merge_ds = kernel(tier, "merge-ds");
    embed.submit(*stream_,
                 {{"sequence_bounds", bounds}, {"px", px_}, {"w_patch", w("patch_embed.proj.weight")},
                  {"b_patch", w("patch_embed.proj.bias")}, {"pos", pos_}},
                 {{"out", x_[0]}});
    auto merger = [&](Model &m, const std::string &p, const Buf &x, const Buf &out) {
        m.submit(*stream_,
                 {{"sequence_bounds", bounds}, {"x", x}, {"norm_w", w(p + "norm.weight")}, {"norm_b", w(p + "norm.bias")},
                  {"w0", w(p + "linear_fc1.weight")}, {"b0", w(p + "linear_fc1.bias")},
                  {"w2", w(p + "linear_fc2.weight")}, {"b2", w(p + "linear_fc2.bias")}},
                 {{"out", out}});
    };
    int cur = 0, deep = 0;
    for (int l = 0; l < kDepth; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".";
        pre.submit(*stream_,
                   {{"sequence_bounds", bounds}, {"x", x_[cur]}, {"cos", cos_}, {"sin", sin_},
                    {"ln1_w", w(p + "norm1.weight")}, {"ln1_b", w(p + "norm1.bias")},
                    {"w_qkv", w(p + "attn.qkv.weight")}, {"b_qkv", w(p + "attn.qkv.bias")}},
                   {{"q_out", q_}, {"k_out", k_}, {"v_out", v_}});
        attn.submit(*stream_, {{"sequence_bounds", bounds}, {"q_in", q_}, {"k_in", k_}, {"v_in", v_}}, {{"out", attn_}});
        post.submit(*stream_,
                    {{"sequence_bounds", bounds}, {"x", x_[cur]}, {"attn", attn_},
                     {"w_o", w(p + "attn.proj.weight")}, {"b_o", w(p + "attn.proj.bias")},
                     {"ln2_w", w(p + "norm2.weight")}, {"ln2_b", w(p + "norm2.bias")},
                     {"w_up", w(p + "mlp.linear_fc1.weight")}, {"b_up", w(p + "mlp.linear_fc1.bias")},
                     {"w_down", w(p + "mlp.linear_fc2.weight")}, {"b_down", w(p + "mlp.linear_fc2.bias")}},
                    {{"out", x_[1 - cur]}});
        cur = 1 - cur;
        if (l == 8 || l == 16 || l == 24) {  // deepstack_visual_indexes
            merger(merge_ds, "deepstack_merger_list." + std::to_string(deep) + ".", x_[cur], out_[deep]);
            ++deep;
        }
    }
    merger(merge, "merger.", x_[cur], out_[3]);
    stream_->sync();

    VisionFeatures f;
    f.tokens = used / 4;
    auto get = [&](int i) {
        std::vector<uint16_t> v(size_t(f.tokens) * kOutDim);
        out_[i].buf().download(v.data(), v.size() * 2);
        return v;
    };
    for (int i = 0; i < 3; ++i) f.deepstack[i] = get(i);
    f.merger = get(3);
    return f;
}

}  // namespace axd
