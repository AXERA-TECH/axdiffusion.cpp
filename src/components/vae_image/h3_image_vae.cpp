// Image VAE decoder (2D ViT, 24 layers) on the NPU, decoded in 256 px tiles and blended on the host.
#include <algorithm>
#include <cmath>
#include <deque>

#include "common.hpp"
#include "h3_dit.hpp"
#include "h3_image_vae.hpp"
#include "safetensors.hpp"

namespace axd {

namespace {
constexpr int kVaeHidden = 1024, kVaeHeads = 16, kVaeHeadDim = 64, kTile = 256, kTileLatent = 16, kSuffix = 5;
constexpr int kSeq = kTileLatent * kTileLatent + kSuffix;  // 261
constexpr int kPatchOut = 768;                               // 3 x 16 x 16
constexpr int kRatio = 16, kMinOverlap = 64, kLayers = 24;

const float kVaeMean[24] = {0.858090341091156f,   -0.9606591463088989f, 1.0661640167236328f,  -0.5090325474739075f,
                            -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
                            -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f,  0.19690127670764923f,
                            -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
                            -0.30133944749832153f, 0.211341992020607f,  -1.1206848621368408f, 0.3581933379173279f,
                            -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f};
const float kVaeStd[24] = {1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f, 1.7549455165863037f,
                           1.5636216402053833f, 2.194143533706665f,  0.9653137922286987f, 1.0569885969161987f,
                           0.841948926448822f,  0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
                           0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f, 0.6936293244361877f,
                           2.961095094680786f,  2.7694199085235596f, 3.0496184825897217f, 2.1088054180145264f,
                           3.276226282119751f,  3.1627357006073f,    2.2816812992095947f, 2.6127843856811523f};
const float kPixelMean[3] = {0.485f, 0.456f, 0.406f};
const float kPixelStd[3] = {0.229f, 0.224f, 0.225f};

// diffusers-compatible tile layout along one axis (pixels).
void split_tiles(int length, std::vector<int> &starts, std::vector<int> &overlaps) {
    starts.clear();
    overlaps.clear();
    if (kTile >= length) {
        starts.push_back(0);
        return;
    }
    int count = (length + kTile - 1) / kTile;
    while (kTile * count - kMinOverlap * (count - 1) - length < 0) ++count;
    overlaps.assign(count - 1, kMinOverlap);
    int sum = kMinOverlap * (count - 1);
    int remaining = kTile * count - sum - length;
    for (int i = 0; i < remaining / kRatio; ++i) overlaps[i % (count - 1)] += kRatio;
    starts.push_back(0);
    for (int i = 0; i < count - 1; ++i) starts.push_back(starts.back() + kTile - overlaps[i]);
}
}  // namespace

struct Vae::Batch {
    int n;
    DeviceBuffer tokens, hidden_a, hidden_b, q, k, v, patch;
};

Vae::Vae(const ModelDir &dir, const VaeWeights &w) : w_(w) {
    const std::pair<const char *, const char *> kinds[] = {{"embed", "image-vae-embed"},
                                                           {"attn_proj", "image-vae-attn-proj"},
                                                           {"attn_out", "image-vae-attn-out"},
                                                           {"ffn", "image-vae-block-ffn"},
                                                           {"final", "image-vae-final"}};
    for (auto &k : kinds)
        for (int b : {1, 5}) models_[k.first][b] = std::make_unique<Model>(dir.axmodel(k.second, b == 1 ? "1 x" : "5 x"));
    cos_ = DeviceBuffer(kSeq * 16 * 2, "vae-cos");
    sin_ = DeviceBuffer(kSeq * 16 * 2, "vae-sin");
    dummy_bf16_ = DeviceBuffer(2, "vae-dummy-bf16");
    dummy_f32_ = DeviceBuffer(4, "vae-dummy-f32");
    dummy_bf16_.buf().fill_zero();
    dummy_f32_.buf().fill_zero();
    // 2D RoPE for one 16x16 tile + 5 suffix tokens at position 0; theta 100, 8 frequencies per axis.
    std::vector<uint16_t> c(kSeq * 16, 0), s(kSeq * 16, 0);
    float inv[8];
    for (int i = 0; i < 8; ++i) inv[i] = 1.0f / std::pow(100.0f, float(i) / 8.0f);
    for (int t = 0; t < kSeq; ++t) {
        float id[2] = {0.f, 0.f};
        if (t < kTileLatent * kTileLatent) {
            id[0] = 2.0f * ((float(t / kTileLatent) + 0.5f) / kTileLatent) - 1.0f;
            id[1] = 2.0f * ((float(t % kTileLatent) + 0.5f) / kTileLatent) - 1.0f;
        }
        for (int a = 0; a < 2; ++a)
            for (int i = 0; i < 8; ++i) {
                float ang = float(2.0 * M_PI) * id[a] * inv[i];
                c[t * 16 + a * 8 + i] = f32_to_bf16(std::cos(ang));
                s[t * 16 + a * 8 + i] = f32_to_bf16(std::sin(ang));
            }
    }
    cos_.buf().upload(c.data(), c.size() * 2);
    sin_.buf().upload(s.data(), s.size() * 2);
    batch(1);
    batch(5);
}

Vae::~Vae() = default;

Vae::Batch &Vae::batch(int n) {
    auto &slot = batches_[n];
    if (slot) return *slot;
    auto b = std::make_unique<Batch>();
    b->n = n;
    const size_t hd = size_t(n) * kVaeHeads * kSeq * kVaeHeadDim * 2;
    b->tokens = DeviceBuffer(size_t(n) * 256 * kLatentChannels * 2, "vae-tokens");
    b->hidden_a = DeviceBuffer(size_t(n) * kSeq * kVaeHidden * 2, "vae-hidden-a");
    b->hidden_b = DeviceBuffer(size_t(n) * kSeq * kVaeHidden * 2, "vae-hidden-b");
    b->q = DeviceBuffer(hd, "vae-q");
    b->k = DeviceBuffer(hd, "vae-k");
    b->v = DeviceBuffer(hd, "vae-v");
    b->patch = DeviceBuffer(size_t(n) * 256 * kPatchOut * 2, "vae-patch");
    slot = std::move(b);
    return *slot;
}

// tokens: [n,256,24] bf16 -> pixels [n,3,256,256] float (normalized pixel space)
void Vae::decode_tiles(const std::vector<uint16_t> &tokens, int n, std::vector<float> &pixels) {
    auto &b = batch(n);
    auto &bank = *w_.bank;
    b.tokens.buf().upload(tokens.data(), tokens.size() * 2);
    models_["embed"][n]->run({{"latent_tokens", b.tokens},
                              {"post_quant_weight", bank.get("post_quant_weight")},
                              {"post_quant_bias", bank.get("post_quant_bias")},
                              {"embed_weight", bank.get("embed_weight")},
                              {"embed_bias", bank.get("embed_bias")},
                              {"register_tokens", bank.get("register_tokens")}},
                             {{"hidden_out", b.hidden_a}});
    for (int l = 0; l < kLayers; ++l) {
        auto g = [&](const char *k) { return bank.get(std::to_string(l) + "." + k); };
        models_["attn_proj"][n]->run({{"hidden", b.hidden_a},
                                      {"q_all", b.q},
                                      {"k_all", b.k},
                                      {"v_all", b.v},
                                      {"cos", cos_},
                                      {"sin", sin_},
                                      {"norm1_weight", g("norm1")},
                                      {"qkv_weight", g("qkv_weight")},
                                      {"qkv_bias", g("qkv_bias")},
                                      {"qkv_scale", dummy_bf16_},
                                      {"qkv_scale2", dummy_f32_}},
                                     {{"q_all", b.q}});
        models_["attn_out"][n]->run({{"hidden", b.hidden_a},
                                     {"q_all", b.q},
                                     {"k_all", b.k},
                                     {"v_all", b.v},
                                     {"out_weight", g("out_weight")},
                                     {"out_bias", g("out_bias")},
                                     {"out_scale", dummy_bf16_},
                                     {"out_scale2", dummy_f32_},
                                     {"scale1", g("scale1")}},
                                    {{"hidden_out", b.hidden_b}});
        models_["ffn"][n]->run({{"hidden", b.hidden_b},
                                {"norm2_weight", g("norm2")},
                                {"w1_weight", g("w1_weight")},
                                {"w1_bias", g("w1_bias")},
                                {"w1_scale", dummy_bf16_},
                                {"w1_scale2", dummy_f32_},
                                {"w2_weight", g("w2_weight")},
                                {"w2_bias", g("w2_bias")},
                                {"w2_scale", dummy_bf16_},
                                {"w2_scale2", dummy_f32_},
                                {"scale2", g("scale2")}},
                               {{"hidden_out", b.hidden_a}});
    }
    models_["final"][n]->run({{"hidden", b.hidden_a},
                              {"norm_weight", bank.get("norm_weight")},
                              {"norm_bias", bank.get("norm_bias")},
                              {"proj_weight", bank.get("proj_weight")},
                              {"proj_bias", bank.get("proj_bias")}},
                             {{"patch_tokens", b.patch}});
    std::vector<uint16_t> patch(size_t(n) * 256 * kPatchOut);
    b.patch.buf().download(patch.data(), patch.size() * 2);
    // patch [n,16,16,3,16,16] -> pixels [n,3,256,256]
    pixels.resize(size_t(n) * 3 * kTile * kTile);
    for (int t = 0; t < n; ++t)
        for (int ty = 0; ty < 16; ++ty)
            for (int tx = 0; tx < 16; ++tx)
                for (int c = 0; c < 3; ++c)
                    for (int py = 0; py < 16; ++py)
                        for (int px = 0; px < 16; ++px) {
                            size_t src = ((((size_t(t) * 16 + ty) * 16 + tx) * 3 + c) * 16 + py) * 16 + px;
                            size_t dst = ((size_t(t) * 3 + c) * kTile + ty * 16 + py) * kTile + tx * 16 + px;
                            pixels[dst] = bf16_to_f32(patch[src]);
                        }
}

std::vector<uint8_t> Vae::decode(const std::vector<float> &latents, int h, int w, double &seconds) {
    double t0 = now_s();
    const int H = h * kRatio, W = w * kRatio;
    // de-normalize latents
    std::vector<float> z(latents.size());
    for (int c = 0; c < kLatentChannels; ++c)
        for (size_t i = 0; i < size_t(h) * w; ++i) z[c * size_t(h) * w + i] = latents[c * size_t(h) * w + i] * kVaeStd[c] + kVaeMean[c];
    std::vector<int> ys, yo, xs, xo;
    split_tiles(H, ys, yo);
    split_tiles(W, xs, xo);
    const int ny = int(ys.size()), nx = int(xs.size());
    // decode every tile, batched by 5 across the whole grid (row-major), the remainder one by one
    const int total = ny * nx;
    std::vector<std::vector<float>> tiles(static_cast<size_t>(total));  // each [3,256,256]
    for (int t0 = 0; t0 < total;) {
        const int n = (total - t0 >= 5) ? 5 : 1;
        std::vector<uint16_t> tok(size_t(n) * 256 * kLatentChannels);
        for (int t = 0; t < n; ++t) {
            const int iy = (t0 + t) / nx, ix = (t0 + t) % nx;
            const int ly = ys[iy] / kRatio, lx = xs[ix] / kRatio;
            for (int y = 0; y < kTileLatent; ++y)
                for (int x = 0; x < kTileLatent; ++x)
                    for (int c = 0; c < kLatentChannels; ++c)
                        tok[(size_t(t) * 256 + y * 16 + x) * kLatentChannels + c] =
                            f32_to_bf16(z[(size_t(c) * h + ly + y) * w + lx + x]);
        }
        std::vector<float> px;
        decode_tiles(tok, n, px);
        for (int t = 0; t < n; ++t)
            tiles[size_t(t0 + t)].assign(px.begin() + long(t) * 3 * kTile * kTile,
                                         px.begin() + long(t + 1) * 3 * kTile * kTile);
        t0 += n;
    }
    // Blend overlaps against the original (unblended) neighbours, then crop each tile's trailing overlap.
    std::vector<float> img(size_t(3) * H * W, 0.f);
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
            const auto &cur = tiles[size_t(iy) * nx + ix];
            const int top = iy > 0 ? yo[iy - 1] : 0, left = ix > 0 ? xo[ix - 1] : 0;
            const int keep_h = iy < ny - 1 ? kTile - yo[iy] : kTile;
            const int keep_w = ix < nx - 1 ? kTile - xo[ix] : kTile;
            const auto *up = iy > 0 ? &tiles[size_t(iy - 1) * nx + ix] : nullptr;
            const auto *lf = ix > 0 ? &tiles[size_t(iy) * nx + ix - 1] : nullptr;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < keep_h; ++y)
                    for (int x = 0; x < keep_w; ++x) {
                        auto at = [&](const std::vector<float> &t, int yy, int xx) {
                            return t[(size_t(c) * kTile + yy) * kTile + xx];
                        };
                        float v = at(cur, y, x);
                        if (up && y < top) {  // vertical blend with the tile above
                            float wb = float(y) / top;
                            v = at(*up, kTile - top + y, x) * (1.f - wb) + v * wb;
                        }
                        if (lf && x < left) {  // horizontal blend with the original left tile
                            float wb = float(x) / left;
                            v = at(*lf, y, kTile - left + x) * (1.f - wb) + v * wb;
                        }
                        const int oy = ys[iy] + y, ox = xs[ix] + x;
                        if (oy < H && ox < W) img[(size_t(c) * H + oy) * W + ox] = v;
                    }
        }
    std::vector<uint8_t> rgb(size_t(H) * W * 3);
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < size_t(H) * W; ++i) {
            float v = img[c * size_t(H) * W + i] * kPixelStd[c] + kPixelMean[c];
            v = std::min(1.f, std::max(0.f, v));
            rgb[i * 3 + c] = uint8_t(std::min(255.f, std::max(0.f, v * 255.f)));
        }
    seconds = now_s() - t0;
    return rgb;
}

std::unique_ptr<VaeWeights> load_vae_weights(const ModelDir &dir) {
    constexpr int C = kLatentChannels, H = 1024, HEADS = 16, HD = 64, LAYERS = 24;
    double t0 = now_s();
    SafeTensors st(dir.file("vae/vae_bf16.safetensors"));
    st.read_all({"decoder.", "post_quant_conv."});  // decoder only; permuted on the host, so read it in parallel
    const double t_read = now_s() - t0;
    auto w = std::make_unique<VaeWeights>();
    w->bank = std::make_unique<TensorBank>("vae");
    // bf16 tensors go to the device straight from the mapping; only q/k/v interleaving and the gated-FFN half
    // swap are row permutations (done on the bf16 bytes). Prepared tensors must outlive materialize_parallel().
    std::deque<std::vector<uint16_t>> keep;  // deque: references stay valid while appending
    auto bf16_of = [&](const std::string &name) -> const uint16_t * {
        auto &t = st.get(name);
        if (t.dtype == "BF16") return reinterpret_cast<const uint16_t *>(t.data);
        keep.push_back(to_bf16(t));
        return keep.back().data();
    };
    auto elems = [&](const std::string &name) { size_t n = 1; for (auto d : st.get(name).shape) n *= size_t(d); return n; };
    auto put_bf16 = [&](const std::string &dst, const std::string &src) {
        w->bank->add(dst, bf16_of(src), elems(src) * 2);
    };
    std::deque<std::vector<float>> keep_f32;
    auto put_f32 = [&](const std::string &dst, const std::string &src) {
        keep_f32.push_back(to_f32(st.get(src)));
        w->bank->add(dst, keep_f32.back().data(), keep_f32.back().size() * 4);
    };

    put_bf16("post_quant_weight", "post_quant_conv.weight");  // [24,24,1,1] -> [24,24]
    put_bf16("post_quant_bias", "post_quant_conv.bias");
    put_bf16("embed_weight", "decoder.proj_in.weight");
    put_bf16("embed_bias", "decoder.proj_in.bias");
    put_bf16("register_tokens", "decoder.register_tokens");  // [1,4,1024] -> [4,1024]
    put_f32("norm_weight", "decoder.norm_out.weight");
    put_f32("norm_bias", "decoder.norm_out.bias");
    put_bf16("proj_weight", "decoder.proj_out.weight");
    put_bf16("proj_bias", "decoder.proj_out.bias");

    for (int l = 0; l < LAYERS; ++l) {
        std::string p = "decoder.transformer_blocks." + std::to_string(l);
        std::string k = std::to_string(l) + ".";
        // q/k/v rows interleaved per head: [h0 q, h0 k, h0 v, h1 q, ...]
        const uint16_t *src_w[3] = {bf16_of(p + ".attn.to_q.weight"), bf16_of(p + ".attn.to_k.weight"),
                                    bf16_of(p + ".attn.to_v.weight")};
        const uint16_t *src_b[3] = {bf16_of(p + ".attn.to_q.bias"), bf16_of(p + ".attn.to_k.bias"),
                                    bf16_of(p + ".attn.to_v.bias")};
        keep.emplace_back(size_t(3) * H * H);
        auto &qkv = keep.back();
        keep.emplace_back(size_t(3) * H);
        auto &qkvb = keep.back();
        for (int h = 0; h < HEADS; ++h)
            for (int part = 0; part < 3; ++part) {
                size_t dst_row = size_t(h) * 3 * HD + size_t(part) * HD;
                std::memcpy(&qkv[dst_row * H], src_w[part] + size_t(h) * HD * H, size_t(HD) * H * 2);
                std::memcpy(&qkvb[dst_row], src_b[part] + size_t(h) * HD, HD * 2);
            }
        // gated FFN: swap the two halves of the first projection
        const std::string w1n = p + ".ff.net.0.proj.weight", b1n = p + ".ff.net.0.proj.bias";
        const uint16_t *w1 = bf16_of(w1n), *b1 = bf16_of(b1n);
        const size_t w1_elems = elems(w1n), half_rows = w1_elems / H / 2;
        keep.emplace_back(w1_elems);
        auto &w1s = keep.back();
        std::memcpy(w1s.data(), w1 + half_rows * H, half_rows * H * 2);
        std::memcpy(&w1s[half_rows * H], w1, half_rows * H * 2);
        keep.emplace_back(2 * half_rows);
        auto &b1s = keep.back();
        std::memcpy(b1s.data(), b1 + half_rows, half_rows * 2);
        std::memcpy(&b1s[half_rows], b1, half_rows * 2);

        put_f32(k + "norm1", p + ".norm1.weight");
        w->bank->add(k + "qkv_weight", qkv.data(), qkv.size() * 2);
        w->bank->add(k + "qkv_bias", qkvb.data(), qkvb.size() * 2);
        put_bf16(k + "out_weight", p + ".attn.to_out.0.weight");
        put_bf16(k + "out_bias", p + ".attn.to_out.0.bias");
        put_f32(k + "scale1", p + ".scale1");
        put_f32(k + "norm2", p + ".norm2.weight");
        w->bank->add(k + "w1_weight", w1s.data(), w1s.size() * 2);
        w->bank->add(k + "w1_bias", b1s.data(), b1s.size() * 2);
        put_bf16(k + "w2_weight", p + ".ff.net.2.weight");
        put_bf16(k + "w2_bias", p + ".ff.net.2.bias");
        put_f32(k + "scale2", p + ".scale2");
    }
    (void)C;
    const double t_prep = now_s() - t0 - t_read;
    materialize_parallel({w->bank.get()});
    AXD_LOG("VAE weights: read %.1f s, prepare %.1f s, upload %.1f s", t_read, t_prep, now_s() - t0 - t_read - t_prep);
    return w;
}


}  // namespace axd
