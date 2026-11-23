#include "edit_host.hpp"

#include <cmath>
#include <cstring>

#include "common.hpp"

namespace axd {
namespace edit {

std::array<int, 2> bound_short_edge(int width, int height, int lo, int hi, int multiple) {
    int short_edge = std::min(width, height);
    if (lo <= short_edge && short_edge <= hi) return {width, height};
    double scale = double(short_edge < lo ? lo : hi) / short_edge;
    auto r = [&](int v) { return std::max(multiple, int(std::nearbyint(v * scale / multiple)) * multiple); };
    return {r(width), r(height)};
}

// ------------------------------------------------------------------ Pillow LANCZOS (libImaging/Resample.c)
namespace {
constexpr int kPrecisionBits = 32 - 8 - 2;

double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= M_PI;
    return std::sin(x) / x;
}
double lanczos(double x) { return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3) : 0.0; }

struct Coeffs {
    int ksize = 0;
    std::vector<int> bounds;  // [out][xmin, count]
    std::vector<int32_t> k;   // [out][ksize], fixed point
};

Coeffs precompute(int in_size, int out_size) {
    Coeffs c;
    double scale = double(in_size) / out_size, filterscale = std::max(scale, 1.0);
    double support = 3.0 * filterscale;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.bounds.resize(size_t(out_size) * 2);
    c.k.assign(size_t(out_size) * c.ksize, 0);
    std::vector<double> kk(c.ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        double center = (xx + 0.5) * scale, ww = 0.0, ss = 1.0 / filterscale;
        int xmin = int(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = int(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        std::fill(kk.begin(), kk.end(), 0.0);
        for (int x = 0; x < xmax; ++x) {
            double w = lanczos((x + xmin - center + 0.5) * ss);
            kk[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; ++x)
            if (ww != 0.0) kk[x] /= ww;
        for (int x = 0; x < c.ksize; ++x) {
            double v = kk[x];
            c.k[size_t(xx) * c.ksize + x] =
                v < 0 ? int32_t(-0.5 + v * (1 << kPrecisionBits)) : int32_t(0.5 + v * (1 << kPrecisionBits));
        }
        c.bounds[xx * 2] = xmin;
        c.bounds[xx * 2 + 1] = xmax;
    }
    return c;
}

inline uint8_t clip8(int64_t in) {
    if (in >= (int64_t(1) << kPrecisionBits << 8)) return 255;
    if (in <= 0) return 0;
    return uint8_t(in >> kPrecisionBits);
}
}  // namespace

Image resize_lanczos(const Image &in, int out_w, int out_h) {
    Image cur = in;
    if (out_w != in.width) {  // horizontal pass
        Coeffs c = precompute(in.width, out_w);
        Image t;
        t.width = out_w, t.height = in.height;
        t.rgb.resize(size_t(out_w) * in.height * 3);
        for (int y = 0; y < in.height; ++y)
            for (int xx = 0; xx < out_w; ++xx) {
                int xmin = c.bounds[xx * 2], n = c.bounds[xx * 2 + 1];
                const int32_t *k = &c.k[size_t(xx) * c.ksize];
                for (int ch = 0; ch < 3; ++ch) {
                    int64_t ss = int64_t(1) << (kPrecisionBits - 1);
                    for (int x = 0; x < n; ++x) ss += int64_t(in.rgb[(size_t(y) * in.width + x + xmin) * 3 + ch]) * k[x];
                    t.rgb[(size_t(y) * out_w + xx) * 3 + ch] = clip8(ss);
                }
            }
        cur = std::move(t);
    }
    if (out_h != cur.height) {  // vertical pass
        Coeffs c = precompute(cur.height, out_h);
        Image t;
        t.width = cur.width, t.height = out_h;
        t.rgb.resize(size_t(cur.width) * out_h * 3);
        for (int yy = 0; yy < out_h; ++yy) {
            int ymin = c.bounds[yy * 2], n = c.bounds[yy * 2 + 1];
            const int32_t *k = &c.k[size_t(yy) * c.ksize];
            for (int x = 0; x < cur.width; ++x)
                for (int ch = 0; ch < 3; ++ch) {
                    int64_t ss = int64_t(1) << (kPrecisionBits - 1);
                    for (int y = 0; y < n; ++y) ss += int64_t(cur.rgb[(size_t(y + ymin) * cur.width + x) * 3 + ch]) * k[y];
                    t.rgb[(size_t(yy) * cur.width + x) * 3 + ch] = clip8(ss);
                }
        }
        cur = std::move(t);
    }
    return cur;
}

Image resize_for_vae(const Image &in) {
    auto s = bound_short_edge(in.width, in.height, 1024, 2048, 32);
    return (s[0] == in.width && s[1] == in.height) ? in : resize_lanczos(in, s[0], s[1]);
}

std::array<int, 2> vae_size(int width, int height) {
    auto s = bound_short_edge(width, height, 1024, 2048, 32);
    return {s[0], s[1]};
}

std::array<int, 2> vit_size(int width, int height) {
    auto s = bound_short_edge(width, height, 1024, 1024, 32);
    return {s[0], s[1]};
}

Image resize_for_vit(const Image &in) {
    auto s = bound_short_edge(in.width, in.height, 1024, 1024, 32);
    return (s[0] == in.width && s[1] == in.height) ? in : resize_lanczos(in, s[0], s[1]);
}

std::vector<float> vae_pixels(const Image &img, int &h, int &w) {
    static const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    h = img.height / 16 * 16;
    w = img.width / 16 * 16;
    std::vector<float> out(size_t(3) * h * w);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float v = float(img.rgb[(size_t(y) * img.width + x) * 3 + c]) / 255.0f;
                out[(size_t(c) * h + y) * w + x] = (v - mean[c]) / stdv[c];
            }
    return out;
}

void split_tiles(int length, int tile, int min_overlap, int ratio, std::vector<int> &starts, std::vector<int> &overlaps) {
    starts.assign(1, 0);
    overlaps.clear();
    if (tile >= length) return;
    int count = (length + tile - 1) / tile;
    while (tile * count - min_overlap * (count - 1) - length < 0) ++count;
    overlaps.assign(count - 1, min_overlap);
    int remaining = tile * count - min_overlap * (count - 1) - length;
    for (int i = 0; i < remaining / ratio; ++i) overlaps[i % (count - 1)] += ratio;
    for (int i = 0; i < count - 1; ++i) starts.push_back(starts.back() + tile - overlaps[i]);
}

std::vector<float> stitch_tiles(const std::vector<std::vector<float>> &tiles, int C, int th, int tw,
                                const std::vector<int> &yo, const std::vector<int> &xo, int out_h, int out_w) {
    const int ny = int(yo.size()) + 1, nx = int(xo.size()) + 1;
    std::vector<float> out(size_t(C) * out_h * out_w, 0.f);
    int oy = 0;
    for (int i = 0; i < ny; ++i) {
        const int top = i > 0 ? std::min({th, th, yo[i - 1]}) : 0;
        const int keep_h = i < ny - 1 ? th - yo[i] : th;
        int ox = 0;
        for (int j = 0; j < nx; ++j) {
            const auto &cur = tiles[size_t(i) * nx + j];
            const int left = j > 0 ? std::min(tw, xo[j - 1]) : 0;
            const int keep_w = j < nx - 1 ? tw - xo[j] : tw;
            const auto *up = i > 0 ? &tiles[size_t(i - 1) * nx + j] : nullptr;
            const auto *lf = j > 0 ? &tiles[size_t(i) * nx + j - 1] : nullptr;
            for (int c = 0; c < C; ++c)
                for (int y = 0; y < keep_h; ++y)
                    for (int x = 0; x < keep_w; ++x) {
                        auto at = [&](const std::vector<float> &t, int yy, int xx) {
                            return t[(size_t(c) * th + yy) * tw + xx];
                        };
                        float v = at(cur, y, x);
                        if (up && y < top) {
                            float wb = float(y) / float(top);
                            v = at(*up, th - top + y, x) * (1.f - wb) + v * wb;
                        }
                        if (lf && x < left) {
                            float wb = float(x) / float(left);
                            v = at(*lf, y, tw - left + x) * (1.f - wb) + v * wb;
                        }
                        if (oy + y < out_h && ox + x < out_w) out[(size_t(c) * out_h + oy + y) * out_w + ox + x] = v;
                    }
            ox += keep_w;
        }
        oy += keep_h;
    }
    return out;
}

namespace {
float round_fp16(float f) {
    // float -> IEEE half (round to nearest even) -> float
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = x & 0x80000000u, ax = x & 0x7FFFFFFFu;
    uint16_t h;
    if (ax >= 0x7F800000u) {
        h = uint16_t((sign >> 16) | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u));
    } else if (ax >= 0x477FF000u) {  // overflow -> inf
        h = uint16_t((sign >> 16) | 0x7C00u);
    } else if (ax < 0x38800000u) {  // subnormal half
        float a;
        std::memcpy(&a, &ax, 4);
        int m = int(std::nearbyint(a * 16777216.0f));  // a / 2^-24
        h = uint16_t((sign >> 16) | uint32_t(m));
    } else {
        uint32_t e = (ax >> 23) - 112, mant = ax & 0x7FFFFFu;
        uint32_t v = (e << 10) | (mant >> 13);
        uint32_t rem = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) ++v;
        h = uint16_t((sign >> 16) | v);
    }
    // half -> float
    uint32_t hs = uint32_t(h & 0x8000u) << 16, he = (h >> 10) & 0x1Fu, hm = h & 0x3FFu, bits;
    if (he == 0) {
        float v = std::ldexp(float(hm), -24);
        std::memcpy(&bits, &v, 4);
        bits |= hs;
    } else if (he == 31) {
        bits = hs | 0x7F800000u | (hm << 13);
    } else {
        bits = hs | ((he + 112) << 23) | (hm << 13);
    }
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}
}  // namespace

std::vector<float> posterior_latent(const std::vector<float> &moments, const std::vector<float> &eps, int h, int w,
                                    const float *latent_mean, const float *latent_std) {
    const size_t hw = size_t(h) * w;
    std::vector<float> out(24 * hw);
    for (int c = 0; c < 24; ++c)
        for (size_t i = 0; i < hw; ++i) {
            float mean = moments[c * hw + i];
            float logvar = std::min(20.f, std::max(-30.f, moments[(24 + c) * hw + i]));
            float sample = round_fp16(mean + std::exp(0.5f * logvar) * eps[c * hw + i]);
            out[c * hw + i] = (sample - latent_mean[c]) / latent_std[c];
        }
    return out;
}

std::vector<float> vit_pixel_values(const Image &img, std::array<int, 3> &grid) {
    if (img.width % 32 || img.height % 32) fail("ViT input must be a multiple of 32 pixels");
    const int gh = img.height / kVitPatch, gw = img.width / kVitPatch, m = kVitMerge, P = kVitPatch;
    const int feat = 3 * kVitTemporal * P * P;  // 1536
    grid = {1, gh, gw};
    std::vector<float> out(size_t(gh) * gw * feat);
    size_t row = 0;
    for (int bh = 0; bh < gh / m; ++bh)
        for (int bw = 0; bw < gw / m; ++bw)
            for (int mh = 0; mh < m; ++mh)
                for (int mw = 0; mw < m; ++mw, ++row) {
                    float *o = &out[row * feat];
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < kVitTemporal; ++t)
                            for (int py = 0; py < P; ++py)
                                for (int px = 0; px < P; ++px) {
                                    int y = (bh * m + mh) * P + py, x = (bw * m + mw) * P + px;
                                    float v = float(img.rgb[(size_t(y) * img.width + x) * 3 + c]) / 255.0f;
                                    o[((c * kVitTemporal + t) * P + py) * P + px] = (v - 0.5f) / 0.5f;
                                }
                }
    return out;
}

Presentation make_presentation(const std::vector<std::vector<int32_t>> &labels, const std::vector<int32_t> &prompt,
                               const std::vector<std::array<int, 3>> &grids) {
    if (labels.size() != grids.size()) fail("one label per reference");
    Presentation p;
    for (size_t i = 0; i < labels.size(); ++i) {
        int nv = grids[i][0] * grids[i][1] * grids[i][2] / (kVitMerge * kVitMerge);
        for (auto id : labels[i]) p.ids.push_back(id), p.token_tags.push_back(1), p.mm_types.push_back(0);
        p.ids.push_back(kVisionStartId), p.token_tags.push_back(0), p.mm_types.push_back(0);
        for (int k = 0; k < nv; ++k) p.ids.push_back(kImageTokenId), p.token_tags.push_back(0), p.mm_types.push_back(1);
        p.ids.push_back(kVisionEndId), p.token_tags.push_back(0), p.mm_types.push_back(0);
    }
    for (auto id : prompt) p.ids.push_back(id), p.token_tags.push_back(1), p.mm_types.push_back(0);
    return p;
}

std::vector<std::array<int32_t, 3>> mrope_positions(const Presentation &p, const std::vector<std::array<int, 3>> &grids) {
    const size_t n = p.ids.size();
    std::vector<std::array<int32_t, 3>> pos(n);
    size_t gi = 0;
    int32_t current = 0;
    for (size_t i = 0; i < n;) {
        int typ = p.mm_types[i];
        size_t j = i + 1;
        while (j < n && p.mm_types[j] == typ) ++j;
        if (typ == 0) {
            for (size_t k = i; k < j; ++k) {
                int32_t v = current + int32_t(k - i);
                pos[k] = {v, v, v};
            }
            current += int32_t(j - i);
        } else {
            if (gi >= grids.size()) fail("more image runs than grids");
            int t = grids[gi][0], gh = grids[gi][1] / kVitMerge, gw = grids[gi][2] / kVitMerge;
            ++gi;
            if (j - i != size_t(t) * gh * gw) fail("image run length does not match its grid");
            for (int r = 0; r < t; ++r)
                for (int y = 0; y < gh; ++y)
                    for (int x = 0; x < gw; ++x) pos[i + size_t(r) * gh * gw + size_t(y) * gw + x] = {current + r, current + y, current + x};
            current += std::max(gh, gw) * t;
        }
        i = j;
    }
    return pos;
}

PackedLayout packed_layout(const std::vector<int32_t> &text_tags, const std::vector<std::array<int, 2>> &refs,
                           int latent_h, int latent_w) {
    PackedLayout L;
    L.text_rows = int(text_tags.size());
    for (auto &r : refs) L.reference_rows += (r[0] / 2) * (r[1] / 2);
    L.target_rows = (latent_h / 2) * (latent_w / 2);
    L.used = L.text_rows + L.reference_rows + L.target_rows;
    L.position_ids.assign(L.used, {0.0, 0.0, 0.0});
    L.token_tags.assign(L.used, 0);
    for (int i = 0; i < L.text_rows; ++i) L.position_ids[i][0] = double(i), L.token_tags[i] = text_tags[i];
    auto put_grid = [&](int start, int h, int w, double t) {
        double sqrt_area = std::sqrt(double(h) * w);
        auto axis = [&](int dim, int k) {
            double ratio = dim / sqrt_area, left = (1.0 - ratio) / 2.0;
            return (left + ratio * k / (dim / 2)) * 32.0;
        };
        for (int y = 0; y < h / 2; ++y)
            for (int x = 0; x < w / 2; ++x) L.position_ids[start + y * (w / 2) + x] = {t, axis(h, y), axis(w, x)};
    };
    int cursor = L.text_rows;
    double rotary = double(L.text_rows);
    for (auto &r : refs) {
        put_grid(cursor, r[0], r[1], rotary);
        cursor += (r[0] / 2) * (r[1] / 2);
        rotary += 1.0;
    }
    put_grid(cursor, latent_h, latent_w, rotary);
    return L;
}

void noise_reference(std::vector<float> &latent, const std::vector<float> &noise) {
    const float t = kReferenceNoiseT, omt = 1.0f - t;
    for (size_t i = 0; i < latent.size(); ++i) latent[i] = t * latent[i] + omt * noise[i];
}

}  // namespace edit
}  // namespace axd
