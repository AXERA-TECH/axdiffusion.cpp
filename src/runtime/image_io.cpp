#include "image_io.hpp"

#include <dlfcn.h>

#include <array>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "common.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace axd {

namespace {

// EXIF orientation (1..8) of a JPEG, or 1 when absent.
int jpeg_orientation(const std::string &b) {
    auto u8 = [&](size_t i) { return i < b.size() ? uint8_t(b[i]) : 0; };
    if (u8(0) != 0xFF || u8(1) != 0xD8) return 1;
    size_t p = 2;
    while (p + 4 < b.size() && u8(p) == 0xFF) {
        uint8_t marker = u8(p + 1);
        size_t len = (size_t(u8(p + 2)) << 8) | u8(p + 3);
        if (marker == 0xE1 && b.compare(p + 4, 6, std::string("Exif\0\0", 6)) == 0) {
            size_t t = p + 10;  // TIFF header
            bool le = u8(t) == 'I';
            auto r16 = [&](size_t i) { return le ? (u8(i) | (u8(i + 1) << 8)) : ((u8(i) << 8) | u8(i + 1)); };
            auto r32 = [&](size_t i) {
                return le ? (uint32_t(u8(i)) | (uint32_t(u8(i + 1)) << 8) | (uint32_t(u8(i + 2)) << 16) |
                             (uint32_t(u8(i + 3)) << 24))
                          : ((uint32_t(u8(i)) << 24) | (uint32_t(u8(i + 1)) << 16) | (uint32_t(u8(i + 2)) << 8) |
                             uint32_t(u8(i + 3)));
            };
            size_t ifd = t + r32(t + 4);
            int n = r16(ifd);
            for (int e = 0; e < n; ++e) {
                size_t ent = ifd + 2 + size_t(e) * 12;
                if (r16(ent) == 0x0112) {
                    int v = r16(ent + 8);
                    return v >= 1 && v <= 8 ? v : 1;
                }
            }
            return 1;
        }
        if (marker == 0xDA) break;  // start of scan
        p += 2 + len;
    }
    return 1;
}

Image orient(const Image &in, int o) {
    if (o == 1) return in;
    const bool swap = o >= 5;
    Image out;
    out.width = swap ? in.height : in.width;
    out.height = swap ? in.width : in.height;
    out.rgb.resize(in.rgb.size());
    for (int y = 0; y < out.height; ++y)
        for (int x = 0; x < out.width; ++x) {
            int sx = 0, sy = 0;
            switch (o) {
                case 2: sx = in.width - 1 - x, sy = y; break;
                case 3: sx = in.width - 1 - x, sy = in.height - 1 - y; break;
                case 4: sx = x, sy = in.height - 1 - y; break;
                case 5: sx = y, sy = x; break;
                case 6: sx = y, sy = in.height - 1 - x; break;
                case 7: sx = in.width - 1 - y, sy = in.height - 1 - x; break;
                case 8: sx = in.width - 1 - y, sy = x; break;
            }
            for (int c = 0; c < 3; ++c)
                out.rgb[(size_t(y) * out.width + x) * 3 + c] = in.rgb[(size_t(sy) * in.width + sx) * 3 + c];
        }
    return out;
}

}  // namespace

Image decode_image(const std::string &bytes) {
    int w = 0, h = 0, ch = 0;
    unsigned char *px = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(bytes.data()), int(bytes.size()),
                                              &w, &h, &ch, 3);
    if (!px) fail(std::string("cannot decode image: ") + stbi_failure_reason());
    Image img;
    img.width = w;
    img.height = h;
    img.rgb.assign(px, px + size_t(w) * h * 3);
    stbi_image_free(px);
    return orient(img, jpeg_orientation(bytes));
}

Image load_image(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open image " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return decode_image(ss.str());
}

namespace {

// PNG writer: Sub-filtered rows deflated by the system zlib (dlopen'd: compress2 at level 1 is ~10x faster than
// stb's encoder and smaller); stb is the fallback when libz is not present.
using compress2_fn = int (*)(unsigned char *, unsigned long *, const unsigned char *, unsigned long, int);
using bound_fn = unsigned long (*)(unsigned long);

struct Zlib {
    compress2_fn compress2 = nullptr;
    bound_fn bound = nullptr;
    Zlib() {
        for (const char *name : {"libz.so.1", "libz.so"})
            if (void *h = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
                compress2 = reinterpret_cast<compress2_fn>(dlsym(h, "compress2"));
                bound = reinterpret_cast<bound_fn>(dlsym(h, "compressBound"));
                if (compress2 && bound) return;
            }
        compress2 = nullptr;
    }
};
const Zlib &zlib() {
    static Zlib z;
    return z;
}

uint32_t png_crc(const unsigned char *p, size_t n, uint32_t c) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t v = i;
            for (int k = 0; k < 8; ++k) v = v & 1 ? 0xedb88320u ^ (v >> 1) : v >> 1;
            t[i] = v;
        }
        return t;
    }();
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c;
}

void put_be32(std::string &out, uint32_t v) {
    const char b[4] = {char(v >> 24), char(v >> 16), char(v >> 8), char(v)};
    out.append(b, 4);
}

void put_chunk(std::string &out, const char *type, const unsigned char *data, size_t n) {
    put_be32(out, uint32_t(n));
    const size_t start = out.size();
    out.append(type, 4);
    out.append(reinterpret_cast<const char *>(data), n);
    uint32_t c = png_crc(reinterpret_cast<const unsigned char *>(out.data() + start), n + 4, 0xffffffffu);
    put_be32(out, c ^ 0xffffffffu);
}

bool png_zlib(const std::vector<uint8_t> &rgb, int w, int h, std::string &out) {
    const auto &z = zlib();
    if (!z.compress2) return false;
    const size_t stride = size_t(w) * 3;
    std::vector<unsigned char> raw(size_t(h) * (1 + stride));
    for (int y = 0; y < h; ++y) {  // filter 1 (Sub)
        unsigned char *d = &raw[size_t(y) * (1 + stride)];
        const uint8_t *src = rgb.data() + size_t(y) * stride;
        d[0] = 1;
        for (size_t i = 0; i < 3; ++i) d[1 + i] = src[i];
        for (size_t i = 3; i < stride; ++i) d[1 + i] = uint8_t(src[i] - src[i - 3]);
    }
    unsigned long n = z.bound(raw.size());
    std::vector<unsigned char> idat(n);
    if (z.compress2(idat.data(), &n, raw.data(), raw.size(), 1) != 0) return false;
    out.clear();
    out.append("\x89PNG\r\n\x1a\n", 8);
    unsigned char ihdr[13] = {0};
    for (int i = 0; i < 4; ++i) {
        ihdr[i] = uint8_t(uint32_t(w) >> (24 - 8 * i));
        ihdr[4 + i] = uint8_t(uint32_t(h) >> (24 - 8 * i));
    }
    ihdr[8] = 8;  // bit depth
    ihdr[9] = 2;  // RGB
    put_chunk(out, "IHDR", ihdr, 13);
    put_chunk(out, "IDAT", idat.data(), n);
    put_chunk(out, "IEND", nullptr, 0);
    return true;
}

}  // namespace

bool write_png(const std::string &path, const std::vector<uint8_t> &rgb, int w, int h) {
    std::string png = encode_png(rgb, w, h);
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f.write(png.data(), std::streamsize(png.size()))) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

std::string encode_png(const std::vector<uint8_t> &rgb, int w, int h) {
    std::string out;
    if (png_zlib(rgb, w, h, out)) return out;
    stbi_write_png_compression_level = 1;
    auto sink = [](void *ctx, void *data, int size) {
        static_cast<std::string *>(ctx)->append(static_cast<const char *>(data), size_t(size));
    };
    if (!stbi_write_png_to_func(sink, &out, w, h, 3, rgb.data(), w * 3)) fail("PNG encoding failed");
    return out;
}

}  // namespace axd
