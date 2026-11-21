#include "image_io.hpp"

#include <fstream>
#include <sstream>

#include "common.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

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

}  // namespace axd
