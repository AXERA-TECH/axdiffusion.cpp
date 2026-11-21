// Image decoding (PNG/JPEG/BMP/... via stb_image) with EXIF orientation, and PNG encoding.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace axd {

struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;  // HWC, 8-bit RGB
};

// Decode an encoded image (any format stb_image supports). JPEG EXIF orientation is applied,
// alpha is dropped, grayscale is expanded to RGB.
Image decode_image(const std::string &bytes);
Image load_image(const std::string &path);

}  // namespace axd
