#include "random.hpp"

#include <cmath>

namespace axd {

// ------------------------------------------------------------------ noise
// Reproduces torch.randn((1,24,1,H,W)) on CPU (MT19937 + 16-wide Box-Muller), so a seed
// gives the same image as the reference pipeline.
namespace {
void normal_fill16(float *d) {
    for (int j = 0; j < 8; ++j) {
        float u1 = 1.0f - d[j], u2 = d[j + 8];
        float radius = sqrtf(-2.0f * logf(u1));
        float theta = float(2.0 * M_PI * double(u2));
        d[j] = radius * cosf(theta);
        d[j + 8] = radius * sinf(theta);
    }
}
}  // namespace

TorchGenerator::TorchGenerator(uint64_t seed) {
    s_[0] = uint32_t(seed);
    for (int i = 1; i < 624; ++i) s_[i] = 1812433253u * (s_[i - 1] ^ (s_[i - 1] >> 30)) + uint32_t(i);
}

uint32_t TorchGenerator::next() {
    if (idx_ >= 624) {
        for (int i = 0; i < 624; ++i) {
            uint32_t y = (s_[i] & 0x80000000u) | (s_[(i + 1) % 624] & 0x7FFFFFFFu);
            s_[i] = s_[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908B0DFu : 0u);
        }
        idx_ = 0;
    }
    uint32_t y = s_[idx_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9D2C5680u;
    y ^= (y << 15) & 0xEFC60000u;
    y ^= y >> 18;
    return y;
}

float TorchGenerator::uniform() { return float(next() & 0xFFFFFFu) * (1.0f / float(1u << 24)); }

double TorchGenerator::uniform_double() {
    uint64_t hi = next(), lo = next();
    return double(((hi << 32) | lo) & ((uint64_t(1) << 53) - 1)) / double(uint64_t(1) << 53);
}

std::vector<float> TorchGenerator::randn(size_t count) {
    std::vector<float> out(count);
    if (count < 16) {  // torch's scalar path: cached Box-Muller pairs in double
        for (size_t i = 0; i < count; i += 2) {
            double u1 = uniform_double(), u2 = uniform_double();
            double r = std::sqrt(-2.0 * std::log1p(-u2)), th = 2.0 * M_PI * u1;
            out[i] = float(r * std::cos(th));
            if (i + 1 < count) out[i + 1] = float(r * std::sin(th));
        }
        return out;
    }
    for (auto &v : out) v = uniform();
    for (size_t start = 0; start + 16 <= count; start += 16) normal_fill16(&out[start]);
    if (count % 16) {
        size_t start = count - 16;
        for (size_t i = start; i < count; ++i) out[i] = uniform();
        normal_fill16(&out[start]);
    }
    return out;
}

std::vector<float> torch_randn(uint64_t seed, size_t count) { return TorchGenerator(seed).randn(count); }

}  // namespace axd
