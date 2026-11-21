// Random numbers compatible with PyTorch's CPU generator, so a seed reproduces the reference pipeline.
#pragma once

#include <cstdint>
#include <vector>

namespace axd {

// torch.Generator (CPU) equivalent: MT19937 + torch's 16-wide Box-Muller. Successive randn() calls continue
// the same stream, like successive torch.randn(..., generator=g) calls.
class TorchGenerator {
public:
    explicit TorchGenerator(uint64_t seed);
    std::vector<float> randn(size_t count);

private:
    uint32_t next();
    float uniform();
    double uniform_double();
    uint32_t s_[624];
    int idx_ = 624;
};

std::vector<float> torch_randn(uint64_t seed, size_t count);  // one-shot convenience

}  // namespace axd
