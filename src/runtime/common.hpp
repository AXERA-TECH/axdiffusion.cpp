// Small shared helpers: errors, timing, logging, BF16 conversion, a parallel-for.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace axd {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const std::string &msg) { throw Error(msg); }

inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

extern bool g_verbose;

#define AXD_LOG(...)                                         \
    do {                                                    \
        if (::axd::g_verbose) {                              \
            std::fprintf(stderr, "[axdiffusion] " __VA_ARGS__);   \
            std::fputc('\n', stderr);                       \
        }                                                   \
    } while (0)

// BF16 <-> FP32 (round to nearest even, same as the reference implementation).
inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

// Run fn(begin, end) over [0, n) on up to `threads` workers.
inline void parallel_for(size_t n, const std::function<void(size_t, size_t)> &fn, unsigned threads = 0) {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    if (n < 2 || threads == 1) {
        fn(0, n);
        return;
    }
    threads = unsigned(std::min<size_t>(threads, n));
    std::vector<std::thread> pool;
    size_t chunk = (n + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        size_t b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        pool.emplace_back(fn, b, e);
    }
    for (auto &th : pool) th.join();
}

}  // namespace axd
