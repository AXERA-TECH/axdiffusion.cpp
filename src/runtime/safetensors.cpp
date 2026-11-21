#include "safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

#include "common.hpp"
#include "json.hpp"

namespace axd {

SafeTensors::SafeTensors(const std::string &path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) fail("cannot open " + path);
    struct stat st {};
    if (fstat(fd_, &st) != 0) fail("cannot stat " + path);
    map_size_ = size_t(st.st_size);
    map_ = mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        fail("cannot mmap " + path);
    }
    // Weights are read once, front to back.
    madvise(map_, map_size_, MADV_SEQUENTIAL);
    auto *base = static_cast<const uint8_t *>(map_);
    base_ = base;
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    if (8 + header_len > map_size_) fail("bad safetensors header in " + path);
    auto header = nlohmann::json::parse(base + 8, base + 8 + header_len);
    const uint8_t *data = base + 8 + header_len;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        TensorView v;
        v.dtype = it.value()["dtype"].get<std::string>();
        v.shape = it.value()["shape"].get<std::vector<int64_t>>();
        auto off = it.value()["data_offsets"].get<std::vector<uint64_t>>();
        v.data = data + off[0];
        v.nbytes = size_t(off[1] - off[0]);
        if (v.data + v.nbytes > base + map_size_) fail("tensor " + it.key() + " out of file bounds in " + path);
        tensors_.emplace(it.key(), std::move(v));
    }
}

SafeTensors::~SafeTensors() {
    if (map_) munmap(map_, map_size_);
    if (fd_ >= 0) ::close(fd_);
}

const TensorView &SafeTensors::get(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("tensor '" + name + "' missing in " + path_);
    return it->second;
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    for (auto &kv : tensors_) out.push_back(kv.first);
    return out;
}

void SafeTensors::read_all(const std::vector<std::string> &prefixes, unsigned threads) {
    if (owned_) return;
    auto wanted = [&](const std::string &name) {
        if (prefixes.empty()) return true;
        for (auto &p : prefixes)
            if (name.compare(0, p.size(), p) == 0) return true;
        return false;
    };
    // byte ranges of the wanted tensors, merged, cut into 16 MB reads
    std::vector<std::pair<size_t, size_t>> ranges;
    for (auto &kv : tensors_)
        if (wanted(kv.first)) {
            size_t o = size_t(kv.second.data - base_);
            ranges.push_back({o, o + kv.second.nbytes});
        }
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<size_t, size_t>> pieces;
    constexpr size_t kPiece = 16u << 20;
    for (size_t i = 0; i < ranges.size();) {
        size_t a = ranges[i].first, b = ranges[i].second;
        for (++i; i < ranges.size() && ranges[i].first <= b; ++i) b = std::max(b, ranges[i].second);
        for (size_t o = a; o < b; o += kPiece) pieces.push_back({o, std::min(b, o + kPiece)});
    }
    owned_.reset(new uint8_t[map_size_]);  // uninitialized: only the read ranges are ever touched
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    auto worker = [&] {
        for (size_t i; (i = next++) < pieces.size();) {
            for (size_t o = pieces[i].first; o < pieces[i].second;) {
                ssize_t r = ::pread(fd_, owned_.get() + o, pieces[i].second - o, off_t(o));
                if (r <= 0) {
                    bad = true;
                    return;
                }
                o += size_t(r);
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::max(1u, threads); ++t) pool.emplace_back(worker);
    for (auto &t : pool) t.join();
    if (bad) fail("read failed: " + path_);
    for (auto &kv : tensors_)
        if (wanted(kv.first)) kv.second.data = owned_.get() + (kv.second.data - base_);
}

std::vector<float> to_f32(const TensorView &t) {
    std::vector<float> out(size_t(t.numel()));
    if (t.dtype == "F32") {
        std::memcpy(out.data(), t.data, out.size() * 4);
    } else if (t.dtype == "BF16") {
        auto *p = reinterpret_cast<const uint16_t *>(t.data);
        parallel_for(out.size(), [&](size_t a, size_t b) {
            for (size_t i = a; i < b; ++i) out[i] = bf16_to_f32(p[i]);
        }, out.size() > (1u << 16) ? 0u : 1u);
    } else if (t.dtype == "F16") {
        auto *p = reinterpret_cast<const uint16_t *>(t.data);
        for (size_t i = 0; i < out.size(); ++i) {
            uint16_t h = p[i];
            uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
            if (exp == 0) {
                if (man == 0) {
                    bits = sign;
                } else {  // subnormal
                    exp = 127 - 15 + 1;
                    while (!(man & 0x400)) man <<= 1, --exp;
                    bits = sign | (exp << 23) | ((man & 0x3FF) << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7F800000u | (man << 13);
            } else {
                bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
            }
            std::memcpy(&out[i], &bits, 4);
        }
    } else {
        fail("LoRA tensor has unsupported dtype " + t.dtype);
    }
    return out;
}

std::vector<uint16_t> to_bf16(const std::vector<float> &v) {
    std::vector<uint16_t> out(v.size());
    parallel_for(v.size(), [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) out[i] = f32_to_bf16(v[i]);
    }, v.size() > (1u << 16) ? 0u : 1u);
    return out;
}

std::vector<uint16_t> to_bf16(const TensorView &t) {
    const size_t n = size_t(t.numel());
    if (t.dtype == "BF16") {
        std::vector<uint16_t> out(n);
        std::memcpy(out.data(), t.data, n * 2);
        return out;
    }
    if (t.dtype == "F32") {
        std::vector<uint16_t> out(n);
        const float *p = reinterpret_cast<const float *>(t.data);
        parallel_for(n, [&](size_t a, size_t b) {
            for (size_t i = a; i < b; ++i) out[i] = f32_to_bf16(p[i]);
        }, n > (1u << 16) ? 0u : 1u);
        return out;
    }
    return to_bf16(to_f32(t));
}

}  // namespace axd
