// Read-only, memory-mapped safetensors file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace axd {

struct TensorView {
    std::string dtype;           // "BF16", "F32", "U8", "I32", ...
    std::vector<int64_t> shape;
    const uint8_t *data = nullptr;
    size_t nbytes = 0;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string &path);
    ~SafeTensors();
    SafeTensors(const SafeTensors &) = delete;
    SafeTensors &operator=(const SafeTensors &) = delete;

    const TensorView &get(const std::string &name) const;
    bool has(const std::string &name) const { return tensors_.count(name) != 0; }
    std::vector<std::string> names() const;  // header order is not preserved; sorted
    const std::string &path() const { return path_; }
    int fd() const { return fd_; }
    uint64_t file_offset(const TensorView &t) const { return uint64_t(t.data - base_); }
    // Read the tensors whose name starts with one of `prefixes` (all when empty) into memory with `threads`
    // parallel preads and point their views at the copy. For tensors that are transformed on the host this is
    // much faster than faulting the mapping in page by page; other tensors keep using the mapping.
    void read_all(const std::vector<std::string> &prefixes = {}, unsigned threads = 6);

private:
    std::string path_;
    int fd_ = -1;
    void *map_ = nullptr;
    size_t map_size_ = 0;
    const uint8_t *base_ = nullptr;
    std::unique_ptr<uint8_t[]> owned_;
    std::map<std::string, TensorView> tensors_;
};

// Convert a float tensor (F32 / BF16 / F16) to float32.
std::vector<float> to_f32(const TensorView &t);
std::vector<uint16_t> to_bf16(const std::vector<float> &v);
std::vector<uint16_t> to_bf16(const TensorView &t);  // F32 / BF16 / F16 tensor -> bf16 words (parallel)

}  // namespace axd
