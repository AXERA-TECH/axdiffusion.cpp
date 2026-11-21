// Thin C++ wrapper over the AXCL runtime engine API.
//
// Design: every weight and activation lives in a device (CMM) buffer that is
// allocated once. A kernel launch only binds buffer addresses to the model's
// named inputs/outputs and executes; nothing is copied per launch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace axd {

// Process-wide runtime (axclInit / device / engine). Create exactly one.
class Runtime {
public:
    explicit Runtime(int device = 0);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    // Used / total CMM bytes.
    void cmm(size_t &used, size_t &total) const;
    // AXCL contexts are per thread: call this in any thread (e.g. HTTP workers) before NPU calls.
    void bind_thread() const;

private:
    int device_;
    void *ctx_ = nullptr;
};

// A device address range (non-owning view).
struct Buf {
    void *ptr = nullptr;
    size_t size = 0;
    Buf() = default;
    Buf(void *p, size_t n) : ptr(p), size(n) {}
    explicit operator bool() const { return ptr != nullptr; }
    Buf view(size_t offset, size_t n) const;
    void upload(const void *host, size_t n, size_t offset = 0) const;  // host -> device
    void download(void *host, size_t n, size_t offset = 0) const;      // device -> host
    void copy_from(const Buf &src, size_t n, size_t dst_offset = 0, size_t src_offset = 0) const;
    void fill_zero() const;
};

// An owned device allocation.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(size_t size, const std::string &name);
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer &&o) noexcept;
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    Buf buf() const { return Buf(ptr_, size_); }
    operator Buf() const { return buf(); }
    size_t size() const { return size_; }

private:
    void *ptr_ = nullptr;
    size_t size_ = 0;
};

// Many named tensors packed (256-byte aligned) into one allocation.
class TensorBank {
public:
    explicit TensorBank(std::string name) : name_(std::move(name)) {}
    // Reserve space; data is copied when materialize() runs. `src` must stay valid until then.
    void add(const std::string &name, const void *src, size_t size);
    void materialize();
    Buf get(const std::string &name) const;
    bool has(const std::string &name) const { return entries_.count(name) != 0; }
    size_t bytes() const { return total_; }

private:
    struct Pending {
        std::string name;
        size_t offset;
        const void *src;
        size_t size;
    };
    std::string name_;
    std::vector<Pending> pending_;
    std::map<std::string, Buf> entries_;
    size_t total_ = 0;
    DeviceBuffer storage_;
};

using Bindings = std::map<std::string, Buf>;

// One compiled AXModel with a reusable IO object.
class Model {
public:
    explicit Model(const std::string &path);
    ~Model();
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;

    // Bind every model input from `inputs` (extra names are ignored, missing
    // names are an error) and every output from `outputs` (missing outputs use
    // a private default buffer), then execute synchronously.
    void run(const Bindings &inputs, const Bindings &outputs = {});
    bool has_input(const std::string &name) const;
    std::string describe() const;
    const std::string &path() const { return path_; }

private:
    struct IO {
        std::string name;
        uint64_t size;
    };
    std::string path_;
    uint64_t model_id_ = 0, context_id_ = 0;
    void *io_info_ = nullptr;
    void *io_ = nullptr;
    std::vector<IO> inputs_, outputs_;
    std::map<std::string, DeviceBuffer> default_outputs_;
};

}  // namespace axd
