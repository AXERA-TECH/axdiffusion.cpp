#include "npu.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

#include "axcl.h"
#include "common.hpp"

namespace axd {

namespace {

void check(axclError rc, const std::string &what) {
    if (rc != 0) fail(what + " failed: axclError=" + std::to_string(int(rc)));
}

// Large single DMA transfers can fail with ENOMEM on a busy board; copy in chunks.
constexpr size_t kCopyChunk = 16u << 20;

void memcpy_chunked(void *dst, const void *src, size_t n, axclrtMemcpyKind kind) {
    auto *d = static_cast<uint8_t *>(dst);
    auto *s = static_cast<const uint8_t *>(src);
    for (size_t done = 0; done < n;) {
        size_t step = std::min(kCopyChunk, n - done);
        check(axclrtMemcpy(d + done, s + done, step, kind), "axclrtMemcpy");
        done += step;
    }
}

}  // namespace

// ---------------------------------------------------------------- Runtime
namespace {
void *g_runtime_ctx = nullptr;  // the process-wide runtime's context, for helper threads
}

Runtime::Runtime(int device) : device_(device) {
    check(axclInit(""), "axclInit");
    uint32_t count = 0;
    check(axclrtGetDeviceCount(&count), "axclrtGetDeviceCount");
    if (count == 0 || device < 0 || uint32_t(device) >= count) fail("NPU device index out of range");
    check(axclrtSetDevice(device), "axclrtSetDevice");
    check(axclrtEngineInit(AXCL_VNPU_DISABLE), "axclrtEngineInit");
    axclrtContext ctx = nullptr;
    check(axclrtGetCurrentContext(&ctx), "axclrtGetCurrentContext");
    ctx_ = ctx;
    g_runtime_ctx = ctx;
}

void Runtime::bind_current_thread() {
    thread_local const void *bound = nullptr;
    if (!g_runtime_ctx || bound == g_runtime_ctx) return;
    check(axclrtSetCurrentContext(axclrtContext(g_runtime_ctx)), "axclrtSetCurrentContext");
    bound = g_runtime_ctx;
}

void Runtime::bind_thread() const {
    // Bind once per thread (querying an unbound thread makes the runtime log an error).
    thread_local const void *bound = nullptr;
    if (bound == ctx_) return;
    check(axclrtSetCurrentContext(axclrtContext(ctx_)), "axclrtSetCurrentContext");
    bound = ctx_;
}

Runtime::~Runtime() {
    axclrtSynchronizeDevice();
    axclrtEngineFinalize();
    axclrtResetDevice(device_);
    axclFinalize();
}

void Runtime::cmm(size_t &used, size_t &total) const {
    bind_thread();
    size_t free_b = 0;
    total = 0;
    check(axclrtGetMemInfo(axclrtMemAttr(0), &free_b, &total), "axclrtGetMemInfo");
    used = total - free_b;
}

// ---------------------------------------------------------------- Buf
Buf Buf::view(size_t offset, size_t n) const {
    if (offset + n > size) fail("device buffer view out of range");
    return Buf(static_cast<uint8_t *>(ptr) + offset, n);
}

void Buf::upload(const void *host, size_t n, size_t offset) const {
    if (offset + n > size) fail("upload larger than device buffer");
    memcpy_chunked(static_cast<uint8_t *>(ptr) + offset, host, n, AXCL_MEMCPY_HOST_TO_DEVICE);
}

void Buf::download(void *host, size_t n, size_t offset) const {
    if (offset + n > size) fail("download larger than device buffer");
    memcpy_chunked(host, static_cast<uint8_t *>(ptr) + offset, n, AXCL_MEMCPY_DEVICE_TO_HOST);
}

void Buf::copy_from(const Buf &src, size_t n, size_t dst_offset, size_t src_offset) const {
    if (dst_offset + n > size || src_offset + n > src.size) fail("device copy out of range");
    memcpy_chunked(static_cast<uint8_t *>(ptr) + dst_offset, static_cast<uint8_t *>(src.ptr) + src_offset, n,
                   AXCL_MEMCPY_DEVICE_TO_DEVICE);
}

void Buf::fill_zero() const {
    static const std::vector<uint8_t> zeros(kCopyChunk, 0);
    for (size_t done = 0; done < size;) {
        size_t step = std::min(kCopyChunk, size - done);
        check(axclrtMemcpy(static_cast<uint8_t *>(ptr) + done, zeros.data(), step, AXCL_MEMCPY_HOST_TO_DEVICE),
              "axclrtMemcpy(zero)");
        done += step;
    }
}

// ---------------------------------------------------------------- DeviceBuffer
DeviceBuffer::DeviceBuffer(size_t size, const std::string &name) : size_(size) {
    check(axclrtMalloc(&ptr_, std::max<size_t>(size, 1), AXCL_MEM_MALLOC_HUGE_FIRST),
          "axclrtMalloc(" + name + ", " + std::to_string(size) + " B)");
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_) axclrtFree(ptr_);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&o) noexcept {
    if (this != &o) {
        if (ptr_) axclrtFree(ptr_);
        ptr_ = o.ptr_;
        size_ = o.size_;
        o.ptr_ = nullptr;
        o.size_ = 0;
    }
    return *this;
}

// ---------------------------------------------------------------- TensorBank
void TensorBank::add(const std::string &name, const void *src, size_t size) {
    if (storage_.size()) fail("tensor bank " + name_ + " already materialized");
    size_t offset = (total_ + 255) / 256 * 256;
    pending_.push_back({name, offset, src, size});
    total_ = offset + size;
}

void TensorBank::add_file(const std::string &name, int fd, uint64_t file_offset, size_t size) {
    if (storage_.size()) fail("tensor bank " + name_ + " already materialized");
    size_t offset = (total_ + 255) / 256 * 256;
    pending_.push_back({name, offset, nullptr, size, fd, file_offset});
    total_ = offset + size;
}

void TensorBank::allocate() {
    if (storage_.size()) return;
    storage_ = DeviceBuffer(total_, name_);
    Buf all = storage_.buf();
    for (auto &p : pending_) entries_[p.name] = all.view(p.offset, p.size);
}

void TensorBank::fill() {
    if (!pending_.empty()) materialize_parallel({this}, 1);
}

void TensorBank::materialize() {
    if (storage_.size() && pending_.empty()) return;
    materialize_parallel({this}, 1);
}

void materialize_parallel(const std::vector<TensorBank *> &banks, unsigned threads) {
    struct Job {
        Buf dst;
        const uint8_t *src;  // memory source, or
        int fd;              // file source
        uint64_t file_offset;
        size_t n;
    };
    constexpr size_t kJob = 16u << 20;  // one pread + one H2D copy
    std::vector<Job> jobs;
    for (auto *b : banks) {
        b->allocate();
        Buf all = b->storage_.buf();
        for (auto &p : b->pending_) {
            if (!p.src && p.fd < 0) continue;
            for (size_t o = 0; o < p.size; o += kJob) {
                size_t n = std::min(kJob, p.size - o);
                jobs.push_back({all.view(p.offset + o, n), p.src ? static_cast<const uint8_t *>(p.src) + o : nullptr,
                                p.fd, p.file_offset + o, n});
            }
        }
    }
    std::atomic<size_t> next{0};
    std::string error;
    std::mutex error_mu;
    auto worker = [&] {
        try {
            Runtime::bind_current_thread();
            std::vector<uint8_t> buf;
            for (size_t i; (i = next++) < jobs.size();) {
                const Job &j = jobs[i];
                if (j.src) {
                    j.dst.upload(j.src, j.n);
                    continue;
                }
                buf.resize(kJob);
                for (size_t done = 0; done < j.n;) {
                    ssize_t r = ::pread(j.fd, buf.data() + done, j.n - done, off_t(j.file_offset + done));
                    if (r <= 0) fail("weight file read failed");
                    done += size_t(r);
                }
                j.dst.upload(buf.data(), j.n);
            }
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lk(error_mu);
            if (error.empty()) error = e.what();
            next = jobs.size();
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::max(1u, threads); ++t) pool.emplace_back(worker);
    for (auto &t : pool) t.join();
    if (!error.empty()) fail(error);
    for (auto *b : banks) b->pending_.clear();
}

Buf TensorBank::get(const std::string &name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) fail("tensor '" + name + "' not found in bank " + name_);
    return it->second;
}

// ---------------------------------------------------------------- Model
Model::Model(const std::string &path) : path_(path) {
    // LoadFromFile sends the whole file in one DMA, which the driver rejects for large models (hundreds of MB);
    // stage large files into device memory in chunks and load from there.
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) fail("model file not found: " + path);
    const size_t bytes = size_t(st.st_size);
    if (bytes <= (64u << 20)) {
        check(axclrtEngineLoadFromFile(path.c_str(), &model_id_), "load " + path);
    } else {
        // read with several workers (one stream is ~0.7 GB/s on the board's NVMe, four ~1.9 GB/s), then upload
        std::unique_ptr<uint8_t[]> host(new uint8_t[bytes]);
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) fail("cannot open " + path);
        std::atomic<size_t> next{0};
        std::atomic<bool> bad{false};
        auto worker = [&] {
            for (size_t o; (o = next.fetch_add(kCopyChunk)) < bytes;) {
                const size_t end = std::min(bytes, o + kCopyChunk);
                for (size_t p = o; p < end;) {
                    ssize_t r = ::pread(fd, host.get() + p, end - p, off_t(p));
                    if (r <= 0) {
                        bad = true;
                        return;
                    }
                    p += size_t(r);
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < 4; ++t) pool.emplace_back(worker);
        for (auto &t : pool) t.join();
        ::close(fd);
        if (bad) fail("short read: " + path);
        DeviceBuffer staging(bytes, "model-staging");
        staging.buf().upload(host.get(), bytes);
        check(axclrtEngineLoadFromMem(staging.buf().ptr, bytes, &model_id_), "load (from device memory) " + path);
    }
    check(axclrtEngineCreateContext(model_id_, &context_id_), "CreateContext " + path);
    axclrtEngineIOInfo info = nullptr;
    check(axclrtEngineGetIOInfo(model_id_, &info), "GetIOInfo " + path);
    io_info_ = info;
    uint32_t ni = axclrtEngineGetNumInputs(info), no = axclrtEngineGetNumOutputs(info);
    for (uint32_t i = 0; i < ni; ++i)
        inputs_.push_back({axclrtEngineGetInputNameByIndex(info, i), axclrtEngineGetInputSizeByIndex(info, 0, i)});
    for (uint32_t i = 0; i < no; ++i)
        outputs_.push_back({axclrtEngineGetOutputNameByIndex(info, i), axclrtEngineGetOutputSizeByIndex(info, 0, i)});
    axclrtEngineIO io = nullptr;
    check(axclrtEngineCreateIO(info, &io), "CreateIO " + path);
    io_ = io;
    for (auto &o : outputs_) default_outputs_.emplace(o.name, DeviceBuffer(o.size, o.name));
}

Model::~Model() {
    default_outputs_.clear();
    if (io_) axclrtEngineDestroyIO(io_);
    if (io_info_) axclrtEngineDestroyIOInfo(io_info_);
    if (model_id_) axclrtEngineUnload(model_id_);
}

bool Model::has_input(const std::string &name) const {
    for (auto &i : inputs_)
        if (i.name == name) return true;
    return false;
}

std::vector<std::string> Model::input_names() const {
    std::vector<std::string> v;
    for (auto &i : inputs_) v.push_back(i.name);
    return v;
}

std::vector<std::string> Model::output_names() const {
    std::vector<std::string> v;
    for (auto &o : outputs_) v.push_back(o.name);
    return v;
}

std::string Model::describe() const {
    std::string s = path_ + "\n  inputs:";
    for (auto &i : inputs_) s += " " + i.name + "(" + std::to_string(i.size) + ")";
    s += "\n  outputs:";
    for (auto &o : outputs_) s += " " + o.name + "(" + std::to_string(o.size) + ")";
    return s;
}

void Model::bind(const Bindings &inputs, const Bindings &outputs) {
    for (auto &in : inputs_) {
        auto it = inputs.find(in.name);
        if (it == inputs.end() || !it->second) fail(path_ + ": input '" + in.name + "' is not bound");
        if (it->second.size < in.size)
            fail(path_ + ": input '" + in.name + "' needs " + std::to_string(in.size) + " B, got " +
                 std::to_string(it->second.size));
        check(axclrtEngineSetInputBufferByName(io_, in.name.c_str(), it->second.ptr, in.size ? in.size : 1),
              "bind input " + in.name);
    }
    for (auto &out : outputs_) {
        auto it = outputs.find(out.name);
        Buf b = it != outputs.end() ? it->second : default_outputs_.at(out.name).buf();
        if (b.size < out.size) fail(path_ + ": output '" + out.name + "' buffer too small");
        check(axclrtEngineSetOutputBufferByName(io_, out.name.c_str(), b.ptr, out.size), "bind output " + out.name);
    }
}

void Model::run(const Bindings &inputs, const Bindings &outputs) {
    bind(inputs, outputs);
    check(axclrtEngineExecute(model_id_, context_id_, 0, io_), "Execute " + path_);
}

void Model::submit(Stream &stream, const Bindings &inputs, const Bindings &outputs) {
    bind(inputs, outputs);
    check(axclrtEngineExecuteAsync(model_id_, context_id_, 0, io_, axclrtStream(stream.handle())),
          "ExecuteAsync " + path_);
}

// ---------------------------------------------------------------- Stream
Stream::Stream() {
    axclrtStream s = nullptr;
    check(axclrtCreateStream(&s), "axclrtCreateStream");
    s_ = s;
}

Stream::~Stream() {
    if (s_) {
        axclrtSynchronizeStream(axclrtStream(s_));
        axclrtDestroyStream(axclrtStream(s_));
    }
}

void Stream::sync() { check(axclrtSynchronizeStream(axclrtStream(s_)), "axclrtSynchronizeStream"); }

}  // namespace axd
