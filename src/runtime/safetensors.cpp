#include "safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

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

}  // namespace axd
