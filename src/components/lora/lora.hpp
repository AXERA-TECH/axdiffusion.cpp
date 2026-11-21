// Dynamic LoRA for the DiT: the NVFP4 base stays resident; LoRA A/B (bf16) are runtime
// inputs of the block kernel's low-rank branch  y = W x + B (A x).
//
// * Formats: diffusers/PEFT (lora_A / lora_B) and ComfyUI/kohya (lora_down / lora_up / alpha).
// * Several LoRAs are concatenated along the rank axis; strength * alpha / rank is folded into B.
//   The kernel is compiled for a fixed max rank (64); unused rank rows are zero.
// * DiT blocks: A/B uploaded once per LoRA combination and cached (LRU).
// * Token refiner (2 layers, bf16 weights): merged on the host into a cached weight copy.
// * Unknown or unsupported keys are an error, never silently ignored.
#pragma once

#include <array>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "npu.hpp"
#include "task.hpp"

namespace axd {

constexpr int kLoraRank = 64;

// One resident LoRA combination.
struct LoraSet {
    std::string key;
    DeviceBuffer blocks;                        // per block: A/B for qkv,out,fc1,fc2
    std::vector<std::array<Buf, 8>> block_ab;   // [50][a_qkv,b_qkv,a_out,b_out,a_fc1,b_fc1,a_fc2,b_fc2]
    std::vector<bool> block_active;             // blocks without any LoRA use the zero set
    std::unique_ptr<TensorBank> refiner;        // merged refiner weights, or null if untouched
    size_t bytes = 0;
};

class LoraManager {
public:
    LoraManager(std::string lora_dir, const std::string &shared_safetensors, int cache_size);
    ~LoraManager();
    // Returns the resident set for this combination (building and caching it if needed).
    const LoraSet &get(const std::vector<LoraSpec> &specs);
    const std::array<Buf, 8> &zero_ab() const { return zero_ab_; }
    std::vector<std::string> available() const;

private:
    std::string resolve(const std::string &name) const;
    std::unique_ptr<LoraSet> build(const std::vector<LoraSpec> &specs, const std::string &key);
    std::string dir_;
    std::string shared_path_;
    int cache_size_;
    std::list<std::unique_ptr<LoraSet>> cache_;  // front = most recently used
    DeviceBuffer zero_;
    std::array<Buf, 8> zero_ab_;
};

// Kernel input names of the LoRA branch, in LoraSet::block_ab order.
extern const char *const kLoraInputs[8];

}  // namespace axd
