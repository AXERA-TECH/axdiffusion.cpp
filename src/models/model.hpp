// Model plugin interface. Each supported model architecture implements ModelPlugin; the runtime, job queue and
// APIs only talk to this interface.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "npu.hpp"
#include "task.hpp"

namespace axd {

struct ModelOptions {
    std::string lora_dir;  // empty: LoRA disabled
    int lora_cache = 4;    // resident LoRA combinations
    int device = 0;
};

struct ModelInfo {
    std::string arch;                       // "minimax-h3-image", ...
    std::string name;                       // id reported by the APIs
    std::vector<Task> tasks;                // what run() accepts
    std::vector<std::array<int, 2>> sizes;  // supported {width, height}
    int max_prompt_tokens = 0;
    int max_references = 0;
    bool lora = false;
    std::vector<int> refine_steps;          // accepted Request::refine_steps values other than 0 (empty: no refine)
    long refine_min_pixels = 0;             // smallest width*height a refine request may ask for
};

// How much of a prompt the model will read for a given request shape.
struct PromptBudget {
    int tokens = 0;      // tokens in the whole prompt
    int limit = 0;       // tokens the model accepts for this size / these references
    int cut_chars = -1;  // Unicode code points of the prompt that fit; -1 when everything fits
};

class ModelPlugin {
public:
    virtual ~ModelPlugin() = default;
    virtual ModelInfo info() const = 0;
    // Runs one request on the calling thread (which must own the NPU, see Runtime::bind_thread).
    virtual Result run(const Request &req, Progress &progress) = 0;
    // Thread-safe, does not touch the NPU. reference_sizes are {width, height} of the edit references (empty: txt2img).
    virtual PromptBudget prompt_budget(const std::string &prompt, int width, int height,
                                       const std::vector<std::array<int, 2>> &reference_sizes) const = 0;
    virtual std::vector<std::string> loras() const { return {}; }
    virtual double load_seconds() const = 0;
    virtual Runtime &runtime() = 0;
};

// Opens a model directory and instantiates the plugin for its architecture (axdiffusion.json "arch").
std::unique_ptr<ModelPlugin> create_model(const std::string &dir, const ModelOptions &opt);
std::vector<std::string> supported_architectures();

}  // namespace axd
