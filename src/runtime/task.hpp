// Model-independent request / result types shared by every model plugin, the job queue and all APIs.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_io.hpp"

namespace axd {

enum class Task { txt2img, edit, txt2video, img2video, ref2video };
const char *task_name(Task t);
bool parse_task(const std::string &s, Task &t);

struct LoraSpec {
    std::string name;  // file stem inside the LoRA directory (or a path for local tools)
    float scale = 1.0f;
};

struct Video {
    int width = 0, height = 0;
    float fps = 0;
    std::vector<Image> frames;
};

struct Audio {
    int sample_rate = 0, channels = 0;
    std::vector<float> samples;  // interleaved
};

struct Request {
    Task task = Task::txt2img;
    std::string prompt, negative_prompt;
    int width = 1024, height = 1024;
    int frames = 1;      // video only
    float fps = 0;       // video only; 0 = model default
    uint64_t seed = 42;
    int steps = 0;       // 0 = model default
    float cfg = 0;       // 0 = model default
    std::vector<LoraSpec> loras;
    std::vector<Image> images;  // reference images / keyframes
    std::vector<Video> videos;  // reference videos
    std::vector<Audio> audios;  // reference audio
    // Two-pass large image: generate at half the size, upscale, re-noise and run only the last N denoise steps
    // at full size. 0 = off; models list what they accept in ModelInfo::refine_steps.
    int refine_steps = 0;
    // debugging
    int blocks = 0;            // 0 = all transformer blocks
    std::string dump_dir;      // write intermediate tensors here
};

struct Result {
    std::vector<Image> images;
    Video video;
    Audio audio;
    std::map<std::string, double> seconds;  // stage timings
    std::map<std::string, double> metrics;  // tokens, sequence length, ...
    std::vector<int32_t> token_ids;
    bool truncated = false;
};

// Progress reporting and cancellation, shared between the worker running a request and its observers.
struct Progress {
    std::function<void(const std::string &stage, int step, int total)> report;  // may be empty
    std::atomic<bool> cancel{false};
    void operator()(const std::string &stage, int step = 0, int total = 0) const {
        if (report) report(stage, step, total);
    }
};

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled") {}
};

}  // namespace axd
