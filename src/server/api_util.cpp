#include <ctime>
#include <fstream>
#include <random>
#include <regex>

#include "api.hpp"
#include "common.hpp"

namespace axd {

std::string base64_encode(const std::string &in) {
    static const char *t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t v = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8) | uint8_t(in[i + 2]);
        out += t[v >> 18], out += t[(v >> 12) & 63], out += t[(v >> 6) & 63], out += t[v & 63];
    }
    if (i < in.size()) {
        uint32_t v = uint8_t(in[i]) << 16;
        if (i + 1 < in.size()) v |= uint8_t(in[i + 1]) << 8;
        out += t[v >> 18], out += t[(v >> 12) & 63];
        out += i + 1 < in.size() ? t[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::string base64_decode(const std::string &in_raw) {
    std::string in = in_raw;
    auto comma = in.find(',');
    if (in.rfind("data:", 0) == 0 && comma != std::string::npos) in = in.substr(comma + 1);
    static int T[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) T[i] = -1;
        const char *a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[uint8_t(a[i])] = i;
        T[uint8_t('-')] = 62, T[uint8_t('_')] = 63;
        init = true;
    }
    std::string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (T[c] < 0) continue;
        val = (val << 6) + T[c];
        bits += 6;
        if (bits >= 0) {
            out.push_back(char((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

void json_error(httplib::Response &res, int status, const std::string &msg) {
    res.status = status;
    res.set_content(nlohmann::json{{"error", {{"message", msg}, {"type", "invalid_request_error"}}}}.dump(),
                    "application/json");
}

uint64_t resolve_seed(int64_t seed) {
    if (seed >= 0) return uint64_t(seed);
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    return rng() & 0xFFFFFFFFull;
}

void parse_size(const std::string &s, int &w, int &h) {
    auto x = s.find('x');
    if (x == std::string::npos) fail("size must look like 1024x1024");
    w = std::stoi(s.substr(0, x));
    h = std::stoi(s.substr(x + 1));
}

std::vector<LoraSpec> loras_from_json(const nlohmann::json &j) {
    std::vector<LoraSpec> out;
    if (j.is_null()) return out;
    for (auto &l : j) {
        LoraSpec s;
        s.name = l.value("name", l.value("path", std::string()));
        s.scale = float(l.value("scale", l.value("weight", 1.0)));
        if (s.name.empty() || s.name.find('/') != std::string::npos || s.name.find("..") != std::string::npos)
            fail("LoRA entries need a 'name' from GET /v1/loras");
        out.push_back(s);
    }
    return out;
}

std::string strip_lora_tags(const std::string &prompt, std::vector<LoraSpec> &loras) {
    static const std::regex re(R"(<lora:([^:>]+)(?::([0-9.+-]+))?>)");
    std::string out;
    size_t last = 0;
    for (auto it = std::sregex_iterator(prompt.begin(), prompt.end(), re); it != std::sregex_iterator(); ++it) {
        LoraSpec s;
        s.name = (*it)[1].str();
        s.scale = (*it)[2].matched ? std::stof((*it)[2].str()) : 1.0f;
        if (s.name.find('/') != std::string::npos || s.name.find("..") != std::string::npos) fail("bad LoRA name");
        loras.push_back(s);
        out += prompt.substr(last, size_t(it->position()) - last);
        last = size_t(it->position() + it->length());
    }
    return out + prompt.substr(last);
}

Result run_sync(ServerContext &ctx, Request req) {
    auto job = ctx.queue.submit(std::move(req));
    ctx.queue.wait(job);
    Result r;
    std::string err;
    JobState st = JobState::queued;
    ctx.queue.read(job, [&](Job &j) {
        st = j.state;
        err = j.error;
        r = j.result;
    });
    if (st == JobState::cancelled) fail("request cancelled" + (err.empty() ? std::string() : ": " + err));
    if (st != JobState::done) fail(err.empty() ? "generation failed" : err);
    return r;
}

nlohmann::json result_info(const Request &req, const Result &r) {
    nlohmann::json loras = nlohmann::json::array();
    for (auto &l : req.loras) loras.push_back({{"name", l.name}, {"scale", l.scale}});
    nlohmann::json j = {{"seed", req.seed},
                        {"width", req.width},
                        {"height", req.height},
                        {"size", std::to_string(req.width) + "x" + std::to_string(req.height)},
                        {"task", task_name(req.task)},
                        {"loras", loras},
                        {"truncated", r.truncated},
                        {"seconds", r.seconds},
                        {"metrics", r.metrics}};
    return j;
}

std::string png_of(const Image &img) { return encode_png(img.rgb, img.width, img.height); }

std::string save_png(ServerContext &ctx, const std::string &png) {
    static std::atomic<uint64_t> counter{0};
    std::string name = "img_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(counter++) + ".png";
    std::ofstream(ctx.out_dir + "/" + name, std::ios::binary) << png;
    return name;
}

}  // namespace axd
