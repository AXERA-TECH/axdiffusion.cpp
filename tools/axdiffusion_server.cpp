// axdiffusion-server: keeps every weight resident on the NPU and serves images over HTTP.
//
//   axdiffusion-server --model DIR [--host 0.0.0.0] [--port 18500] [--out-dir ./output] [--model-name minimax-h3-image]
//
// Standard APIs:
//   OpenAI Images        POST /v1/images/generations   POST /v1/images/edits   GET /v1/models
//   A1111 / SD WebUI     POST /sdapi/v1/txt2img   POST /sdapi/v1/img2img   GET /sdapi/v1/sd-models  GET /sdapi/v1/options
// Others:
//   GET /health                 readiness, CMM usage
//   GET /files/<name>           images returned with response_format=url
//
// Only one image is generated at a time on the NPU; concurrent requests wait their turn.
#include <atomic>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <regex>
#include <string>

#include "common.hpp"
#include "httplib.h"
#include "json.hpp"
#include "pipeline.hpp"

using nlohmann::json;

namespace {

std::string base64(const std::string &in) {
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

std::string base64_decode(const std::string &in) {
    static int T[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) T[i] = -1;
        const char *a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[uint8_t(a[i])] = i;
        T[uint8_t('-')] = 62, T[uint8_t('_')] = 63;  // URL-safe alphabet
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

// Parse "1024x1024" style sizes. Only square 1024/2048 are compiled for now.
int parse_size(const std::string &s, int width, int height) {
    int w = width, h = height;
    if (!s.empty()) {
        auto x = s.find('x');
        if (x == std::string::npos) axd::fail("size must look like 1024x1024");
        w = std::stoi(s.substr(0, x));
        h = std::stoi(s.substr(x + 1));
    }
    if (w != h || (w != 1024 && w != 2048)) axd::fail("supported sizes: 1024x1024, 2048x2048");
    return w;
}

uint64_t pick_seed(int64_t seed) {
    if (seed >= 0) return uint64_t(seed);
    static std::mt19937_64 rng(std::random_device{}());
    return rng() & 0xFFFFFFFFull;
}

// A1111-style "<lora:name:weight>" tags are stripped from the prompt and returned separately.
std::string strip_lora_tags(const std::string &prompt, json &loras) {
    static const std::regex re(R"(<lora:([^:>]+)(?::([0-9.+-]+))?>)");
    std::string out;
    auto begin = std::sregex_iterator(prompt.begin(), prompt.end(), re);
    size_t last = 0;
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        loras.push_back({{"name", (*it)[1].str()}, {"scale", (*it)[2].matched ? std::stod((*it)[2].str()) : 1.0}});
        out += prompt.substr(last, size_t(it->position()) - last);
        last = size_t(it->position() + it->length());
    }
    out += prompt.substr(last);
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    std::string model, host = "0.0.0.0", out_dir = "output", model_name = "minimax-h3-image";
    int port = 18500, lora_cache = 4;
    std::string lora_dir;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--host") host = v;
        else if (a == "--port") port = std::stoi(v);
        else if (a == "--out-dir") out_dir = v;
        else if (a == "--model-name") model_name = v;
        else if (a == "--lora-dir") lora_dir = v;
        else if (a == "--lora-cache") lora_cache = std::stoi(v);
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: axdiffusion-server --model DIR [--host H] [--port P] [--out-dir DIR] [--model-name ID]\n"
                             "                    [--lora-dir DIR] [--lora-cache N]\n");
        return 2;
    }
    std::unique_ptr<axd::Pipeline> pipe;
    try {
        pipe = std::make_unique<axd::Pipeline>(model, lora_dir, lora_cache);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::filesystem::create_directories(out_dir);
    std::mutex mu;
    std::atomic<int> served{0};

    // Runs one generation and returns PNG bytes; timing details go to `info`.
    auto generate = [&](const std::string &prompt, int size, uint64_t seed, int steps, const json &loras, json &info,
                        const std::vector<axd::Image> &refs = {}) {
        axd::GenerateRequest g;
        g.references = refs;
        for (auto &l : loras) {
            axd::LoraSpec s;
            s.name = l.value("name", l.value("path", std::string()));
            s.scale = float(l.value("scale", l.value("weight", 1.0)));
            if (s.name.empty() || s.name.find('/') != std::string::npos || s.name.find("..") != std::string::npos)
                axd::fail("LoRA entries need a 'name' from GET /v1/loras");
            g.loras.push_back(s);
        }
        g.prompt = prompt;
        g.size = size;
        g.seed = seed;
        g.steps = steps;
        std::lock_guard<std::mutex> lock(mu);
        auto r = pipe->generate(g);
        ++served;
        info = {{"seed", seed},
                {"size", size},
                {"tokens", r.ids.size()},
                {"loras", loras},
                {"lora_build_s", r.lora_s},
                {"truncated", r.truncated},
                {"seconds",
                 {{"text_encoder", r.text_s}, {"dit", r.dit_prepare_s + r.dit_steps_s}, {"vae", r.vae_s},
                  {"total", r.total_s}}}};
        return axd::encode_png(r.rgb, r.width, r.height);
    };
    auto save = [&](const std::string &png) {
        static std::atomic<uint64_t> counter{0};
        std::string name = "img_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(counter++) + ".png";
        std::ofstream(out_dir + "/" + name, std::ios::binary) << png;
        return name;
    };
    auto error = [](httplib::Response &res, int status, const std::string &msg) {
        res.status = status;
        res.set_content(json{{"error", {{"message", msg}, {"type", "invalid_request_error"}}}}.dump(),
                        "application/json");
    };

    httplib::Server srv;
    srv.set_read_timeout(600, 0);
    srv.set_write_timeout(600, 0);

    srv.Get("/health", [&](const httplib::Request &, httplib::Response &res) {
        size_t used = 0, total = 0;
        pipe->cmm(used, total);
        res.set_content(json{{"status", "ok"},
                             {"model", model_name},
                             {"load_s", pipe->load_seconds()},
                             {"requests", served.load()},
                             {"cmm_used_gib", used / 1073741824.0}}
                            .dump(),
                        "application/json");
    });

    // ---------------- OpenAI Images API
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(json{{"object", "list"},
                             {"data", {{{"id", model_name}, {"object", "model"}, {"owned_by", "local"}}}}}
                            .dump(),
                        "application/json");
    });
    srv.Post("/v1/images/generations", [&](const httplib::Request &req, httplib::Response &res) {
        try {
            auto b = json::parse(req.body);
            std::string prompt = b.value("prompt", std::string());
            if (prompt.empty()) return error(res, 400, "prompt is required");
            int n = b.value("n", 1);
            if (n < 1 || n > 4) return error(res, 400, "n must be 1..4");
            int size = parse_size(b.value("size", std::string("1024x1024")), 1024, 1024);
            std::string fmt = b.value("response_format", std::string("b64_json"));
            if (fmt != "b64_json" && fmt != "url") return error(res, 400, "response_format must be b64_json or url");
            int64_t seed = b.value("seed", int64_t(-1));
            json loras = b.value("loras", json::array());
            json data = json::array(), infos = json::array();
            for (int i = 0; i < n; ++i) {
                json info;
                uint64_t s = seed >= 0 ? uint64_t(seed) + uint64_t(i) : pick_seed(-1);
                std::string png = generate(prompt, size, s, b.value("steps", axd::kSteps), loras, info);
                if (fmt == "url") {
                    data.push_back({{"url", "http://" + req.get_header_value("Host") + "/files/" + save(png)},
                                    {"revised_prompt", prompt}});
                } else {
                    data.push_back({{"b64_json", base64(png)}, {"revised_prompt", prompt}});
                }
                infos.push_back(info);
            }
            res.set_content(json{{"created", std::time(nullptr)}, {"data", data}, {"info", infos}}.dump(),
                            "application/json");
        } catch (const std::exception &e) {
            error(res, 400, e.what());
        }
    });
    // OpenAI image edits: multipart/form-data with image (or image[] repeated), prompt, n, size, response_format,
    // plus extensions seed and loras (JSON string).
    srv.Post("/v1/images/edits", [&](const httplib::Request &req, httplib::Response &res) {
        try {
            if (!req.is_multipart_form_data()) return error(res, 400, "expected multipart/form-data");
            auto field = [&](const char *k, const std::string &def) {
                return req.has_file(k) ? req.get_file_value(k).content : def;
            };
            std::vector<axd::Image> refs;
            for (const char *k : {"image", "image[]"})
                for (auto &f : req.get_file_values(k)) refs.push_back(axd::decode_image(f.content));
            if (refs.empty()) return error(res, 400, "at least one image is required");
            std::string prompt = field("prompt", "");
            if (prompt.empty()) return error(res, 400, "prompt is required");
            int n = std::stoi(field("n", "1"));
            if (n < 1 || n > 4) return error(res, 400, "n must be 1..4");
            int size = parse_size(field("size", "1024x1024"), 1024, 1024);
            std::string fmt = field("response_format", "b64_json");
            int64_t seed = std::stoll(field("seed", "-1"));
            json loras = json::parse(field("loras", "[]"));
            json data = json::array(), infos = json::array();
            for (int i = 0; i < n; ++i) {
                json info;
                uint64_t s = seed >= 0 ? uint64_t(seed) + uint64_t(i) : pick_seed(-1);
                std::string png = generate(prompt, size, s, axd::kSteps, loras, info, refs);
                if (fmt == "url")
                    data.push_back({{"url", "http://" + req.get_header_value("Host") + "/files/" + save(png)}});
                else
                    data.push_back({{"b64_json", base64(png)}});
                infos.push_back(info);
            }
            res.set_content(json{{"created", std::time(nullptr)}, {"data", data}, {"info", infos}}.dump(),
                            "application/json");
        } catch (const std::exception &e) {
            error(res, 400, e.what());
        }
    });
    srv.Get(R"(/files/([A-Za-z0-9_.-]+\.png))", [&](const httplib::Request &req, httplib::Response &res) {
        std::ifstream f(out_dir + "/" + req.matches[1].str(), std::ios::binary);
        if (!f) return error(res, 404, "not found");
        std::string png((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        res.set_content(png, "image/png");
    });

    // ---------------- A1111 / SD WebUI API
    srv.Post("/sdapi/v1/txt2img", [&](const httplib::Request &req, httplib::Response &res) {
        try {
            auto b = json::parse(req.body);
            json loras = json::array();
            std::string prompt = strip_lora_tags(b.value("prompt", std::string()), loras);
            if (prompt.empty()) return error(res, 400, "prompt is required");
            int size = parse_size("", b.value("width", 1024), b.value("height", 1024));
            int count = std::max(1, b.value("batch_size", 1)) * std::max(1, b.value("n_iter", 1));
            if (count > 4) return error(res, 400, "batch_size * n_iter must be <= 4");
            int64_t seed = b.value("seed", int64_t(-1));
            uint64_t s0 = pick_seed(seed);
            json images = json::array(), seeds = json::array();
            for (int i = 0; i < count; ++i) {
                json info;
                std::string png = generate(prompt, size, s0 + uint64_t(i), axd::kSteps, loras, info);
                images.push_back(base64(png));
                seeds.push_back(s0 + uint64_t(i));
            }
            json params = b;
            json info = {{"prompt", prompt}, {"seed", s0}, {"all_seeds", seeds}, {"width", size}, {"height", size},
                         {"steps", axd::kSteps}, {"sd_model_name", model_name}};
            res.set_content(json{{"images", images}, {"parameters", params}, {"info", info.dump()}}.dump(),
                            "application/json");
        } catch (const std::exception &e) {
            error(res, 400, e.what());
        }
    });
    // A1111 img2img: init_images are base64 (optionally data-URI) reference images; denoising_strength is
    // ignored (the model edits from references with a fixed 6-step schedule).
    srv.Post("/sdapi/v1/img2img", [&](const httplib::Request &req, httplib::Response &res) {
        try {
            auto b = json::parse(req.body);
            std::vector<axd::Image> refs;
            for (auto &s : b.value("init_images", json::array())) {
                std::string v = s.get<std::string>();
                auto comma = v.find(',');
                if (v.rfind("data:", 0) == 0 && comma != std::string::npos) v = v.substr(comma + 1);
                refs.push_back(axd::decode_image(base64_decode(v)));
            }
            if (refs.empty()) return error(res, 400, "init_images is required");
            json loras = json::array();
            std::string prompt = strip_lora_tags(b.value("prompt", std::string()), loras);
            if (prompt.empty()) return error(res, 400, "prompt is required");
            int size = parse_size("", b.value("width", 1024), b.value("height", 1024));
            uint64_t s0 = pick_seed(b.value("seed", int64_t(-1)));
            json info;
            std::string png = generate(prompt, size, s0, axd::kSteps, loras, info, refs);
            res.set_content(json{{"images", {base64(png)}}, {"parameters", b}, {"info", info.dump()}}.dump(),
                            "application/json");
        } catch (const std::exception &e) {
            error(res, 400, e.what());
        }
    });
    srv.Get("/sdapi/v1/sd-models", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(json::array({{{"title", model_name}, {"model_name", model_name}, {"filename", model}}}).dump(),
                        "application/json");
    });
    srv.Get("/sdapi/v1/options", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(json{{"sd_model_checkpoint", model_name}}.dump(), "application/json");
    });
    auto list_loras = [&](httplib::Response &res, bool a1111) {
        json arr = json::array();
        for (auto &n : pipe->loras()) {
            if (a1111) arr.push_back({{"name", n}, {"alias", n}, {"path", n + ".safetensors"}});
            else arr.push_back({{"name", n}});
        }
        res.set_content(a1111 ? arr.dump() : json{{"data", arr}}.dump(), "application/json");
    };
    srv.Get("/v1/loras", [&](const httplib::Request &, httplib::Response &res) { list_loras(res, false); });
    srv.Get("/sdapi/v1/loras", [&](const httplib::Request &, httplib::Response &res) { list_loras(res, true); });
    srv.Get("/sdapi/v1/samplers", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(json::array({{{"name", "Euler (6-step distilled)"}, {"aliases", json::array()}}}).dump(),
                        "application/json");
    });

    std::fprintf(stderr, "[axdiffusion] listening on %s:%d (OpenAI /v1/images/generations, A1111 /sdapi/v1/txt2img)\n",
                 host.c_str(), port);
    if (!srv.listen(host, port)) {
        std::fprintf(stderr, "error: cannot listen on %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
