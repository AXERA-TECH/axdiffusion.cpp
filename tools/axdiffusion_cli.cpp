// axdiffusion-cli: run one request (or N repeats) from the command line.
//   axdiffusion-cli --model DIR --prompt TEXT [--size N | WxH] [--seed N] [--out out.png]
//                   [--image REF]... [--lora-dir DIR] [--lora NAME[:SCALE]]... [--repeat N] [--steps N] [--refine N]
//                   [--blocks N] [--dump-dir DIR] [--quiet]
#include <cstdio>
#include <cstdlib>
#include <string>

#include "common.hpp"
#include "json.hpp"
#include "model.hpp"

static void usage() {
    std::fprintf(stderr,
                 "usage: axdiffusion-cli --model DIR --prompt TEXT [--size N|WxH]\n"
                 "                       [--seed N] [--out PNG] [--image REF]... [--lora-dir DIR] [--lora NAME[:SCALE]]...\n"
                 "                       [--repeat N] [--steps N] [--blocks N] [--dump-dir DIR] [--quiet]\n"
                 "                       [--refine 1|2]   (two-pass: half-size image, then N steps at --size)\n");
}

int main(int argc, char **argv) {
    std::string model_dir, out = "out.png";
    axd::Request req;
    axd::ModelOptions opt;
    int repeat = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") model_dir = next();
        else if (a == "--prompt") req.prompt = next();
        else if (a == "--size") {  // N (square) or WxH
            const std::string v = next();
            const size_t x = v.find_first_of("xX");
            const std::string w = v.substr(0, x), h = x == std::string::npos ? w : v.substr(x + 1);
            auto digits = [](const std::string &t) { return !t.empty() && t.size() < 6 && t.find_first_not_of("0123456789") == std::string::npos; };
            if (!digits(w) || !digits(h)) { std::fprintf(stderr, "bad --size %s (use N or WxH)\n", v.c_str()); return 2; }
            req.width = std::stoi(w);
            req.height = std::stoi(h);
        }
        else if (a == "--width") req.width = std::stoi(next());
        else if (a == "--height") req.height = std::stoi(next());
        else if (a == "--seed") req.seed = std::stoull(next());
        else if (a == "--out") out = next();
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--steps") req.steps = std::stoi(next());
        else if (a == "--refine") req.refine_steps = std::stoi(next());
        else if (a == "--blocks") req.blocks = std::stoi(next());
        else if (a == "--dump-dir") req.dump_dir = next();
        else if (a == "--quiet") axd::g_verbose = false;
        else if (a == "--lora-dir") opt.lora_dir = next();
        else if (a == "--image") req.images.push_back(axd::load_image(next()));
        else if (a == "--lora") {
            std::string v = next();
            auto c = v.rfind(':');
            axd::LoraSpec s;
            if (c != std::string::npos && c + 1 < v.size() && v.find('/', c) == std::string::npos) {
                s.name = v.substr(0, c);
                s.scale = std::stof(v.substr(c + 1));
            } else {
                s.name = v;
            }
            req.loras.push_back(s);
        } else {
            usage();
            return 2;
        }
    }
    if (model_dir.empty() || req.prompt.empty()) {
        usage();
        return 2;
    }
    if (!req.images.empty()) req.task = axd::Task::edit;
    if (!req.loras.empty() && opt.lora_dir.empty()) opt.lora_dir = ".";
    try {
        auto model = axd::create_model(model_dir, opt);
        model->runtime().bind_thread();
        axd::Progress progress;
        progress.report = [](const std::string &stage, int step, int total) {
            if (axd::g_verbose && total) std::fprintf(stderr, "[axdiffusion] %s %d/%d\n", stage.c_str(), step, total);
        };
        for (int k = 0; k < repeat; ++k) {
            auto r = model->run(req, progress);
            std::string path = out;
            if (repeat > 1) {
                auto dot = out.rfind('.');
                path = out.substr(0, dot) + "_" + std::to_string(k) + (dot == std::string::npos ? "" : out.substr(dot));
            }
            double t0 = axd::now_s();
            if (!r.images.empty() && !axd::write_png(path, r.images[0].rgb, r.images[0].width, r.images[0].height))
                throw std::runtime_error("cannot write " + path);
            r.seconds["png"] = axd::now_s() - t0;
            size_t used = 0, total = 0;
            model->runtime().cmm(used, total);
            nlohmann::json j = {{"run", k},          {"out", path},          {"seed", req.seed},
                                {"tokens", r.token_ids.size()}, {"truncated", r.truncated},
                                {"load_s", model->load_seconds()}, {"seconds", r.seconds},
                                {"metrics", r.metrics}, {"cmm_used_gib", used / 1073741824.0}};
            if (!req.dump_dir.empty()) j["ids"] = r.token_ids;
            std::printf("%s\n", j.dump().c_str());
            std::fflush(stdout);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
