// axdiffusion-cli: generate one image from a prompt.
//   axdiffusion-cli --model <MiniMax-H3-Image-AX8860 dir> --prompt "..." [--size 1024|2048] [--seed 42]
//             [--out out.png] [--repeat N] [--steps 6] [--blocks 50] [--dump-dir DIR] [--quiet]
#include <cstdio>
#include <cstdlib>
#include <string>

#include "common.hpp"
#include "json.hpp"
#include "pipeline.hpp"

static void usage() {
    std::fprintf(stderr,
                 "usage: axdiffusion-cli --model DIR --prompt TEXT [--size 1024|2048] [--seed N] [--out PNG]\n"
                 "                 [--repeat N] [--steps 1..6] [--blocks 1..50] [--dump-dir DIR] [--quiet]\n"
                 "                 [--lora-dir DIR] [--lora NAME[:SCALE]]...\n");
}

int main(int argc, char **argv) {
    std::string model;
    axd::GenerateRequest req;
    req.out_png = "out.png";
    int repeat = 1;
    std::string lora_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--prompt") req.prompt = next();
        else if (a == "--size") req.size = std::stoi(next());
        else if (a == "--seed") req.seed = std::stoull(next());
        else if (a == "--out") req.out_png = next();
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--steps") req.steps = std::stoi(next());
        else if (a == "--blocks") req.blocks = std::stoi(next());
        else if (a == "--dump-dir") req.dump_dir = next();
        else if (a == "--quiet") axd::g_verbose = false;
        else if (a == "--lora-dir") lora_dir = next();
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
        }
        else {
            usage();
            return 2;
        }
    }
    if (model.empty() || req.prompt.empty()) {
        usage();
        return 2;
    }
    try {
        if (!req.loras.empty() && lora_dir.empty()) lora_dir = ".";
        axd::Pipeline pipe(model, lora_dir);
        for (int k = 0; k < repeat; ++k) {
            auto r = pipe.generate(req);
            size_t used = 0, total = 0;
            pipe.cmm(used, total);
            nlohmann::json j = {{"run", k},
                                {"tokens", r.ids.size()},
                                {"truncated", r.truncated},
                                {"used", r.used},
                                {"padded", r.padded},
                                {"load_s", pipe.load_seconds()},
                                {"lora_s", r.lora_s},
                                {"tokenize_s", r.tokenize_s},
                                {"text_s", r.text_s},
                                {"dit_prepare_s", r.dit_prepare_s},
                                {"dit_steps_s", r.dit_steps_s},
                                {"step_s", r.step_seconds},
                                {"block_mean_s", r.block_mean_s},
                                {"vae_s", r.vae_s},
                                {"png_s", r.png_s},
                                {"total_s", r.total_s},
                                {"cmm_used_gib", used / 1073741824.0},
                                {"out", req.out_png}};
            if (!req.dump_dir.empty()) j["ids"] = r.ids;
            std::printf("%s\n", j.dump().c_str());
            std::fflush(stdout);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
