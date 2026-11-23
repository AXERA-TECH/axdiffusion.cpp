// axdiffusion-server: loads one model directory, keeps every weight resident on the NPU and serves it over HTTP.
//
//   axdiffusion-server --model DIR [--host 0.0.0.0] [--port 18500] [--out-dir ./output] [--model-name NAME]
//                      [--lora-dir DIR] [--lora-cache N] [--web-dir DIR]
//
// --web-dir DIR serves DIR/index.html as the web page (read on every request) in place of the built-in one.
//
// APIs: web UI (/), OpenAI Images (/v1/images/*), A1111 (/sdapi/v1/*), async jobs (/v1/jobs), /health.
// One NPU worker runs requests in order; SIGINT/SIGTERM stop accepting work and let the running job finish.
#include <csignal>
#include <filesystem>
#include <string>

#include "api.hpp"
#include "common.hpp"
#include "model.hpp"

static httplib::Server *g_server = nullptr;

static void on_signal(int) {
    if (g_server) g_server->stop();
}

int main(int argc, char **argv) {
    std::string model_dir, host = "0.0.0.0", out_dir = "output", model_name, web_dir;
    int port = 18500;
    axd::ModelOptions opt;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model_dir = v;
        else if (a == "--host") host = v;
        else if (a == "--port") port = std::stoi(v);
        else if (a == "--out-dir") out_dir = v;
        else if (a == "--model-name") model_name = v;
        else if (a == "--web-dir") web_dir = v;
        else if (a == "--lora-dir") opt.lora_dir = v;
        else if (a == "--lora-cache") opt.lora_cache = std::stoi(v);
        else if (a == "--device") opt.device = std::stoi(v);
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (model_dir.empty()) {
        std::fprintf(stderr,
                     "usage: axdiffusion-server --model DIR [--host H] [--port P] [--out-dir DIR] [--model-name ID]\n"
                     "                          [--lora-dir DIR] [--lora-cache N] [--web-dir DIR]\n");
        return 2;
    }
    std::unique_ptr<axd::ModelPlugin> model;
    try {
        model = axd::create_model(model_dir, opt);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    if (model_name.empty()) model_name = model->info().name;
    std::filesystem::create_directories(out_dir);

    int rc = 0;
    {
        axd::JobQueue queue(*model);
        axd::ServerContext ctx{*model, queue, out_dir, model_name, web_dir};
        httplib::Server srv;
        srv.set_read_timeout(600, 0);
        srv.set_write_timeout(600, 0);
        srv.set_payload_max_length(size_t(256) << 20);
        // Browser clients (the built-in page, other web front ends) may call the API cross-origin.
        srv.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                                 {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
                                 {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        srv.Options(R"(.*)", [](const httplib::Request &, httplib::Response &res) { res.status = 204; });
        axd::register_web(srv, ctx);
        axd::register_openai(srv, ctx);
        axd::register_a1111(srv, ctx);
        axd::register_jobs(srv, ctx);

        g_server = &srv;
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::fprintf(stderr, "[axdiffusion] %s on %s:%d (web /, OpenAI /v1/images/*, A1111 /sdapi/v1/*, jobs /v1/jobs)\n",
                     model_name.c_str(), host.c_str(), port);
        if (!srv.listen(host, port)) {
            std::fprintf(stderr, "error: cannot listen on %s:%d\n", host.c_str(), port);
            rc = 1;
        }
        g_server = nullptr;
        std::fprintf(stderr, "[axdiffusion] stopping: finishing the running job\n");
        queue.shutdown();
    }  // queue joins its worker here, before the model (and NPU runtime) is released
    model.reset();
    return rc;
}
