// Built-in web page and health check.
#include <fstream>
#include <sstream>

#include "api.hpp"
#include "web_ui.hpp"

namespace axd {

void register_web(httplib::Server &srv, ServerContext &ctx) {
    // With --web-dir the page is read from disk on every request, so editing index.html takes effect on the next
    // browser refresh; other files in that directory are served under /web/.
    for (const char *path : {"/", "/index.html"})
        srv.Get(path, [&](const httplib::Request &, httplib::Response &res) {
            res.set_header("Cache-Control", "no-cache");
            if (!ctx.web_dir.empty()) {
                std::ifstream f(ctx.web_dir + "/index.html", std::ios::binary);
                if (f) {
                    std::stringstream ss;
                    ss << f.rdbuf();
                    return res.set_content(ss.str(), "text/html; charset=utf-8");
                }
            }
            res.set_content(kWebUi, "text/html; charset=utf-8");
        });
    if (!ctx.web_dir.empty()) srv.set_mount_point("/web", ctx.web_dir);
    srv.Get("/health", [&](const httplib::Request &, httplib::Response &res) {
        size_t used = 0, total = 0;
        ctx.queue.cmm(used, total);  // cached: the NPU runtime belongs to the worker thread
        auto info = ctx.model.info();
        nlohmann::json tasks = nlohmann::json::array(), sizes = nlohmann::json::array();
        for (auto t : info.tasks) tasks.push_back(task_name(t));
        for (auto &s : info.sizes) sizes.push_back(std::to_string(s[0]) + "x" + std::to_string(s[1]));
        res.set_content(nlohmann::json{{"status", "ok"},
                                       {"model", ctx.model_name},
                                       {"arch", info.arch},
                                       {"tasks", tasks},
                                       {"sizes", sizes},
                                       {"lora", info.lora},
                                       {"max_references", info.max_references},
                                       {"load_s", ctx.model.load_seconds()},
                                       {"requests", ctx.queue.served()},
                                       {"queued", ctx.queue.pending()},
                                       {"cmm_used_gib", used / 1073741824.0}}
                            .dump(),
                        "application/json");
    });
}

}  // namespace axd
