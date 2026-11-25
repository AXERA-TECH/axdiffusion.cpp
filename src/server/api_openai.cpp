// OpenAI Images API: POST /v1/images/generations, POST /v1/images/edits (multipart), GET /v1/models.
// Extensions: seed, steps, refine_steps, loras. Also GET /v1/loras and GET /files/<name>.png (response_format=url).
#include <ctime>
#include <fstream>

#include "api.hpp"
#include "common.hpp"

namespace axd {

namespace {
nlohmann::json images_response(ServerContext &ctx, const httplib::Request &http, const std::vector<Request> &reqs,
                               const std::vector<Result> &results, const std::string &fmt) {
    nlohmann::json data = nlohmann::json::array(), infos = nlohmann::json::array();
    for (size_t i = 0; i < results.size(); ++i)
        for (auto &img : results[i].images) {
            std::string png = png_of(img);
            if (fmt == "url")
                data.push_back({{"url", "http://" + http.get_header_value("Host") + "/files/" + save_png(ctx, png)},
                                {"revised_prompt", reqs[i].prompt}});
            else
                data.push_back({{"b64_json", base64_encode(png)}, {"revised_prompt", reqs[i].prompt}});
            infos.push_back(result_info(reqs[i], results[i]));
        }
    return {{"created", std::time(nullptr)}, {"data", data}, {"info", infos}};
}
}  // namespace

void register_openai(httplib::Server &srv, ServerContext &ctx) {
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response &res) {
        auto info = ctx.model.info();
        nlohmann::json tasks = nlohmann::json::array();
        for (auto t : info.tasks) tasks.push_back(task_name(t));
        res.set_content(nlohmann::json{{"object", "list"},
                                       {"data",
                                        {{{"id", ctx.model_name},
                                          {"object", "model"},
                                          {"owned_by", "local"},
                                          {"arch", info.arch},
                                          {"tasks", tasks}}}}}
                            .dump(),
                        "application/json");
    });
    srv.Get("/v1/loras", [&](const httplib::Request &, httplib::Response &res) {
        nlohmann::json arr = nlohmann::json::array();
        for (auto &n : ctx.model.loras()) arr.push_back({{"name", n}});
        res.set_content(nlohmann::json{{"data", arr}}.dump(), "application/json");
    });
    srv.Post("/v1/images/generations", [&](const httplib::Request &http, httplib::Response &res) {
        try {
            auto b = nlohmann::json::parse(http.body);
            Request base;
            base.task = Task::txt2img;
            base.prompt = b.value("prompt", std::string());
            if (base.prompt.empty()) return json_error(res, 400, "prompt is required");
            parse_size(b.value("size", std::string("1024x1024")), base.width, base.height);
            base.steps = b.value("steps", 0);
            base.refine_steps = b.value("refine_steps", 0);
            base.loras = loras_from_json(b.value("loras", nlohmann::json::array()));
            int n = b.value("n", 1);
            if (n < 1 || n > 4) return json_error(res, 400, "n must be 1..4");
            std::string fmt = b.value("response_format", std::string("b64_json"));
            if (fmt != "b64_json" && fmt != "url") return json_error(res, 400, "response_format must be b64_json or url");
            int64_t seed = b.value("seed", int64_t(-1));
            std::vector<Request> reqs;
            std::vector<Result> results;
            for (int i = 0; i < n; ++i) {
                Request r = base;
                r.seed = seed >= 0 ? uint64_t(seed) + uint64_t(i) : resolve_seed(-1);
                results.push_back(run_sync(ctx, r));
                reqs.push_back(r);
            }
            res.set_content(images_response(ctx, http, reqs, results, fmt).dump(), "application/json");
        } catch (const std::exception &e) {
            json_error(res, 400, e.what());
        }
    });
    srv.Post("/v1/images/edits", [&](const httplib::Request &http, httplib::Response &res) {
        try {
            if (!http.is_multipart_form_data()) return json_error(res, 400, "expected multipart/form-data");
            auto field = [&](const char *k, const std::string &def) {
                return http.has_file(k) ? http.get_file_value(k).content : def;
            };
            Request base;
            base.task = Task::edit;
            for (const char *k : {"image", "image[]"})
                for (auto &f : http.get_file_values(k)) base.images.push_back(decode_image(f.content));
            if (base.images.empty()) return json_error(res, 400, "at least one image is required");
            base.prompt = field("prompt", "");
            if (base.prompt.empty()) return json_error(res, 400, "prompt is required");
            parse_size(field("size", "1024x1024"), base.width, base.height);
            base.loras = loras_from_json(nlohmann::json::parse(field("loras", "[]")));
            int n = std::stoi(field("n", "1"));
            if (n < 1 || n > 4) return json_error(res, 400, "n must be 1..4");
            std::string fmt = field("response_format", "b64_json");
            int64_t seed = std::stoll(field("seed", "-1"));
            std::vector<Request> reqs;
            std::vector<Result> results;
            for (int i = 0; i < n; ++i) {
                Request r = base;
                r.seed = seed >= 0 ? uint64_t(seed) + uint64_t(i) : resolve_seed(-1);
                results.push_back(run_sync(ctx, r));
                reqs.push_back(r);
            }
            res.set_content(images_response(ctx, http, reqs, results, fmt).dump(), "application/json");
        } catch (const std::exception &e) {
            json_error(res, 400, e.what());
        }
    });
    srv.Get(R"(/files/([A-Za-z0-9_.-]+\.png))", [&](const httplib::Request &http, httplib::Response &res) {
        std::ifstream f(ctx.out_dir + "/" + http.matches[1].str(), std::ios::binary);
        if (!f) return json_error(res, 404, "not found");
        std::string png((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        res.set_content(png, "image/png");
    });
}

}  // namespace axd
