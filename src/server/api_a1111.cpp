// A1111 / SD WebUI API: txt2img, img2img (init_images = references), model/LoRA/option listings.
// steps, cfg_scale, sampler_name and negative_prompt are accepted and ignored by distilled models.
// enable_hr (hr_scale 2, hr_second_pass_steps 1 or 2) maps to the two-pass refine.
#include "api.hpp"
#include "common.hpp"

namespace axd {

namespace {
void run_a1111(ServerContext &ctx, const nlohmann::json &b, Request base, httplib::Response &res) {
    base.prompt = strip_lora_tags(b.value("prompt", std::string()), base.loras);
    if (base.prompt.empty()) return json_error(res, 400, "prompt is required");
    base.width = b.value("width", 1024);
    base.height = b.value("height", 1024);
    if (b.value("enable_hr", false)) {  // hires fix: width x height is the first pass, the result is hr_scale times that
        if (base.task != Task::txt2img) return json_error(res, 400, "enable_hr is for txt2img only");
        if (b.value("hr_scale", 2.0) != 2.0) return json_error(res, 400, "hr_scale must be 2");
        base.width *= 2;
        base.height *= 2;
        base.refine_steps = b.value("hr_second_pass_steps", 0) == 2 ? 2 : 1;
    }
    int count = std::max(1, b.value("batch_size", 1)) * std::max(1, b.value("n_iter", 1));
    if (count > 4) return json_error(res, 400, "batch_size * n_iter must be <= 4");
    uint64_t s0 = resolve_seed(b.value("seed", int64_t(-1)));
    nlohmann::json images = nlohmann::json::array(), seeds = nlohmann::json::array();
    nlohmann::json last;
    for (int i = 0; i < count; ++i) {
        Request r = base;
        r.seed = s0 + uint64_t(i);
        Result out = run_sync(ctx, r);
        for (auto &img : out.images) images.push_back(base64_encode(png_of(img)));
        seeds.push_back(r.seed);
        last = result_info(r, out);
    }
    nlohmann::json info = {{"prompt", base.prompt}, {"seed", s0},       {"all_seeds", seeds},
                           {"width", base.width},   {"height", base.height}, {"sd_model_name", ctx.model_name},
                           {"timings", last}};
    res.set_content(nlohmann::json{{"images", images}, {"parameters", b}, {"info", info.dump()}}.dump(),
                    "application/json");
}
}  // namespace

void register_a1111(httplib::Server &srv, ServerContext &ctx) {
    srv.Post("/sdapi/v1/txt2img", [&](const httplib::Request &http, httplib::Response &res) {
        try {
            Request r;
            r.task = Task::txt2img;
            run_a1111(ctx, nlohmann::json::parse(http.body), r, res);
        } catch (const std::exception &e) {
            json_error(res, 400, e.what());
        }
    });
    srv.Post("/sdapi/v1/img2img", [&](const httplib::Request &http, httplib::Response &res) {
        try {
            auto b = nlohmann::json::parse(http.body);
            Request r;
            r.task = Task::edit;
            for (auto &s : b.value("init_images", nlohmann::json::array()))
                r.images.push_back(decode_image(base64_decode(s.get<std::string>())));
            if (r.images.empty()) return json_error(res, 400, "init_images is required");
            run_a1111(ctx, b, r, res);
        } catch (const std::exception &e) {
            json_error(res, 400, e.what());
        }
    });
    srv.Get("/sdapi/v1/sd-models", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(nlohmann::json::array({{{"title", ctx.model_name}, {"model_name", ctx.model_name}}}).dump(),
                        "application/json");
    });
    srv.Get("/sdapi/v1/loras", [&](const httplib::Request &, httplib::Response &res) {
        nlohmann::json arr = nlohmann::json::array();
        for (auto &n : ctx.model.loras()) arr.push_back({{"name", n}, {"alias", n}, {"path", n + ".safetensors"}});
        res.set_content(arr.dump(), "application/json");
    });
    srv.Get("/sdapi/v1/options", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(nlohmann::json{{"sd_model_checkpoint", ctx.model_name}}.dump(), "application/json");
    });
    srv.Get("/sdapi/v1/samplers", [&](const httplib::Request &, httplib::Response &res) {
        res.set_content(nlohmann::json::array({{{"name", "Euler (distilled)"}, {"aliases", nlohmann::json::array()}}}).dump(),
                        "application/json");
    });
}

}  // namespace axd
