// Asynchronous jobs (used by the web UI and by long tasks such as video):
//   POST   /v1/jobs              {task, prompt, width, height, seed, steps, loras, images:[base64], frames, fps}
//   GET    /v1/jobs/{id}         status, progress; when done also images (base64 PNG) and timings
//   GET    /v1/jobs/{id}/events  Server-Sent Events: one "progress" event per change, then "done"
//   DELETE /v1/jobs/{id}         cancel (queued jobs are dropped, a running job stops at the next step)
#include <map>
#include <mutex>

#include "api.hpp"
#include "common.hpp"

namespace axd {

namespace {
std::mutex png_mu;
std::map<std::string, std::vector<std::string>> png_cache;  // job id -> base64 PNGs

nlohmann::json job_json(ServerContext &ctx, const std::shared_ptr<Job> &job, bool with_result) {
    nlohmann::json j;
    JobState st = JobState::queued;
    Result result;
    ctx.queue.read(job, [&](Job &x) {
        st = x.state;
        j = {{"id", x.id},
             {"object", "job"},
             {"task", task_name(x.request.task)},
             {"status", job_state_name(x.state)},
             {"stage", x.stage},
             {"progress", {{"step", x.step}, {"total", x.total}}},
             {"created", x.created}};
        if (!x.error.empty()) j["error"] = x.error;
        if (x.started > 0) j["started"] = x.started;
        if (x.finished > 0) j["finished"] = x.finished;
        if (with_result && x.state == JobState::done) result = x.result;
    });
    if (with_result && st == JobState::done) {
        std::vector<std::string> b64;
        {
            std::lock_guard<std::mutex> lk(png_mu);
            auto it = png_cache.find(job->id);
            if (it != png_cache.end()) b64 = it->second;
        }
        if (b64.empty()) {
            for (auto &img : result.images) b64.push_back(base64_encode(png_of(img)));
            std::lock_guard<std::mutex> lk(png_mu);
            if (png_cache.size() > 16) png_cache.erase(png_cache.begin());
            png_cache[job->id] = b64;
        }
        j["images"] = b64;
        j["info"] = result_info(job->request, result);
    }
    return j;
}
}  // namespace

void register_jobs(httplib::Server &srv, ServerContext &ctx) {
    srv.Post("/v1/jobs", [&](const httplib::Request &http, httplib::Response &res) {
        try {
            auto b = nlohmann::json::parse(http.body);
            Request r;
            std::string task = b.value("task", std::string("txt2img"));
            if (!parse_task(task, r.task)) return json_error(res, 400, "unknown task " + task);
            r.prompt = b.value("prompt", std::string());
            if (r.prompt.empty()) return json_error(res, 400, "prompt is required");
            r.negative_prompt = b.value("negative_prompt", std::string());
            r.width = b.value("width", 1024);
            r.height = b.value("height", 1024);
            r.frames = b.value("frames", 1);
            r.fps = b.value("fps", 0.0f);
            r.steps = b.value("steps", 0);
            r.cfg = b.value("cfg", 0.0f);
            r.seed = resolve_seed(b.value("seed", int64_t(-1)));
            r.loras = loras_from_json(b.value("loras", nlohmann::json::array()));
            for (auto &s : b.value("images", nlohmann::json::array()))
                r.images.push_back(decode_image(base64_decode(s.get<std::string>())));
            if (!r.images.empty() && r.task == Task::txt2img) r.task = Task::edit;
            auto job = ctx.queue.submit(std::move(r));
            res.status = 202;
            res.set_content(job_json(ctx, job, false).dump(), "application/json");
        } catch (const std::exception &e) {
            json_error(res, 400, e.what());
        }
    });
    srv.Get(R"(/v1/jobs/([A-Za-z0-9_]+))", [&](const httplib::Request &http, httplib::Response &res) {
        auto job = ctx.queue.find(http.matches[1].str());
        if (!job) return json_error(res, 404, "job not found");
        res.set_content(job_json(ctx, job, true).dump(), "application/json");
    });
    srv.Delete(R"(/v1/jobs/([A-Za-z0-9_]+))", [&](const httplib::Request &http, httplib::Response &res) {
        if (!ctx.queue.cancel(http.matches[1].str())) return json_error(res, 404, "job not found");
        res.set_content(nlohmann::json{{"id", http.matches[1].str()}, {"cancel_requested", true}}.dump(),
                        "application/json");
    });
    srv.Get(R"(/v1/jobs/([A-Za-z0-9_]+)/events)", [&](const httplib::Request &http, httplib::Response &res) {
        auto job = ctx.queue.find(http.matches[1].str());
        if (!job) return json_error(res, 404, "job not found");
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [&ctx, job, seen = uint64_t(-1)](size_t, httplib::DataSink &sink) mutable {
            uint64_t now = ctx.queue.wait_update(job, seen, 15.0);
            bool finished = false;
            ctx.queue.read(job, [&](Job &x) { finished = x.state != JobState::queued && x.state != JobState::running; });
            if (now == seen && !finished) {  // keep-alive comment
                return sink.write(": ping\n\n", 8);
            }
            seen = now;
            std::string ev = std::string("event: ") + (finished ? "done" : "progress") + "\ndata: " +
                             job_json(ctx, job, false).dump() + "\n\n";
            if (!sink.write(ev.data(), ev.size())) return false;
            if (finished) sink.done();
            return true;
        });
    });
}

}  // namespace axd
