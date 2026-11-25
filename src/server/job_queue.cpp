#include "job_queue.hpp"

#include <chrono>
#include <random>

#include "common.hpp"

namespace axd {

const char *job_state_name(JobState s) {
    switch (s) {
        case JobState::queued: return "queued";
        case JobState::running: return "running";
        case JobState::done: return "succeeded";
        case JobState::failed: return "failed";
        case JobState::cancelled: return "cancelled";
    }
    return "unknown";
}

JobQueue::JobQueue(ModelPlugin &model, size_t keep_finished) : model_(model), keep_(keep_finished) {
    thread_ = std::thread([this] { worker(); });
}

JobQueue::~JobQueue() {
    shutdown();
    if (thread_.joinable()) thread_.join();
}

std::shared_ptr<Job> JobQueue::submit(Request req) {
    auto job = std::make_shared<Job>();
    job->request = std::move(req);
    job->created = now_s();
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) fail("server is shutting down");
    static std::mt19937_64 rng(std::random_device{}());
    char id[48];
    std::snprintf(id, sizeof(id), "job_%06llu_%08llx", (unsigned long long)next_id_++, (unsigned long long)(rng() & 0xFFFFFFFFull));
    job->id = id;
    job->position = int(queue_.size());
    Job *raw = job.get();
    job->progress.report = [this, raw](const std::string &stage, int step, int total) {
        std::lock_guard<std::mutex> l(mu_);
        raw->stage = stage;
        raw->step = step;
        raw->total = total;
        ++updates_[raw];
        done_cv_.notify_all();
    };
    jobs_[job->id] = job;
    queue_.push_back(job);
    cv_.notify_one();
    return job;
}

std::shared_ptr<Job> JobQueue::find(const std::string &id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = jobs_.find(id);
    return it == jobs_.end() ? nullptr : it->second;
}

bool JobQueue::cancel(const std::string &id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = jobs_.find(id);
    if (it == jobs_.end()) return false;
    auto &job = it->second;
    job->progress.cancel = true;
    if (job->state == JobState::queued) {
        for (auto q = queue_.begin(); q != queue_.end(); ++q)
            if (q->get() == job.get()) {
                queue_.erase(q);
                break;
            }
        job->state = JobState::cancelled;
        job->finished = now_s();
        finished_order_.push_back(job->id);
        ++updates_[job.get()];
        done_cv_.notify_all();
    }
    return true;
}

bool JobQueue::wait(const std::shared_ptr<Job> &job, double timeout_s) {
    std::unique_lock<std::mutex> lk(mu_);
    return done_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] {
        return job->state != JobState::queued && job->state != JobState::running;
    });
}

uint64_t JobQueue::wait_update(const std::shared_ptr<Job> &job, uint64_t seen, double timeout_s) {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] { return updates_[job.get()] != seen; });
    return updates_[job.get()];
}

namespace {
std::string shape_key(const Request &r) {
    return std::string(task_name(r.task)) + "|" + std::to_string(r.width) + "x" + std::to_string(r.height) + "|" +
           std::to_string(r.images.size()) + "|" + (r.loras.empty() ? "" : "lora") + "|" + std::to_string(r.refine_steps);
}
}  // namespace

std::vector<QueueEntry> JobQueue::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<QueueEntry> out;
    const double now = now_s();
    double ahead = 0;
    auto add = [&](const Job &j, bool running) {
        QueueEntry e;
        e.id = j.id;
        e.task = task_name(j.request.task);
        e.stage = j.stage;
        e.running = running;
        e.width = j.request.width;
        e.height = j.request.height;
        e.step = j.step;
        e.total = j.total;
        e.waited_s = now - j.created;
        e.elapsed_s = running ? now - j.started : 0;
        auto it = typical_s_.find(shape_key(j.request));
        e.expect_s = it != typical_s_.end() ? it->second : mean_s_;
        e.start_in_s = running ? 0 : ahead;
        // A running job that has outlived its estimate is assumed to need a little longer, never zero.
        ahead += running ? std::max(1.0, e.expect_s - e.elapsed_s) : e.expect_s;
        out.push_back(std::move(e));
    };
    if (running_) add(*running_, true);
    for (auto &j : queue_) add(*j, false);
    return out;
}

size_t JobQueue::pending() const {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.size();
}

void JobQueue::shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) return;
    stop_ = true;
    for (auto &job : queue_) {
        job->state = JobState::cancelled;
        job->error = "server shutting down";
        job->finished = now_s();
        ++updates_[job.get()];
    }
    queue_.clear();
    cv_.notify_all();
    done_cv_.notify_all();
}

void JobQueue::sample_cmm() {
    try {
        size_t used = 0, total = 0;
        model_.runtime().cmm(used, total);
        cmm_used_ = used;
        cmm_total_ = total;
    } catch (const std::exception &) {
    }
}

void JobQueue::worker() {
    model_.runtime().bind_thread();  // this thread owns the NPU
    sample_cmm();
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stop requested and nothing left
            job = queue_.front();
            queue_.pop_front();
            job->state = JobState::running;
            job->started = now_s();
            running_ = job;
            ++updates_[job.get()];
            done_cv_.notify_all();
        }
        JobState final_state = JobState::done;
        std::string error;
        Result result;
        try {
            result = model_.run(job->request, job->progress);
        } catch (const Cancelled &) {
            final_state = JobState::cancelled;
        } catch (const std::exception &e) {
            final_state = JobState::failed;
            error = e.what();
        }
        sample_cmm();
        std::lock_guard<std::mutex> lk(mu_);
        job->result = std::move(result);
        job->state = final_state;
        job->error = error;
        job->finished = now_s();
        running_.reset();
        if (final_state == JobState::done) {
            ++served_;
            const double took = job->finished - job->started;
            typical_s_[shape_key(job->request)] = took;
            mean_s_ = (mean_s_ * mean_n_ + took) / (mean_n_ + 1);
            ++mean_n_;
        }
        ++updates_[job.get()];
        finished_order_.push_back(job->id);
        while (finished_order_.size() > keep_) {
            auto it = jobs_.find(finished_order_.front());
            if (it != jobs_.end()) {
                updates_.erase(it->second.get());
                jobs_.erase(it);
            }
            finished_order_.pop_front();
        }
        done_cv_.notify_all();
    }
}

}  // namespace axd
