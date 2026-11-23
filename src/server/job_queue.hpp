// One NPU worker thread executing requests in submission order. HTTP threads only parse, submit and wait.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "model.hpp"

namespace axd {

enum class JobState { queued, running, done, failed, cancelled };
const char *job_state_name(JobState s);

struct Job {
    std::string id;
    Request request;
    Progress progress;
    // guarded by JobQueue's mutex
    JobState state = JobState::queued;
    std::string stage, error;
    int step = 0, total = 0;
    int position = 0;  // place in the queue when submitted
    double created = 0, started = 0, finished = 0;
    Result result;
};

// One line of the queue as other clients may see it (no prompt, no images).
struct QueueEntry {
    std::string id, task, stage;
    bool running = false;
    int width = 0, height = 0, step = 0, total = 0;
    double waited_s = 0;   // since submission
    double elapsed_s = 0;  // since it started running
    double expect_s = 0;   // how long a job of this shape took last time (0: unknown)
    double start_in_s = 0; // estimated wait until it starts (0 for the running job)
};

class JobQueue {
public:
    explicit JobQueue(ModelPlugin &model, size_t keep_finished = 64);
    ~JobQueue();
    std::shared_ptr<Job> submit(Request req);
    std::shared_ptr<Job> find(const std::string &id);
    bool cancel(const std::string &id);
    // Blocks until the job leaves queued/running. Returns false on timeout.
    bool wait(const std::shared_ptr<Job> &job, double timeout_s = 1e9);
    // Calls fn under the queue lock, so the job's fields are consistent.
    template <typename F> void read(const std::shared_ptr<Job> &job, F fn) {
        std::lock_guard<std::mutex> lk(mu_);
        fn(*job);
    }
    // Waits for any change of the job (progress or state) after `seen` updates; returns the new update count.
    uint64_t wait_update(const std::shared_ptr<Job> &job, uint64_t seen, double timeout_s);
    size_t pending() const;
    // Running job first, then the waiting jobs in order, with time estimates from earlier jobs of the same shape.
    std::vector<QueueEntry> snapshot() const;
    int served() const { return served_; }
    // NPU memory in use, sampled by the worker thread between jobs. The NPU runtime is only ever called from that
    // thread: HTTP threads must not query it while a job is running.
    void cmm(size_t &used, size_t &total) const {
        used = cmm_used_.load();
        total = cmm_total_.load();
    }
    // Stop accepting work; the running job finishes, queued jobs are cancelled.
    void shutdown();

private:
    void worker();
    ModelPlugin &model_;
    size_t keep_;
    mutable std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::string> finished_order_;
    std::map<const Job *, uint64_t> updates_;
    bool stop_ = false;
    int served_ = 0;
    std::map<std::string, double> typical_s_;  // shape key -> duration of the last such job
    double mean_s_ = 0;
    int mean_n_ = 0;
    std::shared_ptr<Job> running_;
    std::atomic<size_t> cmm_used_{0}, cmm_total_{0};
    void sample_cmm();
    uint64_t next_id_ = 1;
    std::thread thread_;
};

}  // namespace axd
