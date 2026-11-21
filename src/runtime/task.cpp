#include "task.hpp"

namespace axd {

static const std::pair<Task, const char *> kTasks[] = {{Task::txt2img, "txt2img"},
                                                       {Task::edit, "edit"},
                                                       {Task::txt2video, "txt2video"},
                                                       {Task::img2video, "img2video"},
                                                       {Task::ref2video, "ref2video"}};

const char *task_name(Task t) {
    for (auto &k : kTasks)
        if (k.first == t) return k.second;
    return "unknown";
}

bool parse_task(const std::string &s, Task &t) {
    for (auto &k : kTasks)
        if (s == k.second) {
            t = k.first;
            return true;
        }
    return false;
}

}  // namespace axd
