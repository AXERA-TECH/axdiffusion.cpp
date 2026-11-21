#include "model.hpp"

#include <functional>
#include <utility>

#include "common.hpp"
#include "model_dir.hpp"

namespace axd {

// Built-in plugins. Each model directory adds one factory here.
std::unique_ptr<ModelPlugin> create_minimax_h3_image(const std::string &dir, const ModelOptions &opt);

namespace {
using Factory = std::function<std::unique_ptr<ModelPlugin>(const std::string &, const ModelOptions &)>;
const std::vector<std::pair<std::string, Factory>> &registry() {
    static const std::vector<std::pair<std::string, Factory>> r = {
        {"minimax-h3-image", create_minimax_h3_image},
    };
    return r;
}
}  // namespace

std::vector<std::string> supported_architectures() {
    std::vector<std::string> out;
    for (auto &kv : registry()) out.push_back(kv.first);
    return out;
}

std::unique_ptr<ModelPlugin> create_model(const std::string &dir, const ModelOptions &opt) {
    const std::string arch = ModelDir(dir).arch();
    for (auto &kv : registry())
        if (kv.first == arch) return kv.second(dir, opt);
    std::string have;
    for (auto &a : supported_architectures()) have += " " + a;
    fail("unsupported model architecture '" + arch + "' (supported:" + have + ")");
}

}  // namespace axd
