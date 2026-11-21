#include "model_dir.hpp"

#include <sys/stat.h>

#include <cctype>
#include <fstream>

#include "common.hpp"
#include "json.hpp"

namespace axd {

bool ModelDir::exists(const std::string &rel) const {
    struct stat st {};
    return stat(file(rel).c_str(), &st) == 0;
}

ModelDir::ModelDir(const std::string &root) : root_(root) {
    std::ifstream f(root + "/axmodels/manifest.json");
    if (!f) fail("not a MiniMax-H3-Image-AX8860 directory (axmodels/manifest.json missing): " + root);
    auto j = nlohmann::json::parse(f);
    for (auto &m : j.at("models"))
        models_.push_back({m.at("purpose").get<std::string>(), m.at("shape").get<std::string>(),
                           m.at("file").get<std::string>()});
    std::ifstream d(root + "/axdiffusion.json");
    if (d) {
        arch_ = nlohmann::json::parse(d).at("arch").get<std::string>();
    } else if (exists("dit/shared.safetensors") && exists("vae/vae_bf16.safetensors")) {
        arch_ = "minimax-h3-image";  // release layout from before axdiffusion.json existed
    } else {
        fail("cannot tell the model architecture: " + root + "/axdiffusion.json is missing");
    }
}

std::string ModelDir::axmodel(const std::string &purpose, const std::string &shape_prefix) const {
    std::vector<const Entry *> hits, narrowed;
    for (auto &m : models_)
        if (m.purpose == purpose) hits.push_back(&m);
    if (!shape_prefix.empty()) {
        for (auto *m : hits)
            if (m->shape.rfind(shape_prefix, 0) == 0) narrowed.push_back(m);
        if (!narrowed.empty()) hits = narrowed;
    }
    if (hits.size() != 1) fail("expected one AXModel for " + purpose + " / " + shape_prefix);
    return root_ + "/axmodels/" + hits[0]->file;
}

bool ModelDir::has_axmodel(const std::string &purpose, const std::string &shape_prefix) const {
    for (auto &m : models_)
        if (m.purpose == purpose && m.shape.rfind(shape_prefix, 0) == 0) return true;
    return false;
}

namespace {
// first integer after `tag` in `shape` (tag "" = first integer anywhere), or -1
int int_after(const std::string &shape, const std::string &tag) {
    size_t p = tag.empty() ? 0 : shape.find(tag);
    if (p == std::string::npos) return -1;
    p += tag.size();
    while (p < shape.size() && !std::isdigit(static_cast<unsigned char>(shape[p]))) ++p;
    if (p >= shape.size()) return -1;
    return std::stoi(shape.substr(p));
}
}  // namespace

std::map<int, std::string> ModelDir::tiers(const std::string &purpose, const std::string &tag) const {
    std::map<int, std::string> out;
    for (auto &m : models_) {
        if (m.purpose != purpose) continue;
        int n = int_after(m.shape, tag);
        if (n >= 0) out[n] = root_ + "/axmodels/" + m.file;
    }
    return out;
}

std::map<int, std::string> ModelDir::tier_shapes(const std::string &purpose, const std::string &tag) const {
    std::map<int, std::string> out;
    for (auto &m : models_) {
        if (m.purpose != purpose) continue;
        int n = int_after(m.shape, tag);
        if (n >= 0) out[n] = m.shape;
    }
    return out;
}

int ModelDir::shape_int(const std::string &shape, const std::string &key, int fallback) {
    if (shape.find(key) == std::string::npos) return fallback;
    int n = int_after(shape, key);
    return n >= 0 ? n : fallback;
}


}  // namespace axd
