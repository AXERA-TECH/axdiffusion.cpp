// A model directory: its AXModel manifest (axmodels/manifest.json) and self description (axdiffusion.json).
#pragma once

#include <map>
#include <string>
#include <vector>

namespace axd {

class ModelDir {
public:
    explicit ModelDir(const std::string &root);
    // Architecture id from axdiffusion.json ("minimax-h3-image", ...).
    const std::string &arch() const { return arch_; }
    std::string axmodel(const std::string &purpose, const std::string &shape_prefix = "") const;
    bool has_axmodel(const std::string &purpose, const std::string &shape_prefix = "") const;
    // Every AXModel of a purpose keyed by the first integer after `tag` in its shape string
    // (e.g. tag "padded " -> padded sequence length, tag "" with "-row" shapes -> row count).
    std::map<int, std::string> tiers(const std::string &purpose, const std::string &tag) const;
    // The shape strings of the same entries (for per-kernel attributes such as "granule N").
    std::map<int, std::string> tier_shapes(const std::string &purpose, const std::string &tag) const;
    // First integer after `key` in a shape string, or `fallback`.
    static int shape_int(const std::string &shape, const std::string &key, int fallback);
    std::string file(const std::string &rel) const { return root_ + "/" + rel; }
    bool exists(const std::string &rel) const;
    const std::string &root() const { return root_; }

private:
    struct Entry {
        std::string purpose, shape, file;
    };
    std::string root_, arch_;
    std::vector<Entry> models_;
};

}  // namespace axd
