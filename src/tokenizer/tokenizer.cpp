#include "tokenizer.hpp"

#include <climits>
#include <fstream>
#include <sstream>

#include "common.hpp"
#include "json.hpp"
#include "unicode.h"

namespace axd {

namespace {
// Qwen2 pre-tokenizer regex in the form llama.cpp's unicode splitter implements natively.
const char *kQwen2Regex =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::vector<std::string> utf8_chars(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t n = unicode_len_utf8(s[i]);
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}
}  // namespace

Tokenizer::Tokenizer(const std::string &path) {
    std::ifstream f(path);
    if (!f) fail("cannot open tokenizer " + path);
    nlohmann::json j = nlohmann::json::parse(f);
    auto &model = j.at("model");
    if (model.at("type").get<std::string>() != "BPE") fail("tokenizer model is not BPE");
    for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it)
        vocab_.emplace(it.key(), it.value().get<int32_t>());
    int32_t rank = 0;
    for (auto &m : model.at("merges")) {
        std::string key;
        if (m.is_string()) {
            key = m.get<std::string>();
        } else {  // newer format: ["left", "right"]
            key = m[0].get<std::string>() + " " + m[1].get<std::string>();
        }
        merge_rank_.emplace(std::move(key), rank++);
    }
}

std::vector<int32_t> Tokenizer::bpe(const std::string &word) const {
    auto whole = vocab_.find(word);
    if (whole != vocab_.end()) return {whole->second};
    std::vector<std::string> parts = utf8_chars(word);
    while (parts.size() > 1) {
        int best = INT_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            auto it = merge_rank_.find(parts[i] + " " + parts[i + 1]);
            if (it != merge_rank_.end() && it->second < best) {
                best = it->second;
                best_i = i;
            }
        }
        if (best == INT_MAX) break;
        parts[best_i] += parts[best_i + 1];
        parts.erase(parts.begin() + long(best_i) + 1);
    }
    std::vector<int32_t> ids;
    for (auto &p : parts) {
        auto it = vocab_.find(p);
        if (it == vocab_.end()) fail("tokenizer: piece not in vocabulary");
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string &text) const {
    // NFC normalization (tokenizer.json normalizer).
    auto cpts = unicode_cpts_normalize_nfc(unicode_cpts_from_utf8(text));
    std::string normalized;
    for (auto c : cpts) normalized += unicode_cpt_to_utf8(c);
    std::vector<int32_t> ids;
    for (auto &word : unicode_regex_split(normalized, {kQwen2Regex}, /*byte_encode=*/true)) {
        auto w = bpe(word);
        ids.insert(ids.end(), w.begin(), w.end());
    }
    return ids;
}

}  // namespace axd
