// Byte-level BPE tokenizer for the Qwen3-VL text encoder (reads tokenizer.json).
//
// Matches HF `tokenizers` for plain text: NFC normalization, the Qwen2
// pre-tokenizer split, GPT-2 byte-to-unicode mapping and ranked BPE merges.
// Special/added tokens written inside the prompt are not recognized (the
// text-to-image pipeline encodes the raw prompt without a chat template).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace axd {

class Tokenizer {
public:
    explicit Tokenizer(const std::string &tokenizer_json);
    std::vector<int32_t> encode(const std::string &text) const;
    size_t vocab_size() const { return vocab_.size(); }

private:
    std::vector<int32_t> bpe(const std::string &word) const;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank
};

}  // namespace axd
