#include "h3_image_vae_encoder.hpp"

#include "common.hpp"

namespace axd {

namespace {
constexpr size_t kTileIn = size_t(3) * kVaeEncTile * kVaeEncTile * 4;
constexpr size_t kTileOut = size_t(kVaeEncMoments) * kVaeEncTileLatent * kVaeEncTileLatent * 4;
}  // namespace

VaeEncoder::VaeEncoder(const ModelDir &dir) {
    model_ = std::make_unique<Model>(dir.axmodel("image-vae-encode"));
    auto ins = model_->input_names(), outs = model_->output_names();
    if (ins.size() != 1 || outs.size() != 1) fail("image-vae-encode must have one input and one output");
    in_name_ = ins[0];
    out_name_ = outs[0];
}

VaeEncoder::~VaeEncoder() = default;

std::vector<std::vector<float>> VaeEncoder::encode_tiles(const std::vector<std::vector<float>> &tiles) {
    const size_t n = tiles.size();
    if (n > capacity_) {  // grows to the largest request seen (25 tiles per 1024x1024 reference)
        in_ = DeviceBuffer(kTileIn * n, "vae-enc-in");
        out_ = DeviceBuffer(kTileOut * n, "vae-enc-out");
        capacity_ = n;
    }
    if (!stream_) stream_ = std::make_unique<Stream>();
    for (size_t i = 0; i < n; ++i) {
        if (tiles[i].size() * 4 != kTileIn) fail("VAE encoder tile must be [3,256,256]");
        Buf in = in_.buf().view(i * kTileIn, kTileIn);
        in.upload(tiles[i].data(), kTileIn);
        model_->submit(*stream_, {{in_name_, in}}, {{out_name_, out_.buf().view(i * kTileOut, kTileOut)}});
    }
    stream_->sync();
    std::vector<std::vector<float>> out(n, std::vector<float>(kTileOut / 4));
    for (size_t i = 0; i < n; ++i) out_.buf().view(i * kTileOut, kTileOut).download(out[i].data(), kTileOut);
    return out;
}

}  // namespace axd
