#include "qwen_vision.hpp"

#include "common.hpp"

namespace axd {

namespace {
constexpr int kVisionPatchDim = 1536;  // 3 * 2 (temporal) * 16 * 16
constexpr int kVisionOutDim = 5120;
}  // namespace

VisionEncoder::VisionEncoder(const ModelDir &dir) {
    auto stages = dir.tiers("vision-vit", "stage ");
    auto shapes = dir.tier_shapes("vision-vit", "stage ");
    if (stages.empty()) fail("model directory has no vision-vit stages");
    int k = 0;
    for (auto &kv : stages)
        if (kv.first != k++) fail("vision-vit stages must be numbered 0..N-1");
    for (auto &kv : stages) stages_.push_back(std::make_unique<Model>(kv.second));
    grid_ = ModelDir::shape_int(shapes.begin()->second, "grid ", 0);
    if (grid_ <= 0 || grid_ % 2) fail("vision-vit shape must state an even 'grid N'");
    patches_ = grid_ * grid_;
    tokens_ = patches_ / 4;
    if (stages_.front()->input_names().size() != 1) fail("vision stage 0 must have one input");
    size_t extras = 0;
    for (auto &s : stages_) extras += s->output_names().size() - 1;
    if (extras != 4) fail("vision-vit stages must emit deepstack0, deepstack1, merger, deepstack2");
    // hidden width from stage 0's output size is not exposed directly; Qwen3-VL-32B ViT width is 1152
    hidden_ = 1152;
    pixels_ = DeviceBuffer(size_t(patches_) * kVisionPatchDim * 2, "vit-pixels");
    for (auto &h : h_) h = DeviceBuffer(size_t(patches_) * hidden_ * 2, "vit-hidden");
    for (auto &e : extra_) e = DeviceBuffer(size_t(tokens_) * kVisionOutDim * 2, "vit-out");
}

VisionEncoder::~VisionEncoder() = default;

VisionFeatures VisionEncoder::encode(const std::vector<float> &pixel_values, int grid_h, int grid_w) {
    if (grid_h != grid_ || grid_w != grid_)
        fail("the vision encoder is compiled for a " + std::to_string(grid_ * 16) + "x" + std::to_string(grid_ * 16) +
             " reference copy; got " + std::to_string(grid_w * 16) + "x" + std::to_string(grid_h * 16) +
             " (use square reference images)");
    if (pixel_values.size() != size_t(patches_) * kVisionPatchDim) fail("vision pixel_values size mismatch");
    std::vector<uint16_t> px(pixel_values.size());
    for (size_t i = 0; i < px.size(); ++i) px[i] = f32_to_bf16(pixel_values[i]);
    pixels_.buf().upload(px.data(), px.size() * 2);
    if (!stream_) stream_ = std::make_unique<Stream>();
    // extra outputs in stage order: deepstack0, deepstack1, merger, deepstack2
    int next_extra = 0, cur = 0;
    Buf in = pixels_.buf();
    for (size_t s = 0; s < stages_.size(); ++s) {
        auto ins = stages_[s]->input_names(), outs = stages_[s]->output_names();
        Bindings out = {{outs[0], h_[cur].buf()}};
        for (size_t o = 1; o < outs.size(); ++o) out[outs[o]] = extra_[next_extra++].buf();
        stages_[s]->submit(*stream_, {{ins[0], in}}, out);
        in = h_[cur].buf();
        cur = 1 - cur;
    }
    stream_->sync();
    VisionFeatures f;
    f.tokens = tokens_;
    auto get = [&](int i) {
        std::vector<uint16_t> v(size_t(tokens_) * kVisionOutDim);
        extra_[i].buf().download(v.data(), v.size() * 2);
        return v;
    };
    f.deepstack[0] = get(0);
    f.deepstack[1] = get(1);
    f.merger = get(2);
    f.deepstack[2] = get(3);
    return f;
}

}  // namespace axd
