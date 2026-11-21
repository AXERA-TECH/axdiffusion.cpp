# axdiffusion 架构

目标：一套 C++ 运行时在 AXERA NPU 上跑多种扩散模型（文生图、图像编辑、文生视频、图生视频、参考生视频），
新增模型只写"模型插件"，不改运行时、服务和协议层。

## 1. 分层

```
tools/                 axdiffusion-cli, axdiffusion-server            (入口)
src/server/            HTTP、任务队列、协议适配器 (OpenAI / A1111 / 视频 / 网页)
src/models/<name>/     模型插件：组件装配 + 模型特有逻辑 (token 布局、条件方式、调度参数)
src/components/        可复用组件：文本编码器、分词器、DiT 运行器、LoRA、VAE、调度器、媒体 IO
src/runtime/           与模型无关：NPU 设备/内存、AXModel 调用、权重读取、线程、日志
```

依赖只能自上而下：runtime 不知道组件，组件不知道具体模型，模型不知道 HTTP。

## 2. runtime

| 模块 | 职责 |
|---|---|
| `npu` | 设备与上下文（每线程绑定）、设备内存、`TensorBank`（多张量 256 B 对齐打包）、`Model`（按名字绑定 IO 地址后执行，零拷贝） |
| `safetensors` | mmap 读取，任意 dtype |
| `image_io` | 图片解码（EXIF 方向）/ PNG 编码 |
| `common` | 错误、计时、日志、`parallel_for`、bf16 转换 |

**驻留策略**（`Residency`）：组件声明自己的权重大小，可设为 `resident`（启动加载、常驻）、`on_demand`
（首次使用加载，LRU 释放）或 `stage`（只在所属阶段存在）。视频模型的 DiT + 文本编码器 + 视频 VAE 无法同时常驻
48 GiB 时，由插件声明阶段间释放顺序。

## 3. components

| 组件 | 现有实现 | 计划 |
|---|---|---|
| 分词器 | Qwen BPE（tokenizer.json，与 HF 逐 id 一致） | SentencePiece（T5 等） |
| 文本编码器 | Qwen3-VL 前 N 层，分页 KV（256/页），视觉 token + deepstack | T5-XXL、CLIP |
| 视觉编码器 | Qwen3-VL ViT（编辑参考图） | — |
| DiT 运行器 | 块循环、序列档位自动发现、多时间步 AdaLN、参考 token 冻结、LoRA 支路 | 其它 DiT（MMDiT 双流等）按块 ABI 描述接入 |
| LoRA | diffusers/PEFT、ComfyUI、kohya；rank 拼接、SVD 截断兜底、组合缓存 | 文本编码器 LoRA、DoRA |
| VAE | 图像 2D ViT 解码（256 px 分块） | 图像 VAE 编码、3D 视频 VAE（时间+空间分块）、音频 VAE |
| 调度器 | 6 步蒸馏固定 sigma + Euler（velocity） | flow-matching 标准步数、CFG、shift |
| 媒体 IO | PNG/JPEG 输入、PNG 输出 | MP4（板载硬件 H.264/H.265 编码）+ 音频封装 |

组件之间只通过设备缓冲区（`Buf`）和主机张量交换数据；每个组件有自己的 kernel 清单查询
（`ModelDir::tiers(purpose, tag)`），按请求形状选最小可用档位。

## 4. 模型插件

```cpp
struct ModelInfo { std::string arch; std::vector<Task> tasks; /* 支持的分辨率/帧数/档位 */ };

class ModelPlugin {
public:
    virtual ModelInfo info() const = 0;
    virtual Result run(const Request &req, Progress &progress) = 0;   // 在 NPU 线程上执行
};
// 注册：AXD_REGISTER_MODEL("minimax-h3-image", factory)
```

模型目录根下的 `axdiffusion.json` 描述自己：

```json
{ "arch": "minimax-h3-image", "version": 1,
  "components": { "text_encoder": "text_encoder/", "dit": "dit/", "vae": "vae/" },
  "tasks": ["txt2img", "edit"] }
```

`axdiffusion-server --model DIR` 读取 `arch` 选插件，未知架构直接报错。一个进程只加载一个模型
（NPU 内存有限）；多模型由多个进程/端口提供。

计划中的插件：`minimax-h3-image`（txt2img、edit）、`minimax-h3`（视频：t2va、i2va、ref2va、fl2va，带音频）。

## 5. 任务与请求

```cpp
enum class Task { txt2img, edit, txt2video, img2video, ref2video };

struct Request {
    Task task; std::string prompt, negative_prompt;
    int width, height, frames = 1; float fps = 0;
    uint64_t seed; int steps = 0; float cfg = 0;          // 0 = 模型默认
    std::vector<LoraSpec> loras;
    std::vector<Image> images;                            // 参考图 / 首尾帧
    std::vector<Video> videos; std::vector<Audio> audios; // 参考视频 / 音频
};
struct Result { std::vector<Image> images; Video video; Audio audio; json timings; };
```

`Progress` 回调报告阶段（编码 / 第 k 步 / 解码 / 封装）与百分比，服务层转发给客户端。

## 6. 服务层

- **任务队列**：单 NPU 工作线程串行执行；HTTP 线程只做解析和排队。
  - 同步接口（OpenAI Images、A1111）在请求内等待完成。
  - 异步接口（视频、长任务）：`POST` 返回任务号，`GET /v1/jobs/{id}` 轮询，`GET /v1/jobs/{id}/events` 以 SSE 推送进度，支持取消。
- **协议适配器**：每种协议一个文件，只做"协议 ↔ Request/Result"转换。
  - OpenAI Images：`/v1/images/generations`、`/v1/images/edits`
  - A1111：`/sdapi/v1/txt2img`、`/sdapi/v1/img2img`
  - 视频：`/v1/videos`（提交）、`/v1/videos/{id}`（状态/下载），字段对齐 OpenAI 视频接口
  - 网页：`/`
- **安全退出**：SIGTERM 时停止接收、等当前任务完成后释放 NPU，避免强杀导致 NPU 异常。

## 7. 目录落位（迁移目标）

```
src/runtime/        common, npu, safetensors, image_io, (video_io, audio_io)
src/components/     tokenizer/, text_qwen/, vision_qwen/, dit/, lora/, vae_image/, vae_video/, scheduler/
src/models/         minimax_h3_image/, minimax_h3/
src/server/         queue, api_openai, api_a1111, api_video, web_ui
tools/              axdiffusion_cli.cpp, axdiffusion_server.cpp
```
