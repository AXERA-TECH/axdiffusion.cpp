# axdiffusion.cpp

在 AXERA AX8860 NPU 上运行文生图模型的 C++ 推理程序。文本编码器、DiT、VAE 全部在 NPU 上执行；
所有权重启动时一次性常驻 NPU 内存，之后每张图只做计算。提供命令行工具、内置网页，以及兼容
**OpenAI Images API** 与 **A1111 (SD WebUI) API** 的 HTTP 服务，支持**动态 LoRA**。

当前支持的模型：**MiniMax-H3-Image**（6 步蒸馏，无 CFG）。

## 性能（AX8860，4 核 NPU，常驻服务）

| 分辨率 | 单张耗时 | 其中 DiT | 文本编码 | VAE |
|---|---|---|---|---|
| 1024×1024 | 7.3 s | 5.6 s | 0.17 s | 0.64 s |
| 1024×1024 + LoRA（已缓存） | 7.8 s | 6.0 s | 0.17 s | 0.64 s |
| 2048×2048 | 28.2 s | 24.5 s | 0.17 s | 3.5 s |

启动（全部权重载入 NPU 内存）约 55 s；常驻 NPU 内存约 28 GiB。

## 准备模型

下载 AX8860 版模型文件 `AXERA-TECH/MiniMax-H3-Image-AX8860`（含编译好的 AXModel 与转换后的权重），
例如放在 `/data/MiniMax-H3-Image-AX8860`。模型遵循 MiniMax-H3-Image Community License，
使用前请阅读模型仓库中的 `LICENSE`。

## 编译（在板上）

需要 g++ ≥ 9、CMake ≥ 3.16，以及板上自带的 AXCL 运行时（默认 `/usr/local/axhelix`）。

```bash
git clone https://github.com/AXERA-TECH/axdiffusion.cpp.git
cd axdiffusion.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release     # 运行时不在默认位置时加 -DAXCL_ROOT=/path/to/axcl
make -C build -j8
```

产物：`build/axdiffusion-cli`、`build/axdiffusion-server`。

## 命令行

```bash
./build/axdiffusion-cli --model /data/MiniMax-H3-Image-AX8860 \
    --prompt "A photograph of an orange tabby cat sitting on a wooden windowsill" \
    --size 1024 --seed 42 --out cat.png
```

| 选项 | 说明 |
|---|---|
| `--size 1024\|2048` | 输出 1024×1024 或 2048×2048 |
| `--seed N` | 随机种子（与 PyTorch `torch.randn` 的 CPU 结果一致） |
| `--lora-dir DIR --lora NAME[:SCALE]` | 使用 LoRA，可重复指定多个 |
| `--repeat N` | 同一进程连续出 N 张（看稳态耗时） |
| `--quiet` | 关闭进度日志 |

每张图输出一行 JSON，包含各阶段耗时。

## HTTP 服务

```bash
./build/axdiffusion-server --model /data/MiniMax-H3-Image-AX8860 --port 18500 \
    --lora-dir /data/loras --out-dir ./output
```

| API | 路径 |
|---|---|
| OpenAI Images | `POST /v1/images/generations`，`GET /v1/models` |
| A1111 / SD WebUI | `POST /sdapi/v1/txt2img`，`GET /sdapi/v1/sd-models`、`/sdapi/v1/loras`、`/sdapi/v1/options`、`/sdapi/v1/samplers` |
| 网页 | `GET /`：内置生图页面（提示词、尺寸、种子、LoRA 叠加、参考图、历史与下载） |
| 其它 | `GET /health`，`GET /v1/loras`，`GET /files/<name>.png` |

OpenAI 格式（`n` ≤ 4；`response_format` 为 `b64_json` 或 `url`）：

```bash
curl http://BOARD_IP:18500/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "minimax-h3-image",
  "prompt": "a lighthouse on a cliff at sunset",
  "size": "1024x1024",
  "seed": 42,
  "loras": [{"name": "my_style_lora", "scale": 0.8}]
}'
```

A1111 格式（LoRA 写在提示词里）：

```bash
curl http://BOARD_IP:18500/sdapi/v1/txt2img -H 'Content-Type: application/json' -d '{
  "prompt": "a lighthouse on a cliff at sunset <lora:my_style_lora:0.8>",
  "width": 1024, "height": 1024, "seed": -1
}'
```

`steps`、`cfg_scale`、`sampler_name`、`negative_prompt` 会被忽略（模型为 6 步蒸馏、无 CFG）。
同一时刻只在 NPU 上生成一张图，并发请求排队。完整说明见 [docs/api.md](docs/api.md)，
Python 调用示例见 [examples/](examples/)。

## LoRA

- 把 `.safetensors` 格式的 LoRA 放进 `--lora-dir` 目录，按文件名（不含扩展名）引用。
- 支持 diffusers / PEFT（`lora_A`、`lora_B`）与 ComfyUI / kohya（`lora_down`、`lora_up`、`alpha`）两种格式，
  作用于 DiT 50 个块和 token refiner 的 qkv / out / fc1 / fc2。
- 多个 LoRA 可叠加，总 rank（每个投影）不超过 64；每个请求可单独设置强度。
- 底模常驻不动，LoRA 在块 kernel 的低秩支路上执行；某个 LoRA 组合第一次使用时构建并缓存（约 5 s），
  之后切换只需重新绑定地址。
- 文件中出现不支持的层或格式（如 DoRA、LoKr、文本编码器 LoRA）时直接报错并列出，不会静默忽略。
- 1024×1024 与 2048×2048 都支持 LoRA。rank 超过 64 时自动用截断 SVD 压到 64（日志会给出保留的能量比例）。

## 当前限制

- 分辨率：1024×1024、2048×2048（其它宽高比开发中）。
- 提示词最长 256 token，超出部分截断。
- 不含提示词扩写模型；需要扩写时请在外部完成后再传入。

## 代码结构

```
src/runtime/                NPU 运行时封装（设备内存、权重常驻、按名字绑定 IO 地址）、safetensors 读取
src/tokenizer/              Qwen BPE 分词（读取 tokenizer.json）
src/models/minimax_h3_image/  文本编码器、DiT、VAE、LoRA、整体流程
tools/                      axdiffusion-cli、axdiffusion-server
third_party/                nlohmann/json、cpp-httplib、stb_image_write、llama.cpp unicode（见 LICENSES.md）
```

## 许可

代码采用 MIT 许可证（见 `LICENSE`）。模型权重另受其各自许可证约束。
