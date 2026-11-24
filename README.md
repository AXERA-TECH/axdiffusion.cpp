# axdiffusion.cpp

在 AXERA AX8860 NPU 上运行文生图模型的 C++ 推理程序。文本编码器、DiT、VAE 全部在 NPU 上执行；
所有权重启动时一次性常驻 NPU 内存，之后每张图只做计算。提供命令行工具、内置网页，以及兼容
**OpenAI Images API** 与 **A1111 (SD WebUI) API** 的 HTTP 服务，支持**动态 LoRA**。

当前支持的模型：**MiniMax-H3-Image**（6 步蒸馏，无 CFG）。

![内置生图网页](docs/webui.jpg)

内置网页：896×1184，AX8860 上单张 6.8 秒。

## 性能（AX8860，4 核 NPU，常驻服务）

| 任务 | 单张耗时 | 其中 DiT | VAE 解码 |
|---|---|---|---|
| 文生图 512×512 | 2.7 s | 2.2 s | 0.36 s |
| 文生图 1024×1024 | 6.7 s | 5.9 s | 0.64 s |
| 文生图 1024×1024 + LoRA（已缓存） | 7.0 s | 6.2 s | 0.64 s |
| 文生图 2048×2048 | 27.4 s | 24.1 s | 3.1 s |
| 文生图 3840×2176 | 65.9 s | 60.0 s | 5.6 s |
| 文生图 4096×4096 | 167.4 s | 155.5 s | 11.5 s |
| 编辑 1024×1024，1 张参考图 | 22.1 s | 16.8 s | 0.63 s |
| 编辑 2048×2048，1 张参考图 | 49.2 s | 38.1 s | 3.2 s |

启动（全部权重载入 NPU 内存）约 23 s；常驻 NPU 内存约 31 GiB，使用编辑后约 34 GiB。

## 准备模型

下载 AX8860 版模型文件 `AXERA-TECH/MiniMax-H3-Image-AX8860`（含编译好的 AXModel 与转换后的权重），
例如放在 `/data/MiniMax-H3-Image-AX8860`。模型遵循 MiniMax-H3-Image Community License，
使用前请阅读模型仓库中的 `LICENSE`。

## 编译

板端编译：

```bash
git clone https://github.com/AXERA-TECH/axdiffusion.cpp.git
cd axdiffusion.cpp
cmake -S . -B build
cmake --build build -j8
```

在 x86 主机上交叉编译（需要 Arm GNU 工具链 13.2 / 13.3 和板上 `/usr/local/axhelix` 的拷贝）：

```bash
cmake -S . -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
      -DTOOLCHAIN_ROOT=/path/to/arm-gnu-toolchain -DAXCL_ROOT=/path/to/axhelix
cmake --build build-aarch64 -j
```

产物是 `axdiffusion-cli` 和 `axdiffusion-server`。环境准备、部署、选项和常见问题见 [编译指南](docs/build.md)。

## 命令行

```bash
./build/axdiffusion-cli --model /data/MiniMax-H3-Image-AX8860 \
    --prompt "A photograph of an orange tabby cat sitting on a wooden windowsill" \
    --size 1024 --seed 42 --out cat.png
```

| 选项 | 说明 |
|---|---|
| `--size N` / `--size WxH` | 输出尺寸，宽高为 32 的倍数、不小于 256，最大约 1670 万像素（如 4096×4096） |
| `--image FILE` | 参考图编辑，可重复指定，最多 3 张；参考图任意比例（1:1 到 3:1） |
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

浏览器打开 `http://<板子地址>:18500/` 就是内置的生图网页。网页源文件是 `web/index.html`，编译时内嵌进程序。
想改网页又不想重新编译，启动时加 `--web-dir <目录>`：服务每次请求都从 `<目录>/index.html` 读取页面，
改完刷新浏览器即可生效，目录里的其他文件挂在 `/web/` 下。

| API | 路径 |
|---|---|
| OpenAI Images | `POST /v1/images/generations`，`GET /v1/models` |
| A1111 / SD WebUI | `POST /sdapi/v1/txt2img`，`GET /sdapi/v1/sd-models`、`/sdapi/v1/loras`、`/sdapi/v1/options`、`/sdapi/v1/samplers` |
| 网页 | `GET /`：内置生图页面（提示词、尺寸、种子、LoRA 叠加、参考图、历史与下载） |
| 异步任务 | `POST /v1/jobs`、`GET /v1/jobs/{id}`（进度）、`GET /v1/jobs/{id}/events`（SSE）、`DELETE /v1/jobs/{id}`（取消） |
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
同一时刻只在 NPU 上生成一张图，并发请求排队；收到 SIGTERM/SIGINT 时等当前任务完成再退出。完整说明见 [docs/api.md](docs/api.md)，
Python 调用示例见 [examples/](examples/)。

## LoRA

- 把 `.safetensors` 格式的 LoRA 放进 `--lora-dir` 目录，按文件名（不含扩展名）引用。
- 支持 diffusers / PEFT（`lora_A`、`lora_B`）与 ComfyUI / kohya（`lora_down`、`lora_up`、`alpha`）两种格式，
  作用于 DiT 50 个块和 token refiner 的 qkv / out / fc1 / fc2。
- 多个 LoRA 可叠加（按 rank 拼接），每个请求可单独设置强度。
- 底模常驻不动，LoRA 在块 kernel 的低秩支路上执行；某个 LoRA 组合第一次使用时构建并缓存（约 5 s），
  之后切换只需重新绑定地址。
- 文件中出现不支持的层或格式（如 DoRA、LoKr、文本编码器 LoRA）时直接报错并列出，不会静默忽略。
- 文生图和参考图编辑的所有分辨率都支持 LoRA。块 kernel 有 rank 64 / 128 两档：按组合后最大的投影 rank 自动选最小够用的一档（128 档 DiT 约慢 8%）；超过 128 时自动用截断 SVD 压到 128（日志会给出保留的能量比例）。

## 参考图编辑

```bash
./build/axdiffusion-cli --model /data/MiniMax-H3-Image-AX8860 \
    --image cat.png --prompt "Make the cat black and add a red scarf" --size 1024 --out edit.png
```

- 1–3 张参考图（PNG / JPEG），比例 1:1 到 3:1；输出尺寸与文生图一样自由指定，最大 4096×4096。
- 编辑用的 kernel 在第一次编辑时加载（约 3 s）。每张参考图增加约 1 s 视觉编码和 3 s VAE 编码（1:1 时）。
- HTTP 接口里在请求中带 `images` 即为编辑，网页上直接上传参考图；见 [docs/api.md](docs/api.md)。

## 当前限制

- 分辨率与提示词长度共同决定序列长度：文生图为“提示词 token + 宽/32 × 高/32”，上限 16896；
  编辑再加上参考图的 token，上限 20480。超出时报错并给出可用范围。
- NPU 上同一输入多次运行的结果在细节上可能有微小差异，不影响整体画面。
- 不含提示词扩写模型；需要扩写时请在外部完成后再传入。

## 代码结构

分层设计见 [docs/architecture.md](docs/architecture.md)。

```
src/runtime/                NPU 运行时封装（设备内存、权重常驻、按名字绑定 IO 地址）、safetensors 读取
src/tokenizer/              Qwen BPE 分词（读取 tokenizer.json）
src/models/minimax_h3_image/  文本编码器、DiT、VAE、LoRA、整体流程
tools/                      axdiffusion-cli、axdiffusion-server
third_party/                nlohmann/json、cpp-httplib、stb_image_write、llama.cpp unicode（见 LICENSES.md）
```

## 许可

代码采用 MIT 许可证（见 `LICENSE`）。模型权重另受其各自许可证约束。
