# HTTP API

`axdiffusion-server` 同时提供 OpenAI Images 与 A1111 (SD WebUI) 两套接口。所有 JSON 错误返回
`{"error": {"message": "...", "type": "invalid_request_error"}}`，HTTP 状态码 400/404。

## OpenAI Images

### `POST /v1/images/generations`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | 必填 | 提示词（上限由模型目录的档位决定，当前 512 token；超出部分截断，`info.truncated` 为真） |
| `model` | string | — | 可省略；`GET /v1/models` 返回的 id |
| `n` | int | 1 | 1–4 |
| `size` | string | `1024x1024` | 正方形尺寸，可用值见 `GET /health` 的 `sizes`（由模型目录里的 DiT 档位决定） |
| `response_format` | string | `b64_json` | `b64_json` 或 `url` |
| `seed` | int | 随机 | 扩展字段；`n` > 1 时依次 +1 |
| `steps` | int | 6 | 扩展字段，1–6 |
| `refine_steps` | int | 0 | 扩展字段，大图加速：`1` 或 `2`，见下文；`0` 为直接出图 |
| `loras` | array | `[]` | 扩展字段，`[{"name": "...", "scale": 1.0}]` |

响应：

```json
{
  "created": 1795240000,
  "data": [{"b64_json": "iVBORw0...", "revised_prompt": "..."}],
  "info": [{"seed": 42, "size": "1024x1024", "width": 1024, "height": 1024, "tokens": 51, "loras": [], "lora_build_s": 0.0,
            "seconds": {"text_encoder": 0.17, "dit": 5.56, "vae": 0.64, "total": 6.38}}]
}
```

`response_format: "url"` 时 `data[i].url` 指向 `GET /files/<name>.png`。

### `GET /v1/models`

返回 `{"object": "list", "data": [{"id": "minimax-h3-image", ...}]}`。

### `GET /v1/loras`

返回 `{"data": [{"name": "..."}]}`，即 `--lora-dir` 中可用的 LoRA。

## A1111 / SD WebUI

### `POST /sdapi/v1/txt2img`

| 字段 | 说明 |
|---|---|
| `prompt` | 提示词；可含 `<lora:NAME:SCALE>` |
| `width`, `height` | 需相等，可用值见 `GET /health` 的 `sizes` |
| `seed` | `-1` 为随机 |
| `batch_size`, `n_iter` | 乘积 ≤ 4 |
| `steps`, `cfg_scale`, `sampler_name`, `negative_prompt` | 接受但忽略 |
| `enable_hr`, `hr_scale`, `hr_second_pass_steps` | 大图加速：`width`×`height` 是第一遍的尺寸，输出为它的 2 倍（`hr_scale` 只接受 2）；`hr_second_pass_steps` 取 1（默认）或 2 |

响应：`{"images": ["<base64 PNG>", ...], "parameters": {...}, "info": "<JSON 字符串>"}`。

其它：`GET /sdapi/v1/sd-models`、`/sdapi/v1/loras`、`/sdapi/v1/options`、`/sdapi/v1/samplers`。

## 异步任务（网页与长任务使用）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/jobs` | 提交：`{task, prompt, width, height, seed, steps, refine_steps, loras, images:[base64]}`，返回 202 与任务对象 |
| GET | `/v1/jobs/{id}` | 状态 `queued/running/succeeded/failed/cancelled`、阶段 `stage`、进度 `progress.step/total`；成功后带 `images`（base64 PNG）与 `info` |
| GET | `/v1/jobs/{id}/events` | Server-Sent Events：每次进度变化一条 `progress`，结束时一条 `done` |
| DELETE | `/v1/jobs/{id}` | 取消：排队中的直接移除，运行中的在下一步停止 |

`task` 取值：`txt2img`、`edit`（带 `images` 时自动为 edit），以及视频任务（`txt2video` 等，视频模型提供时可用）。
同一时刻只有一个任务在 NPU 上执行，其余排队；同步接口（OpenAI、A1111）内部也走同一个队列。

## 大图加速（`refine_steps`）

可选的两遍出图，只用于文生图，默认关闭。服务先按一半宽高（取 32 的倍数）出一张完整的图，放大到目标尺寸后
编码回潜空间并加噪，再在目标尺寸上只跑最后 1 步或 2 步。

- 输出需不小于 300 万像素（例如 2048×1536）；更小的尺寸直接出图更合适，会返回错误。
- 构图与小图一致，细节比直接出图少一些。补 2 步细节更多、耗时更长，人像皮肤可能多出斑点。
- AX8860 上 2048×2048：直接出图 27.3 s，补 1 步 19.8 s，补 2 步 23.8 s；4096×4096：167.4 s、88.6 s、114.4 s。
- `GET /health` 的 `refine` 给出可用的步数与最小像素数：`{"steps": [1, 2], "min_pixels": 3145728}`；模型目录
  没有 VAE 编码 kernel 时 `steps` 为空。
- 任务阶段依次为 `text_encoder`、`denoise`、`base_vae`、`refine_encode`、`refine`、`vae`；结果 `info` 带
  `refine_steps`，`info.seconds` 里 `base_*` 是第一遍各阶段耗时，`total` 为两遍合计。

## 其它

- `GET /health`：`{"status", "model", "arch", "tasks", "sizes", "lora", "refine", "max_references", "load_s", "requests", "queued", "cmm_used_gib"}`
- `GET /files/<name>.png`：`response_format=url` 生成的图片
