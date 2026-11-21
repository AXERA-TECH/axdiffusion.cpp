# HTTP API

`axdiffusion-server` 同时提供 OpenAI Images 与 A1111 (SD WebUI) 两套接口。所有 JSON 错误返回
`{"error": {"message": "...", "type": "invalid_request_error"}}`，HTTP 状态码 400/404。

## OpenAI Images

### `POST /v1/images/generations`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | 必填 | 提示词（最长 256 token） |
| `model` | string | — | 可省略；`GET /v1/models` 返回的 id |
| `n` | int | 1 | 1–4 |
| `size` | string | `1024x1024` | `1024x1024` 或 `2048x2048` |
| `response_format` | string | `b64_json` | `b64_json` 或 `url` |
| `seed` | int | 随机 | 扩展字段；`n` > 1 时依次 +1 |
| `steps` | int | 6 | 扩展字段，1–6 |
| `loras` | array | `[]` | 扩展字段，`[{"name": "...", "scale": 1.0}]` |

响应：

```json
{
  "created": 1795240000,
  "data": [{"b64_json": "iVBORw0...", "revised_prompt": "..."}],
  "info": [{"seed": 42, "size": 1024, "tokens": 51, "loras": [], "lora_build_s": 0.0,
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
| `width`, `height` | 1024 或 2048（需相等） |
| `seed` | `-1` 为随机 |
| `batch_size`, `n_iter` | 乘积 ≤ 4 |
| `steps`, `cfg_scale`, `sampler_name`, `negative_prompt` | 接受但忽略 |

响应：`{"images": ["<base64 PNG>", ...], "parameters": {...}, "info": "<JSON 字符串>"}`。

其它：`GET /sdapi/v1/sd-models`、`/sdapi/v1/loras`、`/sdapi/v1/options`、`/sdapi/v1/samplers`。

## 异步任务（网页与长任务使用）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/jobs` | 提交：`{task, prompt, width, height, seed, steps, loras, images:[base64]}`，返回 202 与任务对象 |
| GET | `/v1/jobs/{id}` | 状态 `queued/running/succeeded/failed/cancelled`、阶段 `stage`、进度 `progress.step/total`；成功后带 `images`（base64 PNG）与 `info` |
| GET | `/v1/jobs/{id}/events` | Server-Sent Events：每次进度变化一条 `progress`，结束时一条 `done` |
| DELETE | `/v1/jobs/{id}` | 取消：排队中的直接移除，运行中的在下一步停止 |

`task` 取值：`txt2img`、`edit`（带 `images` 时自动为 edit），以及视频任务（`txt2video` 等，视频模型提供时可用）。
同一时刻只有一个任务在 NPU 上执行，其余排队；同步接口（OpenAI、A1111）内部也走同一个队列。

## 其它

- `GET /health`：`{"status", "model", "arch", "tasks", "sizes", "lora", "load_s", "requests", "queued", "cmm_used_gib"}`
- `GET /files/<name>.png`：`response_format=url` 生成的图片
