# Custom Open WebUI fork for video inference

A personal fork of [Open WebUI](https://github.com/open-webui/open-webui) that tracks upstream `main`. It adds video
input for a local LLM server on Apple Silicon, today the owner's [mlx-vlm fork](https://github.com/Dev-Jahn/mlx-vlm),
and a switch for mlx-vlm's prefill offload, which hands the prefill of a long prompt to another GPU machine, the
prefill offload worker.
Everything else is upstream Open WebUI; see the upstream repository and <https://docs.openwebui.com> for the rest.

It is for personal use, and no upstream pull requests are planned. Fork code lives in its own files, and upstream
files carry only a few hook lines, so upstream merges stay simple. `CLAUDE.md` lists every hook.

## Models and servers

- **Served today:** Qwen3.8-Flash-Next-Uncensored-MLX (architecture `qwen4_exp`, Qwen3-VL processor) on mlx-vlm,
  port 8199.
- **Video input** works with any OpenAI-compatible server that accepts the `video_frames` content part and lists
  `video_input` on its `/v1/models` entries. mlx-vlm marks it supported for every model whose processor takes
  videos. The frame math (Qwen `smart_resize`, 2-frame groups, token estimate) is Qwen-VL's, so models on the
  Qwen3-VL processor get the whole feature: Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE and Qwen4-Exp. Per-frame
  timestamps and a per-request pixel budget are sent only when the model's `video_input` says it uses them.
- **Prefill offload** exists in mlx-vlm for Qwen4-Exp only. Its switch shows only for models that list
  `prefill_offload.auto: true`.
- **Other models** behave like upstream, except that a dropped video is refused ("Selected model(s) do not support
  video inputs") instead of being uploaded as an ordinary file.

The server contract is documented in mlx-vlm's `mlx_vlm/server/video_input.py`: a `video_frames` part carries
`frames` (file paths or data URIs), `fps`, `duration` and optional `timestamps`; the request field
`video_pixels.max_pixels` sets the pixel budget; `video` (local path) and `video_url` carry whole files.

## What differs from upstream

**Video input**

- The browser samples the video itself, the Qwen standard way: the frame count is `duration × fps`, clamped between
  the model's minimum and your max frames and rounded down to an even number, and the frames are spaced evenly from
  the first to the last one. Each frame's real time comes from the decoder (`requestVideoFrameCallback`), and
  repeated frames are dropped.
- Frames are resized with Qwen `smart_resize` under a per-frame token cap and JPEG-encoded. The attachment tile shows
  `N frames · W×H · ≈ tokens` before you send.
- The frames are uploaded as one **frame bundle** (`uploads/video_frames/<id>/`, owner-checked). The backend turns it
  into a `video_frames` part and sets `video_pixels.max_pixels` so the server's own resize changes nothing. Sent
  bundles never change, so every later turn replays byte-identical media and hits mlx-vlm's prefix cache.
- User messages put media first (images and videos in attachment order), then text, as Qwen expects. This also
  applies after upstream's `<attached_files>` block.
- Clicking the video poster in a sent message opens a flipbook player of the exact frames the model received.
- Per-model **Video Input** settings in the chat Controls pane: mode (Sampled frames or Original file), max frames,
  sampling fps and tokens per frame. They start from the server's `video_input` values; changing them re-extracts
  videos that are not sent yet.
- **Original file** mode uploads the video itself and sends `{"type": "video", "video": <path>}`, so the server
  samples it.

**Prefill offload**

- An **Offload long prompts** switch per model and user (Controls pane, Prefill Offload section, on by default). On
  lets mlx-vlm offload when the uncached prompt is longer than the break-even length and the worker accepts; off
  keeps the prefill local.
- Status lines on the reply show where the prefill runs and its progress ("Prefill offloading",
  "Prefill (offload worker): 41%", "Prefill (local): 12%"), and end with "Prefill offloaded · N s" or
  "Prefill done locally · N s". Only that last line is saved with the reply.
- Notices such as a busy or unreachable offload worker also appear as a warning toast. The unreachable one adds how
  to start the worker: "To use prefill offload: run prefill-worker start on the offload worker, or mlx-vlm-server
  start --offload."
- Admins get a **Measure** button that runs mlx-vlm's break-even benchmark (1-2 minutes) and applies the result.

**Other**

- While mlx-vlm is not running, a failed chat and an empty model list show "mlx-vlm server is not running. Start
  it: mlx-vlm-server start --offload (or without --offload for local prefill only)." instead of upstream's
  "Model not found" or connection errors.
- A Docker build and compose setup for the permanent service, under `scripts/fork/`.

## Running it

The permanent setup is a Docker image built from this repo ([scripts/fork/README.md](scripts/fork/README.md)):

```bash
scripts/fork/docker-build.sh                        # open-webui-fork:local; refuses uncommitted changes
mkdir -p ~/.local/share/open-webui-fork             # first time only
docker compose -f scripts/fork/compose.yaml up -d   # container open-webui-fork on http://localhost:8080
```

- All state (database, uploads, frame bundles, session secret) lives in `~/.local/share/open-webui-fork`.
- mlx-vlm runs natively on the host, listening on all interfaces at port 8199:
  `mlx_vlm.server --model ~/models/<model> --host 0.0.0.0 --port 8199`. The container reaches it as
  `host.docker.internal:8199`, and mlx-vlm reads the frame files from the host path (`VIDEO_SHARED_PATH`).
- The hints in the UI name `mlx-vlm-server start [--offload]`, the owner's start script (not in this repo;
  `--offload` also starts the offload worker). Set `MLX_VLM_OFFLINE_HINT` and `PREFILL_WORKER_UNREACHABLE_HINT` to
  match another setup.
- The connection URL and `TASK_MODEL_PARAMS` in `compose.yaml` only seed the database on the first start. Change them
  later in Admin Panel > Settings.

For development, the usual native setup works:

```bash
npm install && npm run dev                                   # frontend on :5173
uv sync --frozen                                             # backend venv (Python 3.12)
(cd backend && ../.venv/bin/uvicorn open_webui.main:app --port 8080 --host 127.0.0.1)
NODE_OPTIONS=--max-old-space-size=8192 npm run build         # production build, served by the backend
npx vitest run src/lib/utils/video                           # fork tests
.venv/bin/python -m pytest -q backend/open_webui/test/test_{video,prefill,prefill_router,mlx_vlm_offline}.py
```

Open `http://localhost:8080`, not `127.0.0.1`: replies stream over Socket.IO, which rejects other origins.

| Environment variable | Meaning |
|---|---|
| `VIDEO_FRAMES_TRANSPORT` | `path` (default: the server reads frame files) or `data_uri` (frames inline) |
| `VIDEO_SHARED_PATH` | The uploads directory as the LLM server sees it (for Docker) |
| `PREFILL_WORKER_UNREACHABLE_HINT` | Text added when the offload worker is unreachable; empty turns it off |
| `MLX_VLM_OFFLINE_HINT` | The message shown while mlx-vlm is not running |

## Known limitations

- Temporary chats refuse video: their messages never pass through the backend step that adds the frames.
- Firefox is refused in Sampled frames mode because it reports the requested seek time, not the frame's time.
  Use Chrome or Safari, or Original file mode.
- One chat cannot mix Sampled frames and Original file modes (one pixel budget per request).
- Presets use their base model's `video_input` only when `base_model_id` matches exactly. Connections with an explicit
  model list do not carry `video_input`.
- Multi-turn chats with a video need mlx-vlm commit 74324f05 or later.
- Frame bundles are not deleted with their chat; remove `uploads/video_frames/` entries by hand if needed.

## License

This fork remains under the upstream [Open WebUI License](LICENSE), including its branding terms. See also
`LICENSE_HISTORY` and `LICENSE_NOTICE`.
