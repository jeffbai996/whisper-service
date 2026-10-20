# whisper-service

A small, persistent [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
transcription server. Loads the model into GPU memory once (at startup, or on the
first request with `WHISPER_LAZY_LOAD=1`) and serves transcription over HTTP, so
you pay the ~5s python+torch+CUDA startup cost a
single time instead of on every call. Steady-state latency is just the inference
(~0.2–0.5s for a short clip on a 3090).

`server.py` uses the standard-library HTTP server and no web framework.
`router.py` is a second, model-free script described under [Router](#router).
Both are configured by environment variables.

## Why

One-shot `whisper`/`faster-whisper` CLI invocations re-import torch and re-load
the model every time, which dominates wall-clock for short clips (voice messages,
notes). Keeping the model resident drops that to near-zero.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` (the server shells out to it to normalize any input
to 16 kHz mono WAV before inference). A static build works fine.

## Run

```bash
python3 server.py
```

### Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name or local path |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_COMPUTE` | `float16` | `float16` / `int8_float16` / `int8` |
| `WHISPER_PORT` | `8771` | listen port |
| `WHISPER_HOST` | `127.0.0.1` | bind address |
| `WHISPER_OPENAI_HOST` | `172.17.0.1` | second listener with the same routes; set it empty to disable |
| `WHISPER_OPENAI_PORT` | `8899` | port of the second listener |
| `WHISPER_LAZY_LOAD` | unset | `1` loads the model on the first request instead of at startup |
| `WHISPER_IDLE_UNLOAD_S` | `0` | unload the model after this many idle seconds; `0` never unloads |
| `WHISPER_MIN_FREE_MB` | `0` | answer 503 instead of loading when less VRAM than this is free; `0` skips the check |
| `WHISPER_NVIDIA_SMI` | `nvidia-smi` | nvidia-smi binary used for the free-VRAM check |
| `WHISPER_SCRATCH` | `/mnt/wsl-storage/scratch/whisper` | directory for uploads and ffmpeg output; set it to a disk with room |

By default the server also listens on `172.17.0.1:8899`, the Docker bridge
address, so containers on the host can reach it. If that bind fails it logs the
error and serves on `WHISPER_HOST` only.

On a machine with no system CUDA toolkit, point the linker at the nvidia pip
wheels installed into the venv (ctranslate2 dlopens `libcublas` / `libcudnn` at
inference time):

```bash
export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
```

## API

### `POST /transcribe`

Three ways to send audio:

```bash
# 1. multipart file upload
curl -s -X POST http://127.0.0.1:8771/transcribe -F "file=@clip.ogg"

# 2. JSON with a server-local path (fastest — no upload)
curl -s -X POST http://127.0.0.1:8771/transcribe \
  -H "Content-Type: application/json" -d '{"path":"/abs/path/clip.ogg"}'

# 3. raw audio bytes in the body (optional ?language=xx)
curl -s -X POST "http://127.0.0.1:8771/transcribe?language=en" \
  --data-binary @clip.ogg
```

Optional `language` (form field / JSON key / query param); omit or pass `auto`
to auto-detect. All three forms accept `segments` (return timed segments),
`words` (add word times to each segment) and `vad` (speech filter, on unless
`multilingual`). JSON and multipart requests also accept `initial_prompt` and
`beam_size` (default 5), and a JSON request can set `multilingual`.

Response:

```json
{"text": "...", "language": "en", "duration": 12.3, "infer_ms": 287}
```

With `segments`, the response also has
`"segments": [{"start": 0.0, "end": 2.1, "text": "..."}]`.

### `POST /v1/audio/transcriptions`

OpenAI-compatible. Multipart form with `file` (or `audio`), `model`, `language`,
`prompt` and `response_format` (`json`, `text` or `verbose_json`).

### `GET /v1/models`

OpenAI-style model list with the configured model.

### `GET /health`

```json
{"ok": true, "model": "large-v3-turbo", "device": "cuda", "loaded": true}
```

`loaded` is false while the model is not in memory (lazy load or idle unload).

## Run as a systemd user service (optional)

`~/.config/systemd/user/whisper-server.service`:

```ini
[Unit]
Description=faster-whisper resident transcription server

[Service]
Type=simple
WorkingDirectory=%h/whisper-service
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=%h/whisper-service/venv/lib/python3.12/site-packages/nvidia/cublas/lib:%h/whisper-service/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:%h/whisper-service/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
Environment=WHISPER_MODEL=small
Environment=WHISPER_DEVICE=cuda
Environment=WHISPER_COMPUTE=float16
ExecStart=%h/whisper-service/venv/bin/python %h/whisper-service/server.py
# A wedged GPU model reload should not respawn in a loop and pin VRAM:
Restart=no

[Install]
WantedBy=default.target
```

```bash
systemctl --user start whisper-server
```

## Tests

```bash
python3 -m pytest tests
```

The tests inject a fake model, so they need no GPU. `pytest` is in
`requirements.txt`. The upload tests need `ffmpeg` on `PATH` and a writable
`WHISPER_SCRATCH`, for example `WHISPER_SCRATCH=$(mktemp -d) python3 -m pytest tests`.

## License

MIT — see [LICENSE](LICENSE).

## Router

`router.py` serves the same routes without a model, for a host whose GPU should
not carry Whisper. It forwards each request to the first server in
`WHISPER_UPSTREAMS` (comma-separated, priority order) that answers, skipping one
that is unreachable or answers 5xx. JSON `{"path": ...}` requests are uploaded
(large files re-encoded to 16 kHz mono Opus first); `multilingual` has no upload
form and is dropped. It listens where `server.py` did, so callers keep their URLs.
Run it with `python3 router.py`. `WHISPER_ROUTER_TIMEOUT_S` (default 900) bounds
each forwarded request.

With `WHISPER_LOCAL_URL` and `WHISPER_LOCAL_START` set, a local server is the
last resort: when every upstream fails, the router runs the start command, waits
up to `WHISPER_LOCAL_WAIT_S` for its health, and forwards there. Health reports
it as `on_demand` without starting it.
