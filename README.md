# whisper-service

A small, persistent [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
transcription server. Loads the model into GPU memory once at startup and serves
transcription over HTTP, so you pay the ~5s python+torch+CUDA startup cost a
single time instead of on every call. Steady-state latency is just the inference
(~0.2–0.5s for a short clip on a 3090).

Single file, standard-library HTTP server, no web framework. Configured entirely
by environment variables.

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
| `WHISPER_MODEL` | `small` | faster-whisper model name or local path |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_COMPUTE` | `float16` | `float16` / `int8_float16` / `int8` |
| `WHISPER_PORT` | `8771` | listen port |
| `WHISPER_HOST` | `127.0.0.1` | bind address |

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
to auto-detect.

Response:

```json
{"text": "...", "language": "en", "duration": 12.3, "infer_ms": 287}
```

### `GET /health`

```json
{"ok": true, "model": "small", "device": "cuda"}
```

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

## License

MIT — see [LICENSE](LICENSE).
