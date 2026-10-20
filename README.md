# whisper-service

Resident faster-whisper transcription server (`server.py`, standard-library HTTP,
no framework) and a model-free forwarding router (`router.py`). Both are
configured by environment variables.

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` must be on `PATH`; input is normalized to 16 kHz mono WAV with it.
Without a system CUDA toolkit, add the venv's `nvidia/{cublas,cudnn,cuda_nvrtc}/lib`
directories under `site-packages` to `LD_LIBRARY_PATH` (ctranslate2 dlopens them).

## Usage

```bash
python3 server.py      # model server
python3 router.py      # router
WHISPER_SCRATCH=$(mktemp -d) python3 -m pytest tests   # no GPU needed
```

### `POST /transcribe`

```bash
curl -s -X POST http://127.0.0.1:8771/transcribe -F "file=@clip.ogg"        # multipart
curl -s -X POST http://127.0.0.1:8771/transcribe -H "Content-Type: application/json" \
  -d '{"path":"/abs/path/clip.ogg"}'                                        # server-local path
curl -s -X POST "http://127.0.0.1:8771/transcribe?language=en" --data-binary @clip.ogg   # raw body
```

`language` (form field, JSON key or query param): omit or `auto` to detect.
All forms accept `segments`, `words` and `vad` (on unless `multilingual`).
JSON and multipart also accept `initial_prompt` and `beam_size` (default 5);
JSON accepts `multilingual`.

Response: `{"text", "language", "duration", "infer_ms"}`, plus `segments`
(`start`, `end`, `text`) when requested.

### Other routes

- `POST /v1/audio/transcriptions`: OpenAI-compatible multipart (`file` or `audio`,
  `model`, `language`, `prompt`, `response_format` = `json`|`text`|`verbose_json`).
- `GET /v1/models`: configured model.
- `GET /health`: `{"ok", "model", "device", "loaded"}`; `loaded` is false while the
  model is not in memory (lazy load or idle unload).

## Configuration

Defaults are in `server.py` and `router.py`.

| Var | Used by | Notes |
|-----|---------|-------|
| `WHISPER_MODEL` | server | model name or local path |
| `WHISPER_DEVICE` | server | `cuda` or `cpu` |
| `WHISPER_COMPUTE` | server | `float16` / `int8_float16` / `int8` |
| `WHISPER_HOST`, `WHISPER_PORT` | both | primary listener |
| `WHISPER_OPENAI_HOST`, `WHISPER_OPENAI_PORT` | both | second listener, same routes; empty host disables |
| `WHISPER_LAZY_LOAD` | server | `1` loads on first request |
| `WHISPER_IDLE_UNLOAD_S` | server | unload after N idle seconds; `0` never |
| `WHISPER_MIN_FREE_MB` | server | answer 503 instead of loading when less VRAM is free; `0` skips |
| `WHISPER_NVIDIA_SMI` | server | binary for the free-VRAM check |
| `WHISPER_SCRATCH` | server | directory for uploads and ffmpeg output |
| `WHISPER_UPSTREAMS` | router | comma-separated server URLs, priority order |
| `WHISPER_ROUTER_TIMEOUT_S` | router | per forwarded request |
| `WHISPER_LOCAL_URL`, `WHISPER_LOCAL_START`, `WHISPER_LOCAL_WAIT_S` | router | last-resort local server: start command, wait for its health |

## Router

Forwards each request to the first `WHISPER_UPSTREAMS` entry that answers,
skipping one that is unreachable or answers 5xx. JSON `{"path": ...}` requests are
uploaded (large files re-encoded to 16 kHz mono Opus first); `multilingual` has no
upload form and is dropped. When every upstream fails and `WHISPER_LOCAL_URL` and
`WHISPER_LOCAL_START` are set, it runs the start command, waits up to
`WHISPER_LOCAL_WAIT_S` for health, and forwards there. `/health` reports that
server as `on_demand` without starting it.

## Notes

- The second listener binds the Docker bridge address by default. If that bind fails
  the server logs the error and serves on `WHISPER_HOST` only.
- Run the server under a service manager with `Restart=no`: a wedged GPU model
  reload that respawns in a loop pins VRAM.

## License

MIT, see [LICENSE](LICENSE).
