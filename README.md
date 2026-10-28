# whisper-service

A persistent [faster-whisper](https://github.com/SYSTRAN/faster-whisper) HTTP transcription server (`server.py`) and a model-free forwarding router (`router.py`), both built on the Python standard-library HTTP server with no web framework.

One-shot `whisper` CLI runs reload Python, CUDA and the model weights on every invocation, which takes around 5 seconds before a short clip starts decoding. Keeping the model resident in GPU memory reduces a voice note or short clip to the inference pass itself (~200–500 ms), and lets every speech-to-text caller on the machine share one copy of the weights.

```bash
# 1. Server-local path (no HTTP upload; reads straight from disk)
curl -s -X POST http://127.0.0.1:8771/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/srv/audio/note.ogg", "segments": true}'
```

```json
{
  "text": "Quarterly revenue rose twelve percent year over year.",
  "language": "en",
  "duration": 3.42,
  "infer_ms": 241,
  "segments": [
    {"start": 0.0, "end": 3.42, "text": "Quarterly revenue rose twelve percent year over year."}
  ]
}
```

```bash
# 2. Multipart upload (native route)
curl -s -X POST http://127.0.0.1:8771/transcribe \
  -F "file=@clip.ogg" -F "words=true" -F "initial_prompt=AAPL, MSFT, GOOGL"

# 3. Raw audio bytes in the body
curl -s -X POST "http://127.0.0.1:8771/transcribe?language=en&vad=0" \
  --data-binary @clip.ogg

# 4. OpenAI-compatible route (Open WebUI, OpenAI SDKs)
curl -s -X POST http://127.0.0.1:8771/v1/audio/transcriptions \
  -F "file=@clip.ogg" -F "model=whisper-1" -F "response_format=json"
# -> {"text": "Quarterly revenue rose twelve percent year over year."}
```

## How it works

### Two interfaces over one model (`server.py`)

- **`POST /transcribe` (native):** Accepts a JSON object with a server-local `path` (so transcribing a multi-gigabyte video file does not push it through an HTTP body), a `multipart/form-data` upload (`file` or `audio` part), or raw audio bytes in the request body. Every input is converted to 16 kHz mono WAV via `ffmpeg` before inference.
  - Optional parameters on all three forms: `language` (form field, JSON key or query parameter; omit or pass `auto` to detect), `segments` (include sentence `[start, end, text]` spans), `words` (include per-word `[word, start, end]` timings inside each segment; implies `segments`), and `vad` (speech filter, enabled by default unless `multilingual` is set).
  - JSON and multipart requests also accept `initial_prompt` (primes the decoder vocabulary so tickers, names and figures survive) and `beam_size` (default `5`).
  - JSON requests also accept `multilingual` (`true` disables `condition_on_previous_text` and defaults `vad_filter` to `false` across 30-second chunks).
- **`POST /v1/audio/transcriptions` (OpenAI-compatible):** Multipart form accepting `file` (or `audio`), `model`, `language`, `prompt` (mapped to `initial_prompt`), and `response_format` (`json` returning `{"text": ...}`, `text` returning `text/plain`, or `verbose_json` returning `task`, `text`, `language`, `duration` and numbered `segments`).
- **`GET /v1/models`:** Returns the configured model in OpenAI list format (probed by clients such as Open WebUI before sending audio).
- **`GET /health`:** Returns `{"ok": true, "model": "large-v3-turbo", "device": "cuda", "loaded": true}`. `loaded` is `false` while the weights are not in memory (before the first request under `WHISPER_LAZY_LOAD=1`, or after an idle unload).

### Sharing a GPU

- **Idle unload (`WHISPER_IDLE_UNLOAD_S`):** On a GPU shared with resident LLMs or embedding models, keeping ~1.6 GB of Whisper weights allocated between voice notes can push the card into shared system memory. When `WHISPER_IDLE_UNLOAD_S` is non-zero, a background reaper releases the weights via CTranslate2's `unload_model()` and runs garbage collection once the model has been idle past the window. An in-flight reference counter prevents unloading mid-transcription, and the next request reloads the model on demand.
- **Free-VRAM guard (`WHISPER_MIN_FREE_MB`):** Before loading weights on CUDA, the server checks free memory via `nvidia-smi --query-gpu=memory.free`. When free VRAM is below the threshold, it returns `503 {"error": "GPU busy: ..."}` instead of letting the driver spill into host RAM, allowing a router in front to move to the next server immediately.

### Failover router (`router.py`)

`router.py` exposes the same routes and listeners without importing `faster-whisper`, for a host whose GPU should not carry Whisper or that routes across several machines:

- **Fast health probe before long reads:** A long transcription can run for minutes (`WHISPER_ROUTER_TIMEOUT_S`, default 900 s). Before posting audio to an upstream in `WHISPER_UPSTREAMS`, the router probes its `GET /health` with a 5-second timeout so a sleeping or unreachable machine costs at most 5 seconds to skip. Upstreams that fail the probe, time out or return `5xx` are skipped; `4xx` client errors are returned immediately without retrying.
- **Local path to remote upload:** When a caller sends `{"path": "/abs/path/film.mkv"}`, an upstream on `127.0.0.1`, `localhost` or `::1` receives the JSON body as is and reads the file directly. For a remote upstream, the router converts the request into a multipart upload carrying `language`, `initial_prompt`, `beam_size`, `segments`, `words` and `vad` (`multilingual` has no upload form and is dropped). If the local file exceeds 25 MB, `ffmpeg` strips video and re-encodes the audio to 16 kHz mono 32 kbps Opus first, once per request.
- **Last-resort on-demand local server:** When `WHISPER_LOCAL_URL` and `WHISPER_LOCAL_START` are set and every upstream in `WHISPER_UPSTREAMS` fails, the router runs the start command, polls the local server's `/health` every 2 seconds up to `WHISPER_LOCAL_WAIT_S`, and forwards the request there. `GET /health` on the router reports that fallback with `"on_demand": true` without starting it.

### Listeners and scratch storage

- **Docker bridge listener:** Both `server.py` and `router.py` bind `WHISPER_HOST:WHISPER_PORT` (`127.0.0.1:8771`) and a second listener on `WHISPER_OPENAI_HOST:WHISPER_OPENAI_PORT` (`172.17.0.1:8899`, the Docker bridge gateway) so containerized callers reach the host service without binding `0.0.0.0`. If the second bind fails (for example before Docker creates the bridge interface), the server logs the error and serves on `WHISPER_HOST` only.
- **Explicit scratch directory:** Uploaded clips and temporary 16 kHz WAV files are written to `WHISPER_SCRATCH` without falling back to `/tmp`, so transcribing large media files does not silently write intermediate WAVs to a root SSD.

## Install and run

Requires Python 3.10+ and `ffmpeg` on `PATH`.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On a machine without a system-wide CUDA toolkit, point the dynamic linker at the NVIDIA wheels installed in the virtualenv (CTranslate2 `dlopen`s `libcublas` and `libcudnn` at inference time):

```bash
export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
```

Start the model server or the router:

```bash
WHISPER_SCRATCH=/tmp/whisper python3 server.py
WHISPER_UPSTREAMS="http://gpu-host:8771" python3 router.py
```

## Configuration

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | server | faster-whisper model name or local directory path |
| `WHISPER_DEVICE` | `cuda` | server | `cuda` or `cpu` |
| `WHISPER_COMPUTE` | `float16` | server | `float16`, `int8_float16`, or `int8` |
| `WHISPER_HOST` | `127.0.0.1` | both | primary bind address |
| `WHISPER_PORT` | `8771` | both | primary listen port |
| `WHISPER_OPENAI_HOST` | `172.17.0.1` | both | second listener bind address; set empty (`""`) to disable |
| `WHISPER_OPENAI_PORT` | `8899` | both | second listener port |
| `WHISPER_LAZY_LOAD` | unset | server | `1` loads the model on the first request instead of at startup |
| `WHISPER_IDLE_UNLOAD_S` | `0` | server | unload weights after this many idle seconds (`0` never unloads) |
| `WHISPER_MIN_FREE_MB` | `0` | server | minimum free VRAM in MB required to load (`0` skips check); answers `503` when below |
| `WHISPER_NVIDIA_SMI` | `nvidia-smi` | server | binary queried for free VRAM |
| `WHISPER_SCRATCH` | set it | server | directory for uploads and temporary WAV files; the code's default is a path on the author's machine |
| `WHISPER_UPSTREAMS` | required | router | comma-separated upstream server base URLs, in priority order |
| `WHISPER_ROUTER_TIMEOUT_S` | `900` | router | per-upstream read timeout in seconds |
| `WHISPER_LOCAL_URL` | unset | router | last-resort local server URL |
| `WHISPER_LOCAL_START` | unset | router | command run to start the last-resort local server |
| `WHISPER_LOCAL_WAIT_S` | `180` | router | seconds to wait for the local server to pass `/health` |

### Running under systemd

If running `server.py` under a service manager, set `Restart=no` so a wedged GPU model reload does not respawn in a loop and pin VRAM:

```ini
[Unit]
Description=faster-whisper resident transcription server

[Service]
Type=simple
WorkingDirectory=%h/whisper-service
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=%h/whisper-service/venv/lib/python3.12/site-packages/nvidia/cublas/lib:%h/whisper-service/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:%h/whisper-service/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
Environment=WHISPER_MODEL=large-v3-turbo
Environment=WHISPER_DEVICE=cuda
Environment=WHISPER_COMPUTE=float16
Environment=WHISPER_SCRATCH=/tmp/whisper
ExecStart=%h/whisper-service/venv/bin/python %h/whisper-service/server.py
Restart=no

[Install]
WantedBy=default.target
```

## Tests

The test suite injects a fake model and does not touch the GPU:

```bash
WHISPER_SCRATCH=$(mktemp -d) python3 -m pytest tests
```

## License

MIT. See [LICENSE](LICENSE).
