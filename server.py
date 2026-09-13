#!/usr/bin/env python3
"""Persistent faster-whisper transcription server — the squad's ONE resident STT.

Loads the model into GPU memory once and serves transcription over HTTP, so no
caller pays the ~5s python+CUDA model-load cost per clip. Everything that needs
speech-to-text on this box talks to this process: the Discord voice-note hook,
claude-bot, Open WebUI, fragwire. Before consolidation (2026-09-13) there were
five separate Whisper implementations here, each loading its own copy.

Two interfaces over one loaded model:

  POST /transcribe                  — native
    application/json {"path": "/abs/path"}   server-local path, no upload
    multipart/form-data with a file part
    raw audio bytes in the body
    optional: language ("auto"), segments (bool), initial_prompt, beam_size
    -> {"text", "language", "duration", "infer_ms"[, "segments"]}
    segments are [{"start", "end", "text"}] — a subtitle file in all but format.

  POST /v1/audio/transcriptions     — OpenAI-compatible
    multipart/form-data: file (or audio), model, language, prompt,
    response_format
    -> {"text": ...} | verbose json (with segments) | text/plain

  GET /health                       -> {"ok", "model", "device"}
  GET /v1/models                    -> OpenAI model list (Open WebUI probes it)

The server-local `path` form matters: the media library is 400 GB and must not
be pushed through an HTTP body to be transcribed.

Env:
  WHISPER_MODEL    faster-whisper model name or path (default: large-v3-turbo)
  WHISPER_DEVICE   cuda | cpu (default: cuda)
  WHISPER_COMPUTE  float16 | int8_float16 | int8 (default: float16 — best
                   quality on the 3090; int8 is faster but lossy, so we keep
                   float16 per the "don't compromise whisper" requirement)
  WHISPER_PORT     native listen port (default: 8771)
  WHISPER_HOST     native bind addr (default: 127.0.0.1 — loopback only)
  WHISPER_OPENAI_PORT / WHISPER_OPENAI_HOST
                   second listener, same routes, same model. Defaults to
                   172.17.0.1:8899 — the docker gateway, where Open WebUI and
                   claude-bot already point. Set the host to "" to disable it.
"""
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from faster_whisper import WhisperModel

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
PORT = int(os.environ.get("WHISPER_PORT", "8771"))
HOST = os.environ.get("WHISPER_HOST", "127.0.0.1")
# Docker gateway, NOT 0.0.0.0: on this box 0.0.0.0 also covers the home LAN and
# the tailnet address, and speech is not something to widen access to by accident.
OPENAI_PORT = int(os.environ.get("WHISPER_OPENAI_PORT", "8899"))
OPENAI_HOST = os.environ.get("WHISPER_OPENAI_HOST", "172.17.0.1")

# Scratch for uploads and ffmpeg output. Defaults away from /tmp deliberately:
# /tmp is the root SSD, and transcribing the film library would write hundreds
# of GB of intermediate wav to a disk with a TBW budget worth protecting.
SCRATCH = os.environ.get("WHISPER_SCRATCH", "/mnt/wsl-storage/scratch/whisper")

# The model is loaded on first use, not at import. Importing this module must
# stay free so the test suite (and a syntax check) never touches the GPU.
_model = None
_model_lock = threading.Lock()


def get_model():
    """Return the resident model, loading it once on first call."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                t0 = time.time()
                print(f"[whisper-server] loading model={MODEL_NAME} device={DEVICE} "
                      f"compute={COMPUTE} ...", flush=True)
                _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)
                print(f"[whisper-server] model loaded in {time.time() - t0:.1f}s",
                      flush=True)
    return _model


def _scratch_path(suffix: str) -> str:
    try:
        os.makedirs(SCRATCH, exist_ok=True)
        return tempfile.mktemp(suffix=suffix, dir=SCRATCH)
    except OSError:
        # Scratch volume missing (another box, or D: not mounted) — fall back
        # rather than refuse to transcribe.
        return tempfile.mktemp(suffix=suffix)


def _to_wav(src: str) -> str:
    """ffmpeg-convert any audio to 16kHz mono wav. Returns temp path."""
    dst = _scratch_path(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True, capture_output=True,
    )
    return dst


def _parse_multipart(raw: bytes, content_type: str) -> tuple[bytes | None, dict]:
    """Split a multipart/form-data body into (file bytes, text fields).

    The server originally accepted only JSON {path} or a raw-bytes body, so a
    natural `curl -F file=@clip.ogg` dumped the whole multipart envelope
    (boundary + part headers + bytes) onto disk and ffmpeg choked on it. The
    OpenAI route needs the sibling text fields too — model, language,
    response_format — so this returns both halves.
    """
    import email
    if "boundary=" not in content_type:
        return None, {}
    # Reconstruct a minimal MIME message so email.parser does the heavy lifting.
    header = b"Content-Type: " + content_type.encode() + b"\r\n\r\n"
    msg = email.message_from_bytes(header + raw)
    if not msg.is_multipart():
        return None, {}
    file_bytes: bytes | None = None
    fields: dict[str, str] = {}
    for part in msg.get_payload():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        # A filename marks a file upload; some clients omit it but still use the
        # conventional field name.
        if part.get_filename() or name in ("file", "audio"):
            if file_bytes is None:
                file_bytes = payload
        elif name:
            fields[name] = payload.decode(errors="replace").strip()
    return file_bytes, fields


def _language_or_none(language: str | None) -> str | None:
    """'', 'auto' and None all mean 'let whisper detect it'."""
    if not language or language == "auto":
        return None
    return language


def transcribe(path: str, language: str | None, *, initial_prompt: str | None = None,
               beam_size: int = 5, want_segments: bool = False) -> dict:
    """Transcribe one file.

    `want_segments` returns Whisper's own sentence boundaries with start/end
    offsets — which is what a subtitle file is, and what fragwire needs to place
    an event on its own clock. Callers that just want a string (voice notes)
    don't pay for the extra payload.

    `initial_prompt` primes the decoder with vocabulary: fragwire uses it so
    tickers, currencies and guidance ranges survive.
    """
    t0 = time.time()
    wav = _to_wav(path)
    try:
        segments, info = get_model().transcribe(
            wav,
            language=_language_or_none(language),
            # 5 is openai-whisper's own default. fragwire's live lane passes 1
            # for latency; nobody should lose quality by accident.
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=True,       # drop silence, speeds up + cleaner output
        )
        # `segments` is a generator: consume it once, here.
        spans = [
            {
                "start": round(float(getattr(s, "start", 0.0) or 0.0), 3),
                "end": round(float(getattr(s, "end", 0.0) or 0.0), 3),
                "text": s.text.strip(),
            }
            for s in segments
        ]
        spans = [s for s in spans if s["text"]]
        result = {
            "text": " ".join(s["text"] for s in spans).strip(),
            "language": info.language,
            "duration": round(info.duration, 2),
            "infer_ms": round((time.time() - t0) * 1000),
        }
        if want_segments:
            result["segments"] = spans
        return result
    finally:
        if wav != path:
            try:
                os.unlink(wav)
            except OSError:
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict):
        self._send_raw(code, json.dumps(payload).encode(), "application/json")

    def _send_raw(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ─── GET ────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            self._send(200, {"ok": True, "model": MODEL_NAME, "device": DEVICE})
        elif path == "/v1/models":
            # Open WebUI probes this before it will use a transcription endpoint.
            self._send(200, {
                "object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
            })
        else:
            self._send(404, {"error": "not found"})

    # ─── POST ───────────────────────────────────────────────────────────────

    def do_POST(self):
        parts = urlsplit(self.path)
        path = parts.path
        query = parse_qs(parts.query)
        length = int(self.headers.get("Content-Length", 0))
        ctype = self.headers.get("Content-Type", "")
        raw = self.rfile.read(length)
        try:
            if path == "/transcribe":
                self._handle_native(raw, ctype, query)
            elif path == "/v1/audio/transcriptions":
                self._handle_openai(raw, ctype)
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _handle_native(self, raw: bytes, ctype: str, query: dict):
        language = (query.get("language") or [None])[0]
        opts: dict = {}
        if ctype.startswith("application/json"):
            req = json.loads(raw)
            path = req.get("path")
            if not path or not os.path.isfile(path):
                self._send(400, {"error": "missing or invalid 'path'"})
                return
            self._send(200, transcribe(
                path, req.get("language") or language,
                initial_prompt=req.get("initial_prompt"),
                beam_size=int(req.get("beam_size") or 5),
                want_segments=bool(req.get("segments")),
            ))
            return
        if ctype.startswith("multipart/form-data"):
            payload, fields = _parse_multipart(raw, ctype)
            if not payload:
                self._send(400, {"error": "no file part in multipart body"})
                return
            language = fields.get("language") or language
            opts = {
                "initial_prompt": fields.get("initial_prompt") or None,
                "beam_size": int(fields.get("beam_size") or 5),
                "want_segments": fields.get("segments") in ("1", "true", "True"),
            }
        else:
            payload = raw
        self._send(200, self._transcribe_upload(payload, language, **opts))

    def _handle_openai(self, raw: bytes, ctype: str):
        """OpenAI /v1/audio/transcriptions — what claude-bot and Open WebUI speak."""
        if not ctype.startswith("multipart/form-data"):
            self._send(400, {"error": "expected multipart/form-data with a file part"})
            return
        payload, fields = _parse_multipart(raw, ctype)
        if not payload:
            self._send(400, {"error": "no 'file' part in multipart body"})
            return
        fmt = fields.get("response_format") or "json"
        result = self._transcribe_upload(
            payload, fields.get("language"),
            # OpenAI calls it `prompt`; faster-whisper calls it `initial_prompt`.
            initial_prompt=fields.get("prompt") or None,
            want_segments=(fmt == "verbose_json"),
        )
        if fmt == "text":
            self._send_raw(200, (result["text"] + "\n").encode(), "text/plain")
        elif fmt == "verbose_json":
            self._send(200, {
                "task": "transcribe",
                "text": result["text"],
                "language": result["language"],
                "duration": result["duration"],
                "segments": [
                    {"id": i, **span} for i, span in enumerate(result["segments"])
                ],
            })
        else:
            # The plain `json` contract is {"text": ...} and nothing else.
            self._send(200, {"text": result["text"]})

    def _transcribe_upload(self, payload: bytes, language: str | None,
                           **opts) -> dict:
        tmp = _scratch_path(".audio")
        with open(tmp, "wb") as f:
            f.write(payload)
        try:
            return transcribe(tmp, language, **opts)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def log_message(self, *args):
        pass  # quiet — systemd captures stdout if we want it


def _serve(host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[whisper-server] listening on {host}:{port}", flush=True)
    return httpd


if __name__ == "__main__":
    # Load before listening: a resident server that 404s its own health check
    # for the first 10 seconds is worse than one that starts a little later.
    get_model()
    if OPENAI_HOST:
        try:
            _serve(OPENAI_HOST, OPENAI_PORT)
        except OSError as e:
            # The docker bridge may not exist yet on a cold boot. Serving
            # loopback only beats refusing to start.
            print(f"[whisper-server] gateway listener unavailable ({e}); "
                  f"loopback only", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
