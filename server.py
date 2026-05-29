#!/usr/bin/env python3
"""Persistent faster-whisper transcription server.

Loads the model into GPU memory once at startup and serves transcription over
HTTP. Eliminates the per-call python+torch+CUDA startup cost (~5s) that makes
one-shot whisper invocations slow — steady-state latency drops to just the
inference (~0.2-0.5s for a short clip on a 3090).

POST /transcribe
  multipart/form-data with an `audio` file field, OR
  application/json {"path": "/abs/path/to/audio"}  (server-local path)
  optional `language` field (default: auto-detect)

Returns JSON: {"text": "...", "language": "en", "duration": 12.3, "infer_ms": 287}

Env:
  WHISPER_MODEL    faster-whisper model name or path (default: small)
  WHISPER_DEVICE   cuda | cpu (default: cuda)
  WHISPER_COMPUTE  float16 | int8_float16 | int8 (default: float16 — best
                   quality on the 3090; int8 is faster but lossy, so we keep
                   float16 per the "don't compromise whisper" requirement)
  WHISPER_PORT     listen port (default: 8771)
  WHISPER_HOST     bind addr (default: 127.0.0.1 — tailnet-only via reverse proxy)
"""
import json
import os
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from faster_whisper import WhisperModel

MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
PORT = int(os.environ.get("WHISPER_PORT", "8771"))
HOST = os.environ.get("WHISPER_HOST", "127.0.0.1")

print(f"[whisper-server] loading model={MODEL_NAME} device={DEVICE} compute={COMPUTE} ...", flush=True)
_t0 = time.time()
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)
print(f"[whisper-server] model loaded in {time.time()-_t0:.1f}s, listening on {HOST}:{PORT}", flush=True)


def _to_wav(src: str) -> str:
    """ffmpeg-convert any audio to 16kHz mono wav. Returns temp path."""
    dst = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True, capture_output=True,
    )
    return dst


def _extract_multipart_file(raw: bytes, content_type: str) -> bytes | None:
    """Pull the first file part's raw bytes out of a multipart/form-data body.

    The server originally accepted only JSON {path} or a raw-bytes body, so a
    natural `curl -F file=@clip.ogg` dumped the whole multipart envelope
    (boundary + part headers + bytes) onto disk and ffmpeg choked on it. Parse
    the envelope here and return just the payload. Returns None if the body
    isn't valid multipart or has no file part.
    """
    import email
    if "boundary=" not in content_type:
        return None
    # Reconstruct a minimal MIME message so email.parser does the heavy lifting.
    header = b"Content-Type: " + content_type.encode() + b"\r\n\r\n"
    msg = email.message_from_bytes(header + raw)
    if not msg.is_multipart():
        return None
    for part in msg.get_payload():
        # First part that carries a filename (a file upload) wins.
        if part.get_filename():
            payload = part.get_payload(decode=True)
            if payload:
                return payload
    return None


def transcribe(path: str, language: str | None) -> dict:
    t0 = time.time()
    wav = _to_wav(path)
    try:
        segments, info = model.transcribe(
            wav,
            language=None if (not language or language == "auto") else language,
            beam_size=5,           # same default as openai-whisper — no quality loss
            vad_filter=True,       # drop silence, speeds up + cleaner output
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": round(info.duration, 2),
            "infer_ms": round((time.time() - t0) * 1000),
        }
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": MODEL_NAME, "device": DEVICE})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/transcribe":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        ctype = self.headers.get("Content-Type", "")
        raw = self.rfile.read(length)
        try:
            if ctype.startswith("application/json"):
                req = json.loads(raw)
                path = req.get("path")
                language = req.get("language")
                if not path or not os.path.isfile(path):
                    self._send(400, {"error": "missing or invalid 'path'"})
                    return
                self._send(200, transcribe(path, language))
            elif ctype.startswith("multipart/form-data"):
                # curl -F "file=@clip.ogg" — extract just the file bytes; the
                # whole envelope used to reach ffmpeg and fail (exit 183).
                payload = _extract_multipart_file(raw, ctype)
                if not payload:
                    self._send(400, {"error": "no file part in multipart body"})
                    return
                tmp = tempfile.mktemp(suffix=".audio")
                with open(tmp, "wb") as f:
                    f.write(payload)
                try:
                    lang = None
                    if "?language=" in self.path:
                        lang = self.path.split("?language=", 1)[1]
                    self._send(200, transcribe(tmp, lang))
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            else:
                # raw audio bytes in body; optional ?language=xx
                tmp = tempfile.mktemp(suffix=".audio")
                with open(tmp, "wb") as f:
                    f.write(raw)
                try:
                    lang = None
                    if "?language=" in self.path:
                        lang = self.path.split("?language=", 1)[1]
                    self._send(200, transcribe(tmp, lang))
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):
        pass  # quiet — systemd captures stdout if we want it


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
