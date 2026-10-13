#!/usr/bin/env python3
"""Whisper router: the same routes as server.py, without a model.

For a host whose GPU should not carry Whisper. Each request goes to the first
transcription server in WHISPER_UPSTREAMS that answers; one that is unreachable,
times out or answers 5xx is skipped for that request. A 4xx is the caller's
problem and is returned as is. Callers keep their URLs: the router listens where
server.py listened.

A JSON {"path": ...} request names a file on this host, which the upstream
cannot read, so the router uploads it as multipart with the same options. Files
over UPLOAD_SHRINK_BYTES are re-encoded to 16 kHz mono Opus first (the server
resamples to 16 kHz mono anyway), so a film's audio is not sent as raw video.
`multilingual` has no upload form and is dropped.

A last-resort local server (WHISPER_LOCAL_URL) is used only when every upstream
has failed. It is not kept running: the router starts it then
(WHISPER_LOCAL_START, a command) and waits for its health, and the server itself
unloads the model when idle.

Env:
  WHISPER_UPSTREAMS    comma-separated base URLs, in priority order
  WHISPER_PORT / WHISPER_HOST                    native listener (127.0.0.1:8771)
  WHISPER_OPENAI_PORT / WHISPER_OPENAI_HOST      second listener ("" disables)
  WHISPER_ROUTER_TIMEOUT_S                       per-upstream read timeout (900)
  WHISPER_LOCAL_URL / WHISPER_LOCAL_START        last-resort local server and its start command
  WHISPER_LOCAL_WAIT_S                           how long to wait for it to come up (180)
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

UPSTREAMS = [url.strip().rstrip("/") for url in os.environ.get("WHISPER_UPSTREAMS", "").split(",") if url.strip()]
HOST = os.environ.get("WHISPER_HOST", "127.0.0.1")
PORT = int(os.environ.get("WHISPER_PORT", "8771"))
OPENAI_HOST = os.environ.get("WHISPER_OPENAI_HOST", "172.17.0.1")
OPENAI_PORT = int(os.environ.get("WHISPER_OPENAI_PORT", "8899"))
TIMEOUT_S = float(os.environ.get("WHISPER_ROUTER_TIMEOUT_S", "900"))
HEALTH_TIMEOUT_S = 5.0
UPLOAD_SHRINK_BYTES = 25 * 1024 * 1024
LOCAL_URL = os.environ.get("WHISPER_LOCAL_URL", "").strip().rstrip("/")
LOCAL_START = shlex.split(os.environ.get("WHISPER_LOCAL_START", ""))
LOCAL_WAIT_S = float(os.environ.get("WHISPER_LOCAL_WAIT_S", "180"))
# JSON options of server.py's path form that its upload form also accepts.
UPLOAD_FIELDS = ("language", "initial_prompt", "beam_size", "segments", "words", "vad")


def _field_value(value) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


def multipart(file_bytes: bytes, filename: str, fields: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{_field_value(value)}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def audio_bytes(path: Path) -> tuple[bytes, str]:
    """The file's bytes, re-encoded to small speech audio when it is large."""
    if path.stat().st_size <= UPLOAD_SHRINK_BYTES:
        return path.read_bytes(), path.name
    with tempfile.TemporaryDirectory(prefix="whisper-router-") as folder:
        target = Path(folder) / "audio.ogg"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "libopus", "-b:a", "32k", str(target)], check=True, timeout=1800)
        return target.read_bytes(), "audio.ogg"


def forward(method: str, path: str, body: bytes | None, content_type: str | None, upstreams: list[str], *,
            opener=urllib.request.urlopen, timeout: float = TIMEOUT_S) -> tuple[int, str, bytes, str | None]:
    """(status, content type, body, upstream used) from the first upstream that answers."""
    for base in upstreams:
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(base + path, data=body, headers=headers, method=method)
        try:
            with opener(request, timeout=timeout) as response:
                return response.status, response.headers.get("Content-Type", "application/json"), response.read(), base
        except urllib.error.HTTPError as error:
            if error.code >= 500:
                continue
            return error.code, error.headers.get("Content-Type", "application/json"), error.read(), base
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
    return 503, "application/json", json.dumps({"error": "no transcription server answered"}).encode(), None


class LocalFallback:
    """A local server that is started only when it is needed."""

    def __init__(self, url: str, start, *, wait_s: float = LOCAL_WAIT_S, sleep=time.sleep):
        self.url, self.start, self.wait_s, self.sleep = url, start, wait_s, sleep
        self.lock = threading.Lock()

    def healthy(self, opener) -> dict | None:
        status, _, body, _ = forward("GET", "/health", None, None, [self.url], opener=opener, timeout=HEALTH_TIMEOUT_S)
        if status != 200:
            return None
        try:
            return json.loads(body)
        except ValueError:
            return None

    def ensure(self, opener) -> bool:
        """Start the server unless it is already answering, then wait for it."""
        with self.lock:
            if self.healthy(opener) is not None:
                return True
            print("[whisper-router] every upstream failed; starting the local server", flush=True)
            self.start()
            waited = 0.0
            while waited < self.wait_s:
                if self.healthy(opener) is not None:
                    return True
                self.sleep(2)
                waited += 2
            return False


def _start_command():
    subprocess.run(LOCAL_START, check=False, timeout=60)


LOCAL = LocalFallback(LOCAL_URL, _start_command) if LOCAL_URL and LOCAL_START else None


def answers(base: str, opener) -> bool:
    """Whether the upstream answers its health check within the short timeout. A
    transcription can take minutes, so the long read timeout must not also be how
    long a sleeping host takes to be skipped."""
    status, _, _, used = forward("GET", "/health", None, None, [base], opener=opener, timeout=HEALTH_TIMEOUT_S)
    return used is not None and status < 500


def send(target: str, body: bytes, content_type: str, upstreams: list[str], opener) -> tuple[int, str, bytes, str | None]:
    """POST to each upstream in turn, skipping ones that do not answer the probe."""
    for base in upstreams:
        if not answers(base, opener):
            continue
        result = forward("POST", target, body, content_type, [base], opener=opener)
        if result[3] is not None:
            return result
    return 503, "application/json", json.dumps({"error": "no transcription server answered"}).encode(), None


def handle_post(path: str, query: str, body: bytes, content_type: str, upstreams: list[str], *,
                opener=urllib.request.urlopen, local: LocalFallback | None = None) -> tuple[int, str, bytes, str | None]:
    if path == "/transcribe" and content_type.startswith("application/json"):
        try:
            request = json.loads(body or b"{}")
        except ValueError:
            return 400, "application/json", json.dumps({"error": "invalid JSON"}).encode(), None
        source = Path(str(request.get("path") or ""))
        if not request.get("path") or not source.is_file():
            return 400, "application/json", json.dumps({"error": "missing or invalid 'path'"}).encode(), None
        data, name = audio_bytes(source)
        fields = {key: request[key] for key in UPLOAD_FIELDS if request.get(key) is not None}
        body, content_type = multipart(data, name, fields)
    target = path + (f"?{query}" if query else "")
    result = send(target, body, content_type, upstreams, opener)
    if result[3] is None and local is not None and local.ensure(opener):
        return forward("POST", target, body, content_type, [local.url], opener=opener)
    return result


def health(upstreams: list[str], *, opener=urllib.request.urlopen, local: LocalFallback | None = None) -> tuple[int, dict]:
    for base in upstreams:
        status, _, body, used = forward("GET", "/health", None, None, [base], opener=opener, timeout=HEALTH_TIMEOUT_S)
        if status == 200:
            try:
                document = json.loads(body)
            except ValueError:
                continue
            return 200, {**document, "upstream": used, "router": True}
    if local is not None:
        # Reported, not started: a health probe must not load a model.
        document = local.healthy(opener) or {"ok": True, "loaded": False}
        return 200, {**document, "upstream": local.url, "on_demand": True, "router": True}
    return 503, {"ok": False, "router": True, "error": "no transcription server answered", "upstreams": upstreams}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            code, document = health(UPSTREAMS, local=LOCAL)
            self._send(code, "application/json", json.dumps(document).encode())
        elif path == "/v1/models":
            self._send(*forward("GET", path, None, None, UPSTREAMS, timeout=HEALTH_TIMEOUT_S)[:3])
        else:
            self._send(404, "application/json", b'{"error": "not found"}')

    def do_POST(self):
        parts = urlsplit(self.path)
        if parts.path not in ("/transcribe", "/v1/audio/transcriptions"):
            self._send(404, "application/json", b'{"error": "not found"}')
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            status, content_type, payload, used = handle_post(parts.path, parts.query, body,
                                                              self.headers.get("Content-Type", ""), UPSTREAMS, local=LOCAL)
        except Exception as error:  # a broken local file or ffmpeg failure is this request's error only
            status, content_type, payload, used = 500, "application/json", json.dumps(
                {"error": f"{type(error).__name__}: {error}"}).encode(), None
        print(f"[whisper-router] {parts.path} -> {used or 'none'} {status}", flush=True)
        self._send(status, content_type, payload)


def _serve(host: str, port: int) -> None:
    threading.Thread(target=ThreadingHTTPServer((host, port), Handler).serve_forever, daemon=True).start()


if __name__ == "__main__":
    if not UPSTREAMS:
        raise SystemExit("WHISPER_UPSTREAMS is empty")
    if OPENAI_HOST:
        try:
            _serve(OPENAI_HOST, OPENAI_PORT)
        except OSError as error:
            print(f"[whisper-router] gateway listener unavailable ({error}); loopback only", flush=True)
    print(f"[whisper-router] {HOST}:{PORT} -> {', '.join(UPSTREAMS)}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
