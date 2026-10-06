"""Tests for the resident faster-whisper server.

None of these load a real model. `server` exposes the model lazily and these
tests inject a fake, so the suite never touches the 3090 — which matters because
the card is usually busy with something else.
"""
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


class FakeSegment:
    def __init__(self, text: str, start: float = 0.0, end: float = 1.0):
        self.text = text
        self.start = start
        self.end = end


class FakeInfo:
    def __init__(self, language: str = "en", duration: float = 12.34):
        self.language = language
        self.duration = duration


def test_multilingual_mode_is_explicit_and_disables_previous_text(fake_model, tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"test")
    server.transcribe(str(source), "auto", multilingual=True, want_segments=True)
    call = fake_model.calls[-1]
    assert call["multilingual"] is True
    assert call["language"] is None
    assert call["condition_on_previous_text"] is False
    assert call["chunk_length"] == 30
    assert call["vad_filter"] is False
    server.transcribe(str(source), "en")
    assert fake_model.calls[-1]["multilingual"] is False
    assert fake_model.calls[-1]["condition_on_previous_text"] is True


class FakeModel:
    """Records how it was called so tests can assert on plumbing, not output."""

    def __init__(self, text: str = "hello world", language: str = "en"):
        self._text = text
        self._language = language
        self.calls: list[dict] = []

    def transcribe(self, wav, **kwargs):
        self.calls.append({"wav": wav, **kwargs})
        return (
            [FakeSegment("hello", 0.0, 1.5), FakeSegment("world", 1.5, 3.0)]
            if self._text == "hello world"
            else [FakeSegment(self._text)],
            FakeInfo(self._language),
        )


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(server, "_model", model)
    # ffmpeg conversion is I/O we do not want in a unit test; the identity
    # stand-in keeps the path plumbing honest without shelling out.
    monkeypatch.setattr(server, "_to_wav", lambda src: src)
    return model


@pytest.fixture
def live_server(fake_model):
    """A real HTTP listener on an ephemeral port, serving the fake model."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


def _post(base: str, path: str, body: bytes, content_type: str):
    req = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def _multipart(fields: dict[str, str], filename: str | None = "clip.ogg",
               file_field: str = "file", file_bytes: bytes = b"\x00audio-bytes"):
    """Build a multipart/form-data body the way an OpenAI SDK client would."""
    boundary = "----testboundary123"
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buf.write(f"{value}\r\n".encode())
    if filename is not None:
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'.encode()
        )
        buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
        buf.write(file_bytes + b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


# ─── laziness ────────────────────────────────────────────────────────────────

def test_model_is_not_loaded_at_import():
    """Importing the module must not touch the GPU.

    The original server loaded the model at import time, which made the module
    untestable and meant a syntax check cost 2.5 GB of VRAM.
    """
    assert server._model is None or isinstance(server._model, FakeModel)


def test_get_model_loads_once(monkeypatch):
    """Two calls share one instance — the whole point of a resident server."""
    monkeypatch.setattr(server, "_model", None)
    loads = []

    def fake_loader(*args, **kwargs):
        loads.append(1)
        return FakeModel()

    monkeypatch.setattr(server, "WhisperModel", fake_loader)
    first = server.get_model()
    second = server.get_model()
    assert first is second
    assert len(loads) == 1


# ─── health and discovery ────────────────────────────────────────────────────

def test_health_reports_model_and_device(live_server):
    status, body, _ = _get(live_server, "/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["model"] == server.MODEL_NAME
    assert payload["device"] == server.DEVICE


def test_v1_models_lists_the_loaded_model(live_server):
    """Open WebUI probes /v1/models before it will use a transcription endpoint."""
    status, body, _ = _get(live_server, "/v1/models")
    assert status == 200
    payload = json.loads(body)
    assert payload["object"] == "list"
    assert [m["id"] for m in payload["data"]] == [server.MODEL_NAME]


def test_unknown_path_404s(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(live_server, "/nope")
    assert exc.value.code == 404


# ─── OpenAI-compatible route ─────────────────────────────────────────────────

def test_openai_route_returns_text_only_json(live_server, fake_model):
    body, ctype = _multipart({"model": "whisper-1"})
    status, raw, resp_ctype = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 200
    assert "application/json" in resp_ctype
    payload = json.loads(raw)
    # The OpenAI contract for response_format=json is {"text": ...} and nothing
    # a client is required to ignore.
    assert payload == {"text": "hello world"}


def test_openai_route_response_format_text_returns_plain_text(live_server):
    body, ctype = _multipart({"model": "whisper-1", "response_format": "text"})
    status, raw, resp_ctype = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 200
    assert "text/plain" in resp_ctype
    assert raw.decode().strip() == "hello world"


def test_openai_route_verbose_json_includes_language_and_duration(live_server):
    body, ctype = _multipart({"response_format": "verbose_json"})
    status, raw, _ = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 200
    payload = json.loads(raw)
    assert payload["text"] == "hello world"
    assert payload["language"] == "en"
    assert payload["duration"] == 12.34


def test_openai_route_passes_language_through(live_server, fake_model):
    body, ctype = _multipart({"model": "whisper-1", "language": "zh"})
    status, _, _ = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 200
    assert fake_model.calls[0]["language"] == "zh"


def test_openai_route_auto_language_means_detect(live_server, fake_model):
    """'auto' and '' are both 'let whisper decide', i.e. language=None."""
    body, ctype = _multipart({"language": "auto"})
    _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert fake_model.calls[0]["language"] is None


def test_openai_route_missing_file_part_is_400(live_server):
    body, ctype = _multipart({"model": "whisper-1"}, filename=None)
    status, raw, _ = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 400
    assert "file" in json.loads(raw)["error"].lower()


def test_openai_route_accepts_audio_field_name(live_server):
    """Some clients send the part as `audio` rather than `file`."""
    body, ctype = _multipart({}, file_field="audio")
    status, raw, _ = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 200
    assert json.loads(raw)["text"] == "hello world"


# ─── native route ────────────────────────────────────────────────────────────

def test_native_route_transcribes_a_server_local_path(live_server, tmp_path, fake_model):
    """The library job feeds paths, not uploads — 400 GB of film must not go
    through an HTTP body."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x00")
    body = json.dumps({"path": str(clip)}).encode()
    status, raw, _ = _post(live_server, "/transcribe", body, "application/json")
    assert status == 200
    payload = json.loads(raw)
    assert payload["text"] == "hello world"
    assert payload["language"] == "en"
    assert payload["duration"] == 12.34
    assert "infer_ms" in payload
    assert fake_model.calls[0]["wav"] == str(clip)


def test_native_route_rejects_missing_path(live_server):
    body = json.dumps({"path": "/definitely/not/here.wav"}).encode()
    status, raw, _ = _post(live_server, "/transcribe", body, "application/json")
    assert status == 400
    assert "path" in json.loads(raw)["error"]


def test_native_route_accepts_query_string_language(live_server, fake_model):
    """Regression: routing compared the full path, so `/transcribe?language=zh`
    fell through to a 404 and the query parsing below it was dead code."""
    status, _, _ = _post(
        live_server, "/transcribe?language=zh", b"\x00audio", "application/octet-stream"
    )
    assert status == 200
    assert fake_model.calls[0]["language"] == "zh"


def test_native_route_accepts_multipart_upload(live_server):
    body, ctype = _multipart({}, file_field="audio")
    status, raw, _ = _post(live_server, "/transcribe", body, ctype)
    assert status == 200
    assert json.loads(raw)["text"] == "hello world"


# ─── timed segments and decoding controls ────────────────────────────────────

def test_segments_are_omitted_unless_asked_for(live_server, tmp_path):
    """Voice notes want a string. Only the subtitle work wants the timings, and
    it pays for them explicitly."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x00")
    body = json.dumps({"path": str(clip)}).encode()
    _, raw, _ = _post(live_server, "/transcribe", body, "application/json")
    assert "segments" not in json.loads(raw)


def test_segments_carry_start_end_and_text(live_server, tmp_path):
    """A subtitle file IS timed segments — this is what generating an SRT from
    audio needs, and what fragwire needs to place events on its clock."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x00")
    body = json.dumps({"path": str(clip), "segments": True}).encode()
    _, raw, _ = _post(live_server, "/transcribe", body, "application/json")
    payload = json.loads(raw)
    assert payload["text"] == "hello world"
    assert payload["segments"] == [
        {"start": 0.0, "end": 1.5, "text": "hello"},
        {"start": 1.5, "end": 3.0, "text": "world"},
    ]


def test_initial_prompt_is_passed_to_the_model(live_server, fake_model, tmp_path):
    """fragwire primes the model with financial vocabulary so tickers and
    guidance ranges survive."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x00")
    body = json.dumps({
        "path": str(clip), "initial_prompt": "Preserve tickers and currencies."
    }).encode()
    _post(live_server, "/transcribe", body, "application/json")
    assert fake_model.calls[0]["initial_prompt"] == "Preserve tickers and currencies."


def test_beam_size_is_overridable_and_defaults_to_five(live_server, fake_model,
                                                       tmp_path):
    """fragwire's live lane runs beam_size=1 for latency; the default stays at
    openai-whisper's 5 so nobody silently loses quality."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x00")
    _post(live_server, "/transcribe",
          json.dumps({"path": str(clip)}).encode(), "application/json")
    assert fake_model.calls[0]["beam_size"] == 5

    _post(live_server, "/transcribe",
          json.dumps({"path": str(clip), "beam_size": 1}).encode(),
          "application/json")
    assert fake_model.calls[1]["beam_size"] == 1


def test_openai_prompt_field_maps_to_initial_prompt(live_server, fake_model):
    """OpenAI calls it `prompt`; faster-whisper calls it `initial_prompt`."""
    body, ctype = _multipart({"prompt": "SEC filing audio."})
    _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert fake_model.calls[0]["initial_prompt"] == "SEC filing audio."


def test_openai_verbose_json_includes_segments(live_server):
    """OpenAI's verbose_json carries segments, and clients that ask for it
    expect them."""
    body, ctype = _multipart({"response_format": "verbose_json"})
    _, raw, _ = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    payload = json.loads(raw)
    assert [s["text"] for s in payload["segments"]] == ["hello", "world"]
    assert payload["segments"][0]["start"] == 0.0
    assert payload["segments"][1]["end"] == 3.0


# ─── cleanup ─────────────────────────────────────────────────────────────────

def test_scratch_path_does_not_fall_back_to_default_tempdir(tmp_path, monkeypatch):
    """A missing scratch volume must not redirect large WAVs onto root /tmp."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked")
    monkeypatch.setattr(server, "SCRATCH", str(blocked))

    with pytest.raises(OSError):
        server._scratch_path(".wav")


def test_upload_temp_file_is_removed(live_server, monkeypatch):
    """An uploaded clip is written to scratch; it must not survive the request."""
    written: list[str] = []
    real_mktemp = server.tempfile.mktemp

    def spy(*args, **kwargs):
        path = real_mktemp(*args, **kwargs)
        written.append(path)
        return path

    monkeypatch.setattr(server.tempfile, "mktemp", spy)
    body, ctype = _multipart({})
    status, _, _ = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 200
    assert written, "expected the upload to be spooled to a temp file"
    import os
    assert not any(os.path.exists(p) for p in written), (
        f"temp files left behind: {[p for p in written if os.path.exists(p)]}"
    )


def test_transcribe_failure_is_a_500_not_a_hang(live_server, fake_model):
    def boom(*args, **kwargs):
        raise RuntimeError("cuda fell over")

    fake_model.transcribe = boom
    body, ctype = _multipart({})
    status, raw, _ = _post(live_server, "/v1/audio/transcriptions", body, ctype)
    assert status == 500
    assert "cuda fell over" in json.loads(raw)["error"]
