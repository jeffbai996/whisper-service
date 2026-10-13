"""The fragserv router: no model, forwards to the first transcription server that answers."""
import io
import json
import urllib.error

import pytest

import router

A, B = "http://primary:8771", "http://fallback:8771"


class Upstreams:
    """Fake urlopen: each upstream answers with a status and body, or is unreachable."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, request, timeout=None):
        base = request.full_url.split("/", 3)
        base = "/".join(base[:3])
        self.calls.append((base, request.full_url, request.data, dict(request.header_items())))
        answer = self.answers[base]
        if answer == "down":
            raise urllib.error.URLError("unreachable")
        status, body = answer
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "error", {"Content-Type": "application/json"}, io.BytesIO(body))
        response = io.BytesIO(body)
        response.status, response.headers = status, {"Content-Type": "application/json"}
        return response


OK = (200, b'{"text": "hello"}')


def test_a_server_local_path_is_uploaded_with_its_options(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF....")
    upstreams = Upstreams({A: OK, B: OK})
    body = json.dumps({"path": str(clip), "language": "en", "words": True, "vad": False, "beam_size": 5}).encode()
    status, _, payload, used = router.handle_post("/transcribe", "", body, "application/json", [A, B], opener=upstreams)
    assert (status, payload, used) == (200, OK[1], A)
    sent = upstreams.calls[0]
    assert sent[1] == A + "/transcribe" and sent[3]["Content-type"].startswith("multipart/form-data")
    assert b"RIFF...." in sent[2] and b'name="words"\r\n\r\ntrue' in sent[2] and b'name="vad"\r\n\r\nfalse' in sent[2]
    assert b'name="language"\r\n\r\nen' in sent[2]


def test_a_missing_path_is_refused_locally(tmp_path):
    status, _, payload, _ = router.handle_post("/transcribe", "", json.dumps({"path": str(tmp_path / "nope")}).encode(),
                                               "application/json", [A], opener=Upstreams({A: OK}))
    assert status == 400 and b"path" in payload


@pytest.mark.parametrize("primary", ["down", (500, b'{"error": "cuda"}'), (503, b"{}")])
def test_the_fallback_answers_when_the_primary_cannot(primary):
    upstreams = Upstreams({A: primary, B: OK})
    status, _, payload, used = router.handle_post("/v1/audio/transcriptions", "", b"--x--", "multipart/form-data; boundary=x",
                                                  [A, B], opener=upstreams)
    assert (status, payload, used) == (200, OK[1], B)


def test_a_client_error_is_returned_not_retried():
    upstreams = Upstreams({A: (400, b'{"error": "no file"}'), B: OK})
    status, _, payload, used = router.handle_post("/transcribe", "", b"x", "multipart/form-data; boundary=x", [A, B], opener=upstreams)
    assert (status, used) == (400, A) and len(upstreams.calls) == 1


def test_no_upstream_answering_is_a_503():
    status, _, payload, used = router.handle_post("/transcribe", "", b"x", "multipart/form-data; boundary=x", [A, B],
                                                  opener=Upstreams({A: "down", B: "down"}))
    assert status == 503 and used is None and b"no transcription server" in payload


def test_uploads_pass_through_unchanged_with_the_query_string():
    upstreams = Upstreams({A: OK})
    router.handle_post("/transcribe", "language=zh", b"--b\r\nraw", "multipart/form-data; boundary=b", [A], opener=upstreams)
    _, url, data, headers = upstreams.calls[0]
    assert url == A + "/transcribe?language=zh" and data == b"--b\r\nraw"
    assert headers["Content-type"] == "multipart/form-data; boundary=b"


def test_health_names_the_server_that_answers():
    upstreams = Upstreams({A: "down", B: (200, b'{"ok": true, "model": "large-v3-turbo", "device": "cpu"}')})
    status, document = router.health([A, B], opener=upstreams)
    assert status == 200 and document["upstream"] == B and document["device"] == "cpu"
    assert router.health([A], opener=Upstreams({A: "down"}))[0] == 503
