"""The router: no model, forwards to the first transcription server that answers."""
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
    sent = next(call for call in upstreams.calls if call[1].endswith("/transcribe"))
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
    assert (status, used) == (400, A)
    assert [call[1] for call in upstreams.calls] == [A + "/health", A + "/transcribe"]


def test_no_upstream_answering_is_a_503():
    status, _, payload, used = router.handle_post("/transcribe", "", b"x", "multipart/form-data; boundary=x", [A, B],
                                                  opener=Upstreams({A: "down", B: "down"}))
    assert status == 503 and used is None and b"no transcription server" in payload


def test_uploads_pass_through_unchanged_with_the_query_string():
    upstreams = Upstreams({A: OK})
    router.handle_post("/transcribe", "language=zh", b"--b\r\nraw", "multipart/form-data; boundary=b", [A], opener=upstreams)
    _, url, data, headers = upstreams.calls[-1]
    assert url == A + "/transcribe?language=zh" and data == b"--b\r\nraw"
    assert headers["Content-type"] == "multipart/form-data; boundary=b"


def test_health_names_the_server_that_answers():
    upstreams = Upstreams({A: "down", B: (200, b'{"ok": true, "model": "large-v3-turbo", "device": "cpu"}')})
    status, document = router.health([A, B], opener=upstreams)
    assert status == 200 and document["upstream"] == B and document["device"] == "cpu"
    assert router.health([A], opener=Upstreams({A: "down"}))[0] == 503


# -- last resort: a local server started only when every remote has failed -----

LOCAL = "http://127.0.0.1:18771"


class Local:
    """A local server that answers health only after it has been started."""

    def __init__(self, upstreams, comes_up=True, running=False):
        self.upstreams, self.comes_up, self.running, self.starts = upstreams, comes_up, running, 0

    def start(self):
        self.starts += 1
        self.running = self.comes_up

    def opener(self, request, timeout=None):
        if request.full_url.startswith(LOCAL):
            if not self.running:
                raise urllib.error.URLError("refused")
            path = request.full_url[len(LOCAL):]
            body = b'{"ok": true, "device": "cpu", "loaded": true}' if path == "/health" else b'{"text": "local"}'
            response = io.BytesIO(body)
            response.status, response.headers = 200, {"Content-Type": "application/json"}
            return response
        return self.upstreams(request, timeout)


def post(local, remotes):
    return router.handle_post("/transcribe", "", b"x", "multipart/form-data; boundary=x", remotes,
                              opener=local.opener, local=router.LocalFallback(LOCAL, local.start, wait_s=5, sleep=lambda s: None))


def test_the_local_server_is_started_only_when_every_remote_fails():
    local = Local(Upstreams({A: "down", B: (500, b"{}")}))
    status, _, payload, used = post(local, [A, B])
    assert (status, payload, used, local.starts) == (200, b'{"text": "local"}', LOCAL, 1)


def test_a_working_remote_never_starts_the_local_server():
    local = Local(Upstreams({A: OK}))
    assert post(local, [A])[3] == A and local.starts == 0


def test_a_running_local_server_is_used_without_starting_it_again():
    local = Local(Upstreams({A: "down"}), running=True)
    assert post(local, [A])[3] == LOCAL and local.starts == 0


def test_a_local_server_that_does_not_come_up_is_a_503():
    local = Local(Upstreams({A: "down"}), comes_up=False)
    status, _, payload, used = post(local, [A])
    assert status == 503 and used is None and local.starts == 1


def test_health_reports_the_on_demand_fallback_without_starting_it():
    local = Local(Upstreams({A: "down"}))
    status, document = router.health([A], opener=local.opener, local=router.LocalFallback(LOCAL, local.start))
    assert status == 200 and document["upstream"] == LOCAL and document["on_demand"] is True and document["loaded"] is False
    assert local.starts == 0


def test_a_path_request_reaches_the_local_server_when_every_remote_fails(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF....")
    local = Local(Upstreams({A: "down"}))
    status, _, payload, used = router.handle_post(
        "/transcribe", "", json.dumps({"path": str(clip)}).encode(), "application/json", [A], opener=local.opener,
        local=router.LocalFallback(LOCAL, local.start, wait_s=5, sleep=lambda s: None))
    assert (status, used, local.starts) == (200, LOCAL, 1)


class Hanging(Upstreams):
    """An upstream whose health probe times out, like a sleeping laptop on the tailnet."""

    def __call__(self, request, timeout=None):
        if request.full_url.startswith(A):
            self.calls.append((A, request.full_url, request.data, {}))
            assert timeout is not None and timeout <= router.HEALTH_TIMEOUT_S, "a dead upstream must cost a short timeout"
            raise TimeoutError("timed out")
        return super().__call__(request, timeout)


def test_an_unresponsive_upstream_costs_only_the_short_probe():
    upstreams = Hanging({B: OK})
    status, _, _, used = router.handle_post("/transcribe", "", b"x", "multipart/form-data; boundary=x", [A, B], opener=upstreams)
    assert (status, used) == (200, B)
    assert [url for _, url, _, _ in upstreams.calls if url.startswith(A)] == [A + "/health"]


ON_HOST = "http://127.0.0.1:8774"


def test_a_server_on_this_host_gets_the_path_not_an_upload(tmp_path, monkeypatch):
    clip = tmp_path / "film.mkv"
    clip.write_bytes(b"MKV....")
    monkeypatch.setattr(router, "audio_bytes", lambda path: pytest.fail("a local server reads the file itself"))
    upstreams = Upstreams({ON_HOST: OK, B: OK})
    body = json.dumps({"path": str(clip), "language": "en", "segments": True}).encode()
    status, _, payload, used = router.handle_post("/transcribe", "", body, "application/json", [ON_HOST, B], opener=upstreams)
    assert (status, used) == (200, ON_HOST)
    sent = next(call for call in upstreams.calls if call[1].endswith("/transcribe"))
    assert sent[2] == body and sent[3]["Content-type"] == "application/json"


def test_the_upload_is_built_when_the_request_moves_to_a_remote_server(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF....")
    upstreams = Upstreams({ON_HOST: (503, b'{"error": "GPU busy"}'), B: OK})
    body = json.dumps({"path": str(clip), "language": "en"}).encode()
    status, _, _, used = router.handle_post("/transcribe", "", body, "application/json", [ON_HOST, B], opener=upstreams)
    assert (status, used) == (200, B)
    sent = [call for call in upstreams.calls if call[1].endswith("/transcribe")]
    assert [call[0] for call in sent] == [B]
    assert sent[0][3]["Content-type"].startswith("multipart/form-data") and b"RIFF...." in sent[0][2]
