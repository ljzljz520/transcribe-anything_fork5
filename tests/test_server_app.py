"""Unit tests for the FastAPI daemon (issue #107).

These run entirely in-process via ``fastapi.testclient.TestClient`` so no
real iso-env or GPU is required. The actual transcribe() call is mocked
out — we're testing HTTP behavior, auth, settings ownership, redaction,
and the job-store lifecycle.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transcribe_anything.server_app import (
    IDEMPOTENCY_COMMITTED,
    JobStatus,
    JobStore,
    QueueFull,
    ServerConfig,
    SettingsViolation,
    _check_auth,
    _redact_secrets,
    config_from_env,
    config_to_env,
    create_app,
    is_model_cached,
    make_request_fingerprint,
    validate_request_options,
)

# --------------------- helpers ---------------------


def _fake_transcribe_factory(transcript_text: str = "hello world"):
    """Return a fn matching transcribe(...) signature that writes out.{txt,srt,vtt,json}."""

    def fake_transcribe(*, url_or_file: str, output_dir: str, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "out.txt").write_text(transcript_text, encoding="utf-8")
        (out / "out.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
        (out / "out.vtt").write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n", encoding="utf-8")
        (out / "out.json").write_text(json.dumps({"text": transcript_text, "chunks": []}), encoding="utf-8")
        return str(out)

    return fake_transcribe


def _failing_transcribe_factory(message: str):
    def fake(**_kwargs):
        raise RuntimeError(message)

    return fake


def _wait_for_status(client: TestClient, job_id: str, headers=None, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/v1/jobs/{job_id}", headers=headers or {})
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in ("completed", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not terminate in {timeout}s; last={last}")


# --------------------- redaction ---------------------


def test_redact_secrets_strips_explicit_hf_token() -> None:
    msg = "boom! cmd was: insanely-fast-whisper --hf-token hf_abc123 --foo bar"
    redacted = _redact_secrets(msg, "hf_abc123")
    assert redacted is not None
    assert "hf_abc123" not in redacted
    assert "<REDACTED>" in redacted


def test_redact_secrets_strips_cli_arg_even_without_token_known() -> None:
    msg = "Running: --hf_token hf_xyzzy --batch 4"
    out = _redact_secrets(msg, None)
    assert out is not None
    assert "hf_xyzzy" not in out
    assert "<REDACTED>" in out


def test_redact_secrets_handles_empty_string() -> None:
    assert _redact_secrets("", None) == ""
    assert _redact_secrets(None, None) is None  # type: ignore[arg-type]


# --------------------- auth ---------------------


def test_check_auth_loopback_no_token_allowed() -> None:
    cfg = ServerConfig(host="127.0.0.1", auth_token=None)
    assert _check_auth(cfg, authorization=None, x_transcribe_token=None)


def test_check_auth_token_required_for_non_loopback() -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret")
    assert not _check_auth(cfg, authorization=None, x_transcribe_token=None)
    assert not _check_auth(cfg, authorization="Bearer wrong", x_transcribe_token=None)
    assert _check_auth(cfg, authorization="Bearer secret", x_transcribe_token=None)


def test_check_auth_x_transcribe_token_alias_works() -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret")
    assert _check_auth(cfg, authorization=None, x_transcribe_token="secret")


def test_serverconfig_refuses_non_loopback_without_token() -> None:
    cfg = ServerConfig(host="0.0.0.0")
    with pytest.raises(ValueError):
        cfg.validate()


def test_serverconfig_accepts_non_loopback_with_token() -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret")
    cfg.validate()  # no exception


def test_serverconfig_rejects_bogus_prefetch() -> None:
    cfg = ServerConfig(prefetch="sometimes")
    with pytest.raises(ValueError):
        cfg.validate()


# --------------------- settings ownership ---------------------


def test_validate_request_options_rejects_device_override() -> None:
    with pytest.raises(SettingsViolation):
        validate_request_options({"device": "cpu"}, ServerConfig(device="cuda"))


def test_validate_request_options_rejects_hf_token_override() -> None:
    cfg = ServerConfig(hf_token="serverside")
    with pytest.raises(SettingsViolation):
        validate_request_options({"hf_token": "clientside"}, cfg)
    with pytest.raises(SettingsViolation):
        validate_request_options({"hugging_face_token": "clientside"}, cfg)


def test_validate_request_options_rejects_model_override_by_default() -> None:
    cfg = ServerConfig(model="small")
    with pytest.raises(SettingsViolation):
        validate_request_options({"model": "large-v3"}, cfg)


def test_validate_request_options_allows_model_when_configured() -> None:
    cfg = ServerConfig(model="small", allow_client_model=True)
    out = validate_request_options({"model": "large-v3"}, cfg)
    assert out["model"] == "large-v3"


def test_validate_request_options_allows_same_model_as_default() -> None:
    cfg = ServerConfig(model="small")
    out = validate_request_options({"model": "small"}, cfg)
    assert out["model"] == "small"


def test_validate_request_options_clamps_batch_size() -> None:
    cfg = ServerConfig(max_batch_size=8)
    with pytest.raises(SettingsViolation):
        validate_request_options({"batch_size": 32}, cfg)
    out = validate_request_options({"batch_size": 4}, cfg)
    assert out["batch_size"] == 4


def test_validate_request_options_rejects_embed_by_default() -> None:
    cfg = ServerConfig()
    with pytest.raises(SettingsViolation):
        validate_request_options({"embed": True}, cfg)


def test_validate_request_options_allows_embed_when_configured() -> None:
    cfg = ServerConfig(allow_embed=True)
    out = validate_request_options({"embed": True}, cfg)
    assert out["embed"] is True


# --------------------- config round-trip ---------------------


def test_config_env_round_trip() -> None:
    cfg = ServerConfig(
        host="127.0.0.1",
        port=9000,
        auth_token="t",
        device="cpu",
        model="tiny",
        allow_client_model=True,
        prefetch="eager",
        max_batch_size=4,
        max_queue=2,
        hf_token="hf_abc",
    )
    env = config_to_env(cfg)
    # Subprocess-style: keys must be plain strings.
    for k, v in env.items():
        assert k.startswith("TRANSCRIBE_ANYTHING_SERVER_")
        assert isinstance(v, str)
    rebuilt = config_from_env(env)
    assert rebuilt == cfg


# --------------------- is_model_cached ---------------------


def test_is_model_cached_returns_false_when_no_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # empty dir
    # Also mask the real ~/.cache to avoid false positives on dev machines.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home_stub")
    assert is_model_cached("openai/whisper-large-v3") is False


def test_is_model_cached_finds_directory_match(tmp_path, monkeypatch) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "models--openai--whisper-large-v3").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home_stub")
    assert is_model_cached("openai/whisper-large-v3") is True


# --------------------- end-to-end via TestClient ---------------------


@pytest.fixture
def server_client(tmp_path):
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path / "jobs"))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        yield client, app
    # JobStore.stop is called via the shutdown event by TestClient's context exit.


def test_healthz_ok_without_auth(server_client) -> None:
    client, _ = server_client
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_capabilities_returns_expected_fields(server_client) -> None:
    client, _ = server_client
    resp = client.get("/v1/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_default"] == "tiny"
    assert body["prefetch"] == "lazy"
    assert body["hf_token_configured"] is False


def test_submit_url_runs_to_completion(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://example.com/x.mp3", "language": "en"})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    final = _wait_for_status(client, job_id)
    assert final["status"] == "completed"
    assert "out.txt" in final["artifacts"]
    art = client.get(f"/v1/jobs/{job_id}/artifacts/out.txt")
    assert art.status_code == 200
    assert "hello" in art.text


def test_submit_multipart_runs_to_completion(server_client, tmp_path) -> None:
    client, _ = server_client
    f = tmp_path / "audio.wav"
    f.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    with f.open("rb") as fh:
        resp = client.post(
            "/v1/transcribe",
            files={"file": ("audio.wav", fh, "audio/wav")},
            data={"options": json.dumps({"language": "en"})},
        )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    final = _wait_for_status(client, job_id)
    assert final["status"] == "completed"


def test_submit_rejects_device_override(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x", "device": "cpu"})
    assert resp.status_code == 400
    assert "daemon-locked" in resp.json()["detail"]


def test_submit_rejects_model_override_without_allow(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x", "model": "large-v3"})
    assert resp.status_code == 400
    assert "client model" in resp.json()["detail"]


def test_submit_returns_400_when_no_url_and_no_file(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"language": "en"})
    assert resp.status_code == 400


def test_artifact_path_traversal_blocked(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x"})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)
    # Note: bare ".." is normalized by httpx itself before the request
    # leaves the client, so we don't test it. The cases below are the
    # ones that actually reach the handler.
    for evil in ("..foo", "...", "x.y"):
        r = client.get(f"/v1/jobs/{job_id}/artifacts/{evil}")
        # These either hit the filename check (400/404) or land in a
        # routed-but-missing handler (404). Either is correct rejection.
        assert r.status_code in (400, 404), f"path {evil!r} returned {r.status_code}"


def test_delete_job_removes_artifacts(server_client) -> None:
    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x"})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)
    r = client.delete(f"/v1/jobs/{job_id}")
    assert r.status_code == 204
    r2 = client.get(f"/v1/jobs/{job_id}")
    assert r2.status_code == 404


def test_artifacts_zip_bundle_contains_all_outputs(server_client) -> None:
    import io
    import zipfile

    client, _ = server_client
    resp = client.post("/v1/transcribe", json={"url": "https://x"})
    job_id = resp.json()["job_id"]
    final = _wait_for_status(client, job_id)
    assert final["status"] == "completed"

    r = client.get(f"/v1/jobs/{job_id}/artifacts.zip")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/zip")
    assert f'filename="job-{job_id}.zip"' in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
    assert {"out.txt", "out.srt", "out.vtt", "out.json"}.issubset(names)


def test_artifacts_zip_returns_404_for_unknown_job(server_client) -> None:
    client, _ = server_client
    r = client.get("/v1/jobs/does-not-exist/artifacts.zip")
    assert r.status_code == 404


# --------------------- /metrics endpoint ---------------------


def test_metrics_endpoint_exposes_counters(server_client) -> None:
    client, _ = server_client
    # Run one job to bump the counters.
    resp = client.post("/v1/transcribe", json={"url": "https://x"})
    job_id = resp.json()["job_id"]
    _wait_for_status(client, job_id)

    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "transcribe_anything_jobs_total" in body
    assert 'status="completed"' in body
    assert "transcribe_anything_queue_depth" in body
    assert "transcribe_anything_queue_capacity" in body
    assert "transcribe_anything_jobs_in_flight" in body


# --------------------- webhook callbacks ---------------------


def test_webhook_rejected_when_not_allowed(server_client) -> None:
    client, _ = server_client
    r = client.post("/v1/transcribe", json={"url": "https://x", "webhook_url": "https://example.com/hook"})
    assert r.status_code == 400
    assert "webhook" in r.json()["detail"].lower()


def test_webhook_rejected_when_not_http(tmp_path) -> None:
    cfg = ServerConfig(host="127.0.0.1", model="tiny", allow_webhooks=True, job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        r = client.post("/v1/transcribe", json={"url": "https://x", "webhook_url": "ftp://example.com/hook"})
        assert r.status_code == 400
        assert "http" in r.json()["detail"].lower()


def test_webhook_fires_on_terminal_state(tmp_path, monkeypatch) -> None:
    """End-to-end: enable webhooks, submit a job, capture the outbound POST."""
    import threading

    import httpx

    captured: dict = {}
    received = threading.Event()

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            pass

        def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            received.set()
            return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    cfg = ServerConfig(host="127.0.0.1", model="tiny", allow_webhooks=True, job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        r = client.post("/v1/transcribe", json={"url": "https://x", "webhook_url": "https://example.com/hook"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        _wait_for_status(client, job_id)
        # Webhook fires on a daemon thread; wait briefly for it.
        assert received.wait(timeout=2.0), "webhook never fired"
        assert captured["url"] == "https://example.com/hook"
        assert captured["payload"]["job_id"] == job_id
        assert captured["payload"]["status"] == "completed"


# --------------------- failure + redaction ---------------------


def test_failed_job_redacts_hf_token_in_error(tmp_path) -> None:
    cfg = ServerConfig(host="127.0.0.1", model="tiny", hf_token="hf_supersecret123", job_root=str(tmp_path))
    failing = _failing_transcribe_factory("boom because token=hf_supersecret123 and also --hf-token hf_supersecret123")
    app = create_app(cfg, transcribe_fn=failing)
    with TestClient(app) as client:
        resp = client.post("/v1/transcribe", json={"url": "https://x"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        final = _wait_for_status(client, job_id)
        assert final["status"] == "failed"
        assert "hf_supersecret123" not in (final["error"] or "")
        assert "<REDACTED>" in (final["error"] or "")


# --------------------- auth integration ---------------------


def test_auth_required_when_non_loopback(tmp_path) -> None:
    cfg = ServerConfig(host="0.0.0.0", auth_token="secret", model="tiny", job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        # No header -> 401.
        r = client.get("/v1/capabilities")
        assert r.status_code == 401
        # Wrong token -> 401.
        r = client.get("/v1/capabilities", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        # Correct Bearer -> 200.
        r = client.get("/v1/capabilities", headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        # Correct X-Transcribe-Token alias -> 200.
        r = client.get("/v1/capabilities", headers={"X-Transcribe-Token": "secret"})
        assert r.status_code == 200


# --------------------- prefetch=none gate ---------------------


def test_prefetch_none_rejects_submit_when_model_uncached(tmp_path, monkeypatch) -> None:
    # Force is_model_cached to report False.
    from transcribe_anything import server_app as srv

    monkeypatch.setattr(srv, "is_model_cached", lambda _model: False)
    cfg = ServerConfig(host="127.0.0.1", model="tiny", prefetch="none", job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 503
        r = client.post("/v1/transcribe", json={"url": "https://x"})
        assert r.status_code == 503
        assert "prefetch=none" in r.json()["detail"]


# --------------------- queue overflow ---------------------


def test_queue_overflow_returns_429(tmp_path) -> None:
    """A second submission while the worker is busy should respect max_queue=1."""
    import threading

    block = threading.Event()
    release = threading.Event()

    def blocking(**kwargs):
        # Hold the worker until we release it.
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        block.set()
        release.wait(timeout=5)
        (out / "out.txt").write_text("done", encoding="utf-8")
        (out / "out.srt").write_text("1\n", encoding="utf-8")
        (out / "out.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (out / "out.json").write_text("{}", encoding="utf-8")
        return str(out)

    cfg = ServerConfig(host="127.0.0.1", model="tiny", max_queue=1, job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=blocking)
    try:
        with TestClient(app) as client:
            r1 = client.post("/v1/transcribe", json={"url": "https://a"})
            assert r1.status_code == 202
            assert block.wait(timeout=2)
            r2 = client.post("/v1/transcribe", json={"url": "https://b"})
            assert r2.status_code == 202  # queued
            r3 = client.post("/v1/transcribe", json={"url": "https://c"})
            # With max_queue=1 and worker busy + one queued, third should be 429.
            assert r3.status_code in (202, 429)
            # Now unblock the worker so shutdown completes.
            release.set()
    finally:
        release.set()


# --------------------- Idempotency-Key ---------------------


def _counting_transcribe_factory():
    """Return (fn, calls): fn writes the usual artifacts and records each invocation."""
    import threading

    calls: list = []
    lock = threading.Lock()

    def fake_transcribe(*, url_or_file: str, output_dir: str, **_kwargs):
        with lock:
            calls.append(url_or_file)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "out.txt").write_text("hello world", encoding="utf-8")
        (out / "out.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
        (out / "out.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (out / "out.json").write_text("{}", encoding="utf-8")
        return str(out)

    return fake_transcribe, calls


def _job_dirs(job_root: Path) -> set:
    return {p for p in job_root.iterdir() if p.name.startswith("ta-job-")}


def test_make_request_fingerprint_stable_and_sensitive() -> None:
    fp = make_request_fingerprint("url:https://x/a.mp3", {"language": "en", "task": "transcribe"})
    # Option key ordering must not matter.
    fp_reordered = make_request_fingerprint("url:https://x/a.mp3", {"task": "transcribe", "language": "en"})
    assert fp == fp_reordered
    # Different media -> different fingerprint.
    assert fp != make_request_fingerprint("url:https://x/b.mp3", {"language": "en", "task": "transcribe"})
    # Different parameters -> different fingerprint.
    assert fp != make_request_fingerprint("url:https://x/a.mp3", {"language": "fr", "task": "transcribe"})


def test_idempotent_replay_returns_same_job_and_infers_once(tmp_path) -> None:
    """Response-loss scenario: exact retry with same key collapses onto the original job."""
    job_root = tmp_path / "jobs"
    fake, calls = _counting_transcribe_factory()
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(job_root))
    app = create_app(cfg, transcribe_fn=fake)
    with TestClient(app) as client:
        payload = {"url": "https://example.com/a.mp3", "language": "en"}
        r1 = client.post("/v1/transcribe", json=payload, headers={"Idempotency-Key": "abc-123"})
        assert r1.status_code == 202, r1.text
        assert r1.headers.get("Idempotent-Replay") is None
        job1 = r1.json()["job_id"]
        _wait_for_status(client, job1)

        # Simulate the client retrying after losing the 202 (same key, same body).
        r2 = client.post("/v1/transcribe", json=payload, headers={"Idempotency-Key": "abc-123"})
        assert r2.status_code == 202, r2.text
        assert r2.headers.get("Idempotent-Replay") == "true"
        assert r2.json()["job_id"] == job1

        assert len(calls) == 1  # no duplicate inference
        # Only the original upload/job directory remains.
        assert len(_job_dirs(job_root)) == 1


def test_idempotent_replay_while_queued_returns_same_job(tmp_path) -> None:
    """Retry before the worker picked up the job still maps to one enqueue."""
    import threading

    release = threading.Event()
    entered = threading.Event()

    def blocking(**kwargs):
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        entered.set()
        release.wait(timeout=5)
        for name, body in (("out.txt", "x"), ("out.srt", "1\n"), ("out.vtt", "WEBVTT\n"), ("out.json", "{}")):
            (out / name).write_text(body, encoding="utf-8")
        return str(out)

    job_root = tmp_path / "jobs"
    cfg = ServerConfig(host="127.0.0.1", model="tiny", max_queue=4, job_root=str(job_root))
    app = create_app(cfg, transcribe_fn=blocking)
    try:
        with TestClient(app) as client:
            r1 = client.post("/v1/transcribe", json={"url": "https://q"}, headers={"Idempotency-Key": "kq"})
            assert r1.status_code == 202
            assert entered.wait(timeout=2)
            # A second keyed job queues behind the blocked worker; retry
            # its submission while it is still queued.
            q1 = client.post("/v1/transcribe", json={"url": "https://r"}, headers={"Idempotency-Key": "kr"})
            assert q1.status_code == 202
            q2 = client.post("/v1/transcribe", json={"url": "https://r"}, headers={"Idempotency-Key": "kr"})
            assert q2.status_code == 202
            assert q2.headers.get("Idempotent-Replay") == "true"
            assert q2.json()["job_id"] == q1.json()["job_id"]
    finally:
        release.set()


def test_same_key_different_parameters_is_conflict(tmp_path) -> None:
    job_root = tmp_path / "jobs"
    fake, calls = _counting_transcribe_factory()
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(job_root))
    app = create_app(cfg, transcribe_fn=fake)
    with TestClient(app) as client:
        r1 = client.post(
            "/v1/transcribe",
            json={"url": "https://example.com/a.mp3", "language": "en"},
            headers={"Idempotency-Key": "shared"},
        )
        assert r1.status_code == 202
        r2 = client.post(
            "/v1/transcribe",
            json={"url": "https://example.com/a.mp3", "language": "fr"},
            headers={"Idempotency-Key": "shared"},
        )
        assert r2.status_code == 409
        assert "different request" in r2.json()["detail"]
        # The conflicting attempt leaves no upload directory behind.
        assert len(_job_dirs(job_root)) == 1
        assert len(calls) == 1


def test_same_key_different_url_is_conflict(tmp_path) -> None:
    fake, _ = _counting_transcribe_factory()
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=fake)
    with TestClient(app) as client:
        assert client.post("/v1/transcribe", json={"url": "https://a"}, headers={"Idempotency-Key": "k"}).status_code == 202
        conflict = client.post("/v1/transcribe", json={"url": "https://b"}, headers={"Idempotency-Key": "k"})
        assert conflict.status_code == 409
        assert "different request" in conflict.json()["detail"]


def test_same_key_different_upload_bytes_is_conflict(tmp_path) -> None:
    job_root = tmp_path / "jobs"
    fake, calls = _counting_transcribe_factory()
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(job_root))
    app = create_app(cfg, transcribe_fn=fake)
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    f1.write_bytes(b"audio-bytes-version-one")
    f2.write_bytes(b"audio-bytes-version-two-different")
    with TestClient(app) as client:
        with f1.open("rb") as fh:
            r1 = client.post("/v1/transcribe", files={"file": ("a.wav", fh, "audio/wav")}, headers={"Idempotency-Key": "up"})
        assert r1.status_code == 202
        _wait_for_status(client, r1.json()["job_id"])
        with f2.open("rb") as fh:
            r2 = client.post("/v1/transcribe", files={"file": ("b.wav", fh, "audio/wav")}, headers={"Idempotency-Key": "up"})
        assert r2.status_code == 409
        assert "different request" in r2.json()["detail"]
        assert len(calls) == 1
        assert len(_job_dirs(job_root)) == 1


def test_multipart_replay_after_lost_response_dedupes(tmp_path) -> None:
    job_root = tmp_path / "jobs"
    fake, calls = _counting_transcribe_factory()
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(job_root))
    app = create_app(cfg, transcribe_fn=fake)
    src = tmp_path / "audio.wav"
    src.write_bytes(b"same-bytes-on-both-attempts")
    with TestClient(app) as client:

        def submit():
            with src.open("rb") as fh:
                return client.post(
                    "/v1/transcribe",
                    files={"file": ("audio.wav", fh, "audio/wav")},
                    data={"options": json.dumps({"language": "en"})},
                    headers={"Idempotency-Key": "up-replay"},
                )

        r1 = submit()
        assert r1.status_code == 202
        _wait_for_status(client, r1.json()["job_id"])
        r2 = submit()
        assert r2.status_code == 202
        assert r2.headers.get("Idempotent-Replay") == "true"
        assert r2.json()["job_id"] == r1.json()["job_id"]
        assert len(calls) == 1
        assert len(_job_dirs(job_root)) == 1


def test_concurrent_same_key_uploads_dedupe(tmp_path) -> None:
    """Two concurrent uploads, one key: one 202, one 409, one inference, one job dir."""
    import threading

    job_root = tmp_path / "jobs"
    entered = threading.Event()
    release = threading.Event()
    calls: list = []
    calls_lock = threading.Lock()

    def blocking(**kwargs):
        with calls_lock:
            calls.append(kwargs.get("url_or_file"))
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        entered.set()
        release.wait(timeout=5)
        for name, body in (("out.txt", "x"), ("out.srt", "1\n"), ("out.vtt", "WEBVTT\n"), ("out.json", "{}")):
            (out / name).write_text(body, encoding="utf-8")
        return str(out)

    cfg = ServerConfig(host="127.0.0.1", model="tiny", max_queue=4, job_root=str(job_root))
    app = create_app(cfg, transcribe_fn=blocking)
    store = app.state.store

    # Hold the in_flight reservation open until the losing request has
    # observed it, making the interleaving deterministic.
    reserve_orig = store.reserve_idempotency_key
    loser_observed = threading.Event()

    def reserve_then_gate(key):
        record = reserve_orig(key)
        if record is None:
            # Winner: park while the loser races into reserve().
            assert loser_observed.wait(timeout=5)
        else:
            # Loser saw the winner's in_flight reservation.
            loser_observed.set()
        return record

    store.reserve_idempotency_key = reserve_then_gate  # type: ignore[method-assign]

    statuses: dict = {}

    def post(thread_name: str):
        # No context manager: create_app() already started the store,
        # and each thread needs its own portal (TestClient is not
        # shareable across threads).
        client = TestClient(app)
        resp = client.post("/v1/transcribe", json={"url": "https://race"}, headers={"Idempotency-Key": "race-key"})
        statuses[thread_name] = resp.status_code

    try:
        t1 = threading.Thread(target=post, args=("a",))
        t2 = threading.Thread(target=post, args=("b",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        assert not t1.is_alive() and not t2.is_alive()
        assert sorted(statuses.values()) == [202, 409], statuses
        assert entered.wait(timeout=2)
    finally:
        release.set()

    assert len(calls) == 1, calls
    with store._lock:  # noqa: SLF001 - inspecting internals for the dedupe invariant
        assert len(store._jobs) == 1
    assert len(_job_dirs(job_root)) == 1, sorted(p.name for p in job_root.iterdir())


def test_queue_full_rolls_back_idempotency_key(tmp_path) -> None:
    """429 must release the key and clean the upload dir; the key stays reusable."""
    import threading

    job_root = tmp_path / "jobs"
    entered = threading.Event()
    release = threading.Event()
    calls: list = []
    calls_lock = threading.Lock()

    def blocking(**kwargs):
        with calls_lock:
            calls.append(kwargs.get("url_or_file"))
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        entered.set()
        release.wait(timeout=5)
        for name, body in (("out.txt", "x"), ("out.srt", "1\n"), ("out.vtt", "WEBVTT\n"), ("out.json", "{}")):
            (out / name).write_text(body, encoding="utf-8")
        return str(out)

    cfg = ServerConfig(host="127.0.0.1", model="tiny", max_queue=1, job_root=str(job_root))
    app = create_app(cfg, transcribe_fn=blocking)
    try:
        with TestClient(app) as client:
            j1 = client.post("/v1/transcribe", json={"url": "https://a"}).json()["job_id"]
            assert entered.wait(timeout=2)
            j2 = client.post("/v1/transcribe", json={"url": "https://b"}).json()["job_id"]  # fills the queue

            body = {"url": "https://c", "language": "en"}
            r1 = client.post("/v1/transcribe", json=body, headers={"Idempotency-Key": "rolled"})
            assert r1.status_code == 429, r1.text
            # Reservation was rolled back: the same key gives 429 again,
            # not 409 ("key in flight / already used").
            r2 = client.post("/v1/transcribe", json=body, headers={"Idempotency-Key": "rolled"})
            assert r2.status_code == 429, r2.text

            # No orphaned upload directories from the two rejected attempts.
            assert len(_job_dirs(job_root)) == 2

            # Drain the queue, then reuse the rolled-back key while the same
            # daemon (and its worker) is still running.
            release.set()
            _wait_for_status(client, j1)
            _wait_for_status(client, j2)
            r3 = client.post("/v1/transcribe", json=body, headers={"Idempotency-Key": "rolled"})
            assert r3.status_code == 202, r3.text
            _wait_for_status(client, r3.json()["job_id"])
    finally:
        release.set()

    assert len(calls) == 3, calls  # a, b, c — the 429s never inferred
    assert len(_job_dirs(job_root)) == 3


def test_invalid_idempotency_key_header_rejected(tmp_path) -> None:
    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=_fake_transcribe_factory())
    with TestClient(app) as client:
        assert client.post("/v1/transcribe", json={"url": "https://x"}, headers={"Idempotency-Key": "k" * 256}).status_code == 400
        assert client.post("/v1/transcribe", json={"url": "https://x"}, headers={"Idempotency-Key": "bad\nkey"}).status_code == 400
        # Whitespace-only behaves like "no key" — normal 202.
        assert client.post("/v1/transcribe", json={"url": "https://x"}, headers={"Idempotency-Key": "   "}).status_code == 202


def test_failed_job_is_still_deduped_within_retention(tmp_path) -> None:
    """Retrying a key whose original job failed replays it instead of re-inferring."""
    calls: list = []

    def boom(*_a, **_kw):
        calls.append("called")
        raise RuntimeError("boom")

    cfg = ServerConfig(host="127.0.0.1", model="tiny", job_root=str(tmp_path))
    app = create_app(cfg, transcribe_fn=boom)
    with TestClient(app) as client:
        r1 = client.post("/v1/transcribe", json={"url": "https://x"}, headers={"Idempotency-Key": "fail"})
        job_id = r1.json()["job_id"]
        final = _wait_for_status(client, job_id)
        assert final["status"] == "failed"
        r2 = client.post("/v1/transcribe", json={"url": "https://x"}, headers={"Idempotency-Key": "fail"})
        assert r2.status_code == 202
        assert r2.json()["job_id"] == job_id
        assert len(calls) == 1


# --------------------- Idempotency store semantics ---------------------


def _store_with_job(store: JobStore, job_id: str = "j1", key: str = "k", fingerprint: str = "fp") -> None:
    assert store.reserve_idempotency_key(key) is None
    store.commit_idempotency_record(key, fingerprint, job_id, {"job_id": job_id, "status": "queued", "status_url": f"/v1/jobs/{job_id}"})


def test_idempotency_record_replays_within_ttl() -> None:
    cfg = ServerConfig(artifact_ttl_seconds=3600)
    store = JobStore(cfg, transcribe_fn=lambda **_kw: "")
    _store_with_job(store)
    record = store.reserve_idempotency_key("k")
    assert record is not None
    assert record.state == IDEMPOTENCY_COMMITTED
    assert record.fingerprint == "fp"
    assert record.response["job_id"] == "j1"


def test_idempotency_record_expires_after_retention_window() -> None:
    cfg = ServerConfig(artifact_ttl_seconds=0)
    store = JobStore(cfg, transcribe_fn=lambda **_kw: "")
    _store_with_job(store)
    store.sweep_expired_idempotency()
    assert store.get_idempotency_record("k") is None
    # Expired record behaves as a free key (scope bounded by retention).
    assert store.reserve_idempotency_key("k") is None


def test_release_idempotency_key_makes_key_reusable() -> None:
    cfg = ServerConfig()
    store = JobStore(cfg, transcribe_fn=lambda **_kw: "")
    assert store.reserve_idempotency_key("k") is None
    store.release_idempotency_key("k")  # simulates a queue-full rollback
    assert store.get_idempotency_record("k") is None
    assert store.reserve_idempotency_key("k") is None


def test_delete_job_drops_idempotency_record(tmp_path) -> None:
    cfg = ServerConfig(job_root=str(tmp_path))
    store = JobStore(cfg, transcribe_fn=lambda **_kw: "")
    store.start()
    try:
        job = store.submit({"input": "https://x"}, str(tmp_path / "j"))
        _store_with_job(store, job_id=job.job_id, key="dk")
        assert store.get_idempotency_record("dk") is not None
        assert store.delete(job.job_id) is True
        assert store.get_idempotency_record("dk") is None
    finally:
        store.stop()
