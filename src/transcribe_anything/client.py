"""
HTTP client used by the ``--remote`` CLI / Python-API path.

The client mirrors the local :func:`transcribe_anything.api.transcribe`
contract: the caller hands over a URL or file path and (eventually) gets
the same ``out.txt``/``out.srt``/``out.vtt``/``out.json`` artifact layout
in their ``output_dir``.

Uses ``httpx`` (synchronous client). ``httpx`` is in the base package
deps so client mode works out of the box.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import httpx


class RemoteTranscriberError(Exception):
    """Raised when a remote daemon rejects or fails a transcription request."""


# Submission is retried only when the response could have been lost
# (connection/timeout transport errors) or when the daemon could not
# definitively process the request. 4xx (incl. 409 key conflicts, 429
# queue full, 400 validation) are terminal: retrying them blindly would
# either mask a caller bug or hammer a saturated queue.
SUBMIT_MAX_ATTEMPTS = 3
RETRYABLE_SUBMIT_STATUSES = {502, 503, 504}
_RETRY_BACKOFF_BASE_SECONDS = 0.05


def _build_headers(token: Optional[str], idempotency_key: Optional[str] = None) -> dict:
    headers: dict = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _resolve_idempotency_key(idempotency_key: Optional[str]) -> str:
    """Use the caller's key or mint one per logical call.

    A freshly minted key protects the client's own bounded retries: if
    the daemon enqueued the job but the 202 was lost on the wire, the
    retry carries the same token and gets the original job back instead
    of triggering a second inference.
    """
    return idempotency_key or uuid.uuid4().hex


def _submit_with_retries(
    send_attempt: Callable[[], "httpx.Response"],
    *,
    base_url: str,
    max_attempts: int = SUBMIT_MAX_ATTEMPTS,
    sleep: Optional[Callable[[float], None]] = None,
) -> "httpx.Response":
    """Sync bounded-retry wrapper around the POST /v1/transcribe call.

    Every attempt sends the SAME Idempotency-Key (set on the client
    headers by the caller), so a retry after a lost response can never
    enqueue twice.
    """
    sleep_fn = sleep if sleep is not None else time.sleep
    last_response: Optional[httpx.Response] = None
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = send_attempt()
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                raise RemoteTranscriberError(f"lost connection submitting to {base_url} after {max_attempts} attempts: {exc}") from exc
            sleep_fn(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt))
            continue
        last_response = response
        if response.status_code in RETRYABLE_SUBMIT_STATUSES and attempt + 1 < max_attempts:
            sleep_fn(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt))
            continue
        return response
    # Unreachable while max_attempts >= 1; keeps type checkers honest.
    raise RemoteTranscriberError(f"daemon at {base_url} kept failing submission: {last_response or last_exc}")


async def _submit_with_retries_async(
    send_attempt: Callable[[], Any],
    *,
    base_url: str,
    max_attempts: int = SUBMIT_MAX_ATTEMPTS,
    sleep: Optional[Callable[[float], Any]] = None,
) -> "httpx.Response":
    """Async mirror of :func:`_submit_with_retries`."""
    sleep_fn = sleep if sleep is not None else asyncio.sleep
    last_response: Optional[httpx.Response] = None
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = await send_attempt()
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                raise RemoteTranscriberError(f"lost connection submitting to {base_url} after {max_attempts} attempts: {exc}") from exc
            await sleep_fn(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt))
            continue
        last_response = response
        if response.status_code in RETRYABLE_SUBMIT_STATUSES and attempt + 1 < max_attempts:
            await sleep_fn(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt))
            continue
        return response
    raise RemoteTranscriberError(f"daemon at {base_url} kept failing submission: {last_response or last_exc}")


def _collect_options(
    *,
    model: Optional[str],
    task: Optional[str],
    language: Optional[str],
    initial_prompt: Optional[str],
    align: bool,
    align_model: Optional[str],
    other_args: Optional[list[str]],
    embed: bool,
) -> dict:
    options: dict[str, Any] = {}
    if model is not None:
        options["model"] = model
    if task:
        options["task"] = task
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    if align:
        options["align"] = True
    if align_model:
        options["align_model"] = align_model
    if other_args:
        options["other_args"] = list(other_args)
    if embed:
        options["embed"] = True
    return options


def transcribe_remote(
    url_or_file: str,
    *,
    remote: str,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    align: bool = False,
    align_model: Optional[str] = None,
    other_args: Optional[list[str]] = None,
    embed: bool = False,
    poll_interval_seconds: float = 1.0,
    request_timeout_seconds: float = 30.0,
    job_timeout_seconds: float = 60 * 60 * 4,
    idempotency_key: Optional[str] = None,
    submit_max_attempts: int = SUBMIT_MAX_ATTEMPTS,
) -> str:
    """Submit a transcription job to a remote daemon and download artifacts locally.

    Returns the absolute path to the output directory, matching the shape
    of :func:`transcribe_anything.api.transcribe`.

    ``idempotency_key`` is optional: pass a stable token to make a logical
    request safely retryable across processes, or leave it unset and one
    is minted for this call. The same key is reused on every bounded
    submission retry, and the daemon de-duplicates by (key, media,
    parameters): a repeated request returns the original job while a
    same-key request with different media/parameters gets a 409.
    """
    base_url = _normalize_base_url(remote)
    idem_key = _resolve_idempotency_key(idempotency_key)
    headers = _build_headers(token, idem_key)
    options = _collect_options(
        model=model,
        task=task,
        language=language,
        initial_prompt=initial_prompt,
        align=align,
        align_model=align_model,
        other_args=other_args,
        embed=embed,
    )

    # Decide submission mode: URL or multipart file upload.
    is_url = url_or_file.startswith("http://") or url_or_file.startswith("https://") or url_or_file.startswith("ftp://")

    if not is_url:
        local = Path(url_or_file)
        if not local.is_file():
            raise RemoteTranscriberError(f"local file not found: {url_or_file}")

    client = httpx.Client(timeout=request_timeout_seconds, headers=headers)
    try:
        submit_url = f"{base_url}/v1/transcribe"

        def _attempt() -> "httpx.Response":
            # Rebuild the multipart body per attempt: the file handle is
            # consumed (and closed) by the previous POST.
            if is_url:
                payload = {"url": url_or_file, **options}
                return client.post(submit_url, json=payload)
            with local.open("rb") as fh:
                files = {"file": (local.name, fh, "application/octet-stream")}
                data = {"options": json.dumps(options)} if options else {}
                return client.post(submit_url, files=files, data=data)

        # Bounded retry after a lost response; the idempotency key makes
        # a retried enqueue collapse onto the original job.
        resp = _submit_with_retries(_attempt, base_url=base_url, max_attempts=submit_max_attempts)
        if resp.status_code == 409:
            raise RemoteTranscriberError(
                f"daemon at {base_url} rejected submission with 409 idempotency conflict for key {idem_key!r} "
                f"(same key reused with different media/parameters, or a concurrent request is in flight): {resp.text}"
            )
        if resp.status_code >= 400:
            raise RemoteTranscriberError(f"daemon at {base_url} rejected submission: {resp.status_code} {resp.text}")
        body = resp.json()
        job_id = body["job_id"]

        # Poll until terminal.
        deadline = time.time() + job_timeout_seconds
        last_status = None
        while True:
            if time.time() > deadline:
                raise RemoteTranscriberError(f"job {job_id} timed out after {job_timeout_seconds}s")
            jr = client.get(f"{base_url}/v1/jobs/{job_id}")
            if jr.status_code >= 400:
                raise RemoteTranscriberError(f"daemon returned {jr.status_code} fetching job status: {jr.text}")
            job = jr.json()
            status = job.get("status")
            if status != last_status:
                sys.stderr.write(f"remote job {job_id}: {status}\n")
                last_status = status
            if status == "completed":
                break
            if status == "failed":
                raise RemoteTranscriberError(f"daemon job {job_id} failed: {job.get('error')}")
            time.sleep(poll_interval_seconds)

        # Resolve output_dir similarly to api.transcribe's logic.
        if output_dir is None:
            base = Path(url_or_file).name
            stem = Path(base).stem if not is_url else (base or "remote")
            output_dir = f"text_{stem or 'remote'}"
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        artifacts = job.get("artifacts") or []
        if not artifacts:
            raise RemoteTranscriberError(f"daemon job {job_id} reported no artifacts")
        for name in artifacts:
            dest = out_path / name
            with client.stream("GET", f"{base_url}/v1/jobs/{job_id}/artifacts/{name}") as r:
                if r.status_code >= 400:
                    raise RemoteTranscriberError(f"failed to download artifact {name}: {r.status_code} {r.text}")
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
        return str(out_path)
    finally:
        client.close()


async def transcribe_remote_async(
    url_or_file: str,
    *,
    remote: str,
    output_dir: Optional[str] = None,
    token: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    align: bool = False,
    align_model: Optional[str] = None,
    other_args: Optional[list[str]] = None,
    embed: bool = False,
    poll_interval_seconds: float = 1.0,
    request_timeout_seconds: float = 30.0,
    job_timeout_seconds: float = 60 * 60 * 4,
    idempotency_key: Optional[str] = None,
    submit_max_attempts: int = SUBMIT_MAX_ATTEMPTS,
) -> str:
    """Async mirror of :func:`transcribe_remote`.

    Same contract as the sync variant: submit a job, poll until terminal,
    download artifacts into ``output_dir``, return the path. Uses
    ``httpx.AsyncClient`` and ``asyncio.sleep`` so callers can drive
    many remote transcriptions concurrently without burning a thread per
    job.

    Like the sync client, an optional ``idempotency_key`` is reused for
    all bounded submission retries (one is minted per call when unset).
    """
    base_url = _normalize_base_url(remote)
    idem_key = _resolve_idempotency_key(idempotency_key)
    headers = _build_headers(token, idem_key)
    options = _collect_options(
        model=model,
        task=task,
        language=language,
        initial_prompt=initial_prompt,
        align=align,
        align_model=align_model,
        other_args=other_args,
        embed=embed,
    )

    is_url = url_or_file.startswith("http://") or url_or_file.startswith("https://") or url_or_file.startswith("ftp://")

    if not is_url:
        local = Path(url_or_file)
        if not local.is_file():
            raise RemoteTranscriberError(f"local file not found: {url_or_file}")

    async with httpx.AsyncClient(timeout=request_timeout_seconds, headers=headers) as client:
        submit_url = f"{base_url}/v1/transcribe"

        async def _attempt() -> "httpx.Response":
            # Rebuild the multipart body per attempt: the file handle is
            # consumed (and closed) by the previous POST.
            if is_url:
                payload = {"url": url_or_file, **options}
                return await client.post(submit_url, json=payload)
            with local.open("rb") as fh:
                files = {"file": (local.name, fh, "application/octet-stream")}
                data = {"options": json.dumps(options)} if options else {}
                return await client.post(submit_url, files=files, data=data)

        # Bounded retry after a lost response; the idempotency key makes
        # a retried enqueue collapse onto the original job.
        resp = await _submit_with_retries_async(_attempt, base_url=base_url, max_attempts=submit_max_attempts)
        if resp.status_code == 409:
            raise RemoteTranscriberError(
                f"daemon at {base_url} rejected submission with 409 idempotency conflict for key {idem_key!r} "
                f"(same key reused with different media/parameters, or a concurrent request is in flight): {resp.text}"
            )
        if resp.status_code >= 400:
            raise RemoteTranscriberError(f"daemon at {base_url} rejected submission: {resp.status_code} {resp.text}")
        body = resp.json()
        job_id = body["job_id"]

        deadline = time.time() + job_timeout_seconds
        last_status = None
        while True:
            if time.time() > deadline:
                raise RemoteTranscriberError(f"job {job_id} timed out after {job_timeout_seconds}s")
            jr = await client.get(f"{base_url}/v1/jobs/{job_id}")
            if jr.status_code >= 400:
                raise RemoteTranscriberError(f"daemon returned {jr.status_code} fetching job status: {jr.text}")
            job = jr.json()
            status = job.get("status")
            if status != last_status:
                sys.stderr.write(f"remote job {job_id}: {status}\n")
                last_status = status
            if status == "completed":
                break
            if status == "failed":
                raise RemoteTranscriberError(f"daemon job {job_id} failed: {job.get('error')}")
            await asyncio.sleep(poll_interval_seconds)

        if output_dir is None:
            base = Path(url_or_file).name
            stem = Path(base).stem if not is_url else (base or "remote")
            output_dir = f"text_{stem or 'remote'}"
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        artifacts = job.get("artifacts") or []
        if not artifacts:
            raise RemoteTranscriberError(f"daemon job {job_id} reported no artifacts")
        for name in artifacts:
            dest = out_path / name
            async with client.stream("GET", f"{base_url}/v1/jobs/{job_id}/artifacts/{name}") as r:
                if r.status_code >= 400:
                    raise RemoteTranscriberError(f"failed to download artifact {name}: {r.status_code} {await r.aread()!r}")
                with dest.open("wb") as fh:
                    async for chunk in r.aiter_bytes():
                        fh.write(chunk)
        return str(out_path)


def resolve_remote_and_token(
    *,
    remote_arg: Optional[str],
    token_arg: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Combine CLI flags + env vars to determine the active remote URL and token."""
    remote = remote_arg or os.environ.get("TRANSCRIBE_ANYTHING_REMOTE") or None
    token = token_arg or os.environ.get("TRANSCRIBE_ANYTHING_TOKEN") or None
    return remote, token


# ---------------------------------------------------------------- streaming


def _http_to_ws_url(url: str) -> str:
    """Normalize an http(s):// or ws(s):// daemon URL to its ws variant."""
    base = url.rstrip("/")
    if base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    return base


async def stream_remote(
    audio_iter,
    *,
    remote: str,
    token: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    sample_rate: int = 16000,
    on_event=None,
):
    """Stream PCM16-LE chunks from ``audio_iter`` to the daemon's ``WS /v1/stream``.

    ``audio_iter`` may be either a regular iterable / generator or an
    async iterable / async generator of ``bytes`` chunks. Each event
    from the server (``partial`` / ``final`` / ``metrics`` / ``done`` /
    ``error``) is forwarded to ``on_event(event_dict)`` if provided, and
    also yielded so callers can ``async for`` over the stream.

    Raises :class:`RemoteTranscriberError` if the daemon rejects the
    connection or emits an ``error`` frame.
    """
    import json as _json

    try:
        from websockets.asyncio.client import (
            connect as _ws_connect,  # type: ignore[import-not-found]
        )
    except ImportError as exc:  # pragma: no cover - exercised only without websockets
        raise RemoteTranscriberError("websockets is required for --stream-in: pip install websockets") from exc

    ws_url = _http_to_ws_url(remote) + "/v1/stream"
    headers = []
    if token:
        headers.append(("Authorization", f"Bearer {token}"))

    hello = {
        "type": "hello",
        "model": model or "small.en",
        "language": language,
        "sample_rate": sample_rate,
        "encoding": "pcm16le",
    }

    async with _ws_connect(ws_url, additional_headers=headers or None) as ws:
        await ws.send(_json.dumps(hello))
        first = await ws.recv()
        first_evt = _json.loads(first)
        if first_evt.get("type") == "error":
            raise RemoteTranscriberError(f"daemon rejected stream: {first_evt}")
        if first_evt.get("type") != "ready":
            raise RemoteTranscriberError(f"unexpected first event from daemon: {first_evt}")

        import asyncio as _asyncio

        async def _push_audio():
            if hasattr(audio_iter, "__aiter__"):
                async for chunk in audio_iter:
                    if not chunk:
                        break
                    await ws.send(chunk)
            else:
                for chunk in audio_iter:
                    if not chunk:
                        break
                    await ws.send(chunk)
            # Graceful EOF — daemon flushes outstanding partials then sends `done`.
            await ws.send(_json.dumps({"type": "end_of_input"}))

        pusher = _asyncio.create_task(_push_audio())
        try:
            async for raw in ws:
                evt = _json.loads(raw)
                if on_event is not None:
                    on_event(evt)
                yield evt
                if evt.get("type") == "done":
                    break
                if evt.get("type") == "error":
                    raise RemoteTranscriberError(f"daemon error: {evt}")
        finally:
            pusher.cancel()
            try:
                await pusher
            except (BaseException,):
                pass


def stream_pcm_from_stdin(
    *,
    remote: str,
    token: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    chunk_bytes: int = 3200,  # 100ms @ 16kHz s16le mono
) -> None:
    """Sync entry point for ``transcribe-anything --remote ws://… --stream-in``.

    Reads PCM16-LE bytes from stdin, streams them to the daemon, prints
    transcript text to stdout as ``partial`` / ``final`` events arrive.
    Blocks until the input pipe closes and the daemon sends ``done``.
    """
    import asyncio as _asyncio
    import sys as _sys

    def _stdin_chunks():
        buf = _sys.stdin.buffer
        while True:
            chunk = buf.read(chunk_bytes)
            if not chunk:
                return
            yield chunk

    async def _run():
        last_locked_end = 0  # length of confirmed-final text so far
        agen = stream_remote(
            _stdin_chunks(),
            remote=remote,
            token=token,
            model=model,
            language=language,
        )
        async for evt in agen:
            kind = evt.get("type")
            if kind == "partial":
                _sys.stdout.write(f"\r[partial] {evt.get('text', '')}\033[K")
                _sys.stdout.flush()
            elif kind == "final":
                _sys.stdout.write(f"\r[final]   {evt.get('text', '')}\n")
                _sys.stdout.flush()
                last_locked_end += len(evt.get("text", ""))
            elif kind == "error":
                raise RemoteTranscriberError(f"daemon error: {evt}")
        _sys.stdout.write("\n")
        _sys.stdout.flush()

    _asyncio.run(_run())
