"""Local HTTP shim that makes OpenRouter look like a full Anthropic API to Claude Code.

Claude Code (the binary under claude-agent-sdk) makes auxiliary calls to endpoints
like /v1/organizations/me, /v1/me, and /v1/models/<id> that OpenRouter does not
implement (all 404). When Claude CLI gets a 404 from these pre-flight checks it
reports the cryptic 'There's an issue with the selected model' error before
ever calling /v1/messages.

This shim:
  - Listens on 127.0.0.1:<free port>
  - Stubs /v1/me, /v1/organizations*, /v1/models* with plausible 200 responses
  - Streams /v1/messages straight through to OpenRouter (rewriting the Bearer header)

Usage:
    from openrouter_anthropic_shim import AnthropicShim
    with AnthropicShim(openrouter_api_key="sk-or-...") as shim:
        os.environ["ANTHROPIC_BASE_URL"] = shim.url  # http://127.0.0.1:NNN
        ...
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
from typing import Any

import httpx
import uvicorn

_DEBUG = bool(os.environ.get("ANTHROPIC_SHIM_DEBUG"))
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _build_app(openrouter_api_key: str) -> Starlette:
    """Construct the Starlette app that fronts OpenRouter."""

    async def stub_me(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "id": "user_openrouter",
                "name": "OpenRouter Proxy User",
                "email_address": "noreply@openrouter.ai",
            }
        )

    async def stub_org(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "data": [
                    {
                        "id": "org_openrouter",
                        "name": "OpenRouter",
                        "billing_type": "credit",
                        "rate_limit_tier": "tier_4",
                    }
                ]
            }
        )

    async def stub_org_billing(_: Request) -> JSONResponse:
        # Claude Code reads this to decide which model tiers are available.
        return JSONResponse(
            {
                "tier": "tier_4",
                "has_payment_method": True,
                "credits": {"remaining": 9999, "total": 9999},
            }
        )

    async def stub_models(_: Request) -> JSONResponse:
        # A list shape the SDK can iterate; the IDs here aren't actually used.
        return JSONResponse({"data": [], "has_more": False, "first_id": None, "last_id": None})

    async def stub_model_by_id(request: Request) -> JSONResponse:
        # Claude Code does GET /v1/models/<id> to confirm availability.
        # Echo a plausible model object.
        model_id = request.path_params["model_id"]
        return JSONResponse(
            {
                "id": model_id,
                "type": "model",
                "display_name": model_id,
                "created_at": "2025-01-01T00:00:00Z",
            }
        )

    async def stub_complete(_: Request) -> JSONResponse:
        # Some clients still call the legacy /v1/complete endpoint.
        return JSONResponse({"error": {"message": "legacy endpoint not supported"}}, status_code=400)

    async def forward_messages(request: Request) -> Response:
        """Forward /v1/messages (the actual model call) to OpenRouter, streaming."""
        body = await request.body()
        if _DEBUG:
            print(f"[shim] /v1/messages POST body={len(body)}B accept={request.headers.get('accept')!r}", flush=True)
        # Whitelist the headers we forward (drop the client's Authorization).
        fwd_headers: dict[str, str] = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": request.headers.get("content-type", "application/json"),
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        }
        beta = request.headers.get("anthropic-beta")
        if beta:
            fwd_headers["anthropic-beta"] = beta
        accept = request.headers.get("accept")
        if accept:
            fwd_headers["Accept"] = accept

        target_url = f"{OPENROUTER_BASE}/messages"

        # Detect SSE / streaming: when the request body has "stream": true, or
        # the Accept header is text/event-stream, we must stream the response back.
        try:
            import json as _json

            is_stream = bool(_json.loads(body or b"{}").get("stream", False))
        except Exception:
            is_stream = "stream" in (accept or "").lower()

        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=20.0))
        try:
            if is_stream:
                # Stream chunks to the client as they arrive.
                req = client.build_request("POST", target_url, content=body, headers=fwd_headers)
                upstream = await client.send(req, stream=True)

                async def gen():
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    finally:
                        await upstream.aclose()
                        await client.aclose()

                return StreamingResponse(
                    gen(),
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "text/event-stream"),
                )
            # Buffered (non-streaming)
            r = await client.post(target_url, content=body, headers=fwd_headers)
            await client.aclose()
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                await client.aclose()
            return JSONResponse(
                {"error": {"type": "proxy_error", "message": str(exc)}},
                status_code=502,
            )

    async def catchall(request: Request) -> Response:
        # Anything we haven't explicitly stubbed: return an empty 200 so the CLI
        # treats it as "feature absent / no-op" rather than "auth/route broken".
        if _DEBUG:
            print(f"[shim] catchall  {request.method} {request.url.path}", flush=True)
        return JSONResponse({})

    routes = [
        Route("/v1/me", stub_me),
        Route("/v1/organizations", stub_org),
        Route("/v1/organizations/me", stub_me),
        Route("/v1/organizations/{org_id}/billing", stub_org_billing),
        Route("/v1/models", stub_models),
        Route("/v1/models/{model_id:path}", stub_model_by_id),
        Route("/v1/complete", stub_complete, methods=["POST"]),
        Route("/v1/messages", forward_messages, methods=["POST"]),
        Route("/{rest:path}", catchall, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ]
    return Starlette(routes=routes)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AnthropicShim:
    """Context manager that runs the shim in a background thread."""

    def __init__(self, openrouter_api_key: str, port: int | None = None) -> None:
        self.port = port or _pick_free_port()
        self.host = "127.0.0.1"
        self._key = openrouter_api_key
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        # Anthropic-SDK style — the SDK appends '/v1/messages' itself.
        return f"http://{self.host}:{self.port}"

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        app = _build_app(self._key)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="anthropic-shim")
        self._thread.start()
        # Wait until uvicorn flips started=True
        import time as _t

        for _ in range(100):
            if self._server.started:
                return
            _t.sleep(0.05)
        raise RuntimeError("AnthropicShim failed to start in time")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "AnthropicShim":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
