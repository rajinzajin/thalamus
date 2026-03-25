from __future__ import annotations
"""
HTTP/2 client for Cursor API (api2.cursor.sh).

Connects to the Cloudflare IP while preserving correct TLS SNI (api2.cursor.sh).
"""

import os
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import httpcore

from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("h2-client", "DEBUG")

CURSOR_CLOUDFLARE_IP: str = os.environ.get("CURSOR_CLOUDFLARE_IP", "104.18.19.125")
CURSOR_API_HOST: str = "api2.cursor.sh"

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def _build_ssl_context() -> ssl.SSLContext:
    """TLS context with a reliable CA bundle.

    Stock Python on macOS often ships without a usable default CA store, which
    yields CERTIFICATE_VERIFY_FAILED against api2.cursor.sh. Prefer certifi's
    bundle; override with SSL_CERT_FILE for a custom CA (e.g. corporate proxy).
    """
    cafile = os.environ.get("SSL_CERT_FILE") or None
    if not cafile:
        try:
            import certifi

            cafile = certifi.where()
        except ImportError:
            cafile = None
    ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    ctx.set_alpn_protocols(["h2"])
    return ctx


class _CloudflareOverrideBackend(httpcore.AsyncNetworkBackend):
    """Redirects TCP connections for api2.cursor.sh to the Cloudflare IP
    while keeping the original hostname for TLS SNI."""

    def __init__(self) -> None:
        from httpcore._backends.auto import AutoBackend
        self._inner = AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object | None = None,
    ) -> httpcore.AsyncNetworkStream:
        target = CURSOR_CLOUDFLARE_IP if host == CURSOR_API_HOST else host
        if target != host:
            logger.debug(f"DNS override: {host} -> {target}:{port}")
        return await self._inner.connect_tcp(
            target, port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: object | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._inner.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _build_client() -> httpx.AsyncClient:
    """Build an httpx AsyncClient that routes api2.cursor.sh to the Cloudflare IP.

    The URL host stays as api2.cursor.sh so httpx/httpcore use it for TLS SNI.
    The custom network backend intercepts the TCP connect and redirects to the IP.
    """
    ssl_ctx = _build_ssl_context()
    backend = _CloudflareOverrideBackend()

    pool = httpcore.AsyncConnectionPool(
        ssl_context=ssl_ctx,
        http2=True,
        max_connections=10,
        max_keepalive_connections=5,
        network_backend=backend,
    )

    transport = httpx.AsyncHTTPTransport(http2=True, verify=ssl_ctx)
    transport._pool = pool  # noqa: SLF001

    return httpx.AsyncClient(
        transport=transport,
        base_url=f"https://{CURSOR_API_HOST}",
        timeout=_TIMEOUT,
    )


@asynccontextmanager
async def open_streaming_h2_request(
    path: str,
    headers: dict[str, str],
    body: bytes,
) -> AsyncIterator[AsyncIterator[bytes]]:
    """Open a server-streaming HTTP/2 POST to api2.cursor.sh.

    The caller can break out of the iterator early (e.g. after receiving a tool
    call). The context manager will forcefully close the connection rather than
    waiting for the server to finish the stream.
    """
    client = _build_client()
    response_ctx = client.stream("POST", path, headers=headers, content=body)
    response = await response_ctx.__aenter__()
    try:
        logger.debug(
            f"Streaming response started: status={response.status_code} path={path}"
        )
        yield response.aiter_bytes()
    finally:
        try:
            await response.aclose()
        except Exception:
            pass
        try:
            await response_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass


async def send_unary_h2_request(
    path: str,
    headers: dict[str, str],
    body: bytes,
) -> dict:
    """Send a unary (non-streaming) HTTP/2 POST and return the full response."""
    async with _build_client() as client:
        response = await client.post(path, headers=headers, content=body)
        logger.debug(
            f"Unary response: status={response.status_code} path={path} size={len(response.content)}"
        )
        return {"status": response.status_code, "buffer": response.content}
