import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from claude_code.normalizers import normalize_anthropic
from claude_code.pipeline import run_pipeline
from core.token_manager import capture_token_from_request
from utils.structured_logging import ThalamusStructuredLogger
from utils.thalamus_api_logger import (
    log_thalamus_request,
    log_thalamus_response,
    log_thalamus_api_call,
    parse_sse_to_events,
)

logger = ThalamusStructuredLogger.get_logger("anthropic-messages", "DEBUG")
router = APIRouter()

ENDPOINT_MESSAGES = "/v1/messages"


def _headers_summary(request: Request) -> dict:
    """Build safe headers summary for logging (mask sensitive values)."""
    try:
        h = dict(request.headers)
        summary = {}
        for k, v in h.items():
            kl = k.lower()
            if kl in ("authorization", "x-api-key"):
                summary[k] = "***" if v else "(absent)"
            elif kl in ("content-type", "content-length"):
                summary[k] = v
        return summary
    except Exception:
        return {}


@router.post(ENDPOINT_MESSAGES)
async def create_message(request: Request):
    """Handle Anthropic Messages API requests."""
    start_time = time.time()

    auth_token = request.headers.get("x-api-key", "") or request.headers.get("authorization", "")

    capture_token_from_request(request.headers.get("authorization", ""))

    try:
        payload = await request.json()
    except Exception as e:
        err_body = {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}
        request_id = f"req_{time.time_ns() // 1000000}"
        try:
            req_path = log_thalamus_request(
                request_id, ENDPOINT_MESSAGES, "POST",
                {"_parse_error": str(e)},
                _headers_summary(request),
            )
            res_path = log_thalamus_response(request_id, ENDPOINT_MESSAGES, 400, err_body, latency_ms=0)
            log_thalamus_api_call(request_id, ENDPOINT_MESSAGES, "POST", 400, 0, req_path, res_path, error=str(e))
        except Exception:
            pass
        return JSONResponse(status_code=400, content=err_body)

    request_id = f"req_{time.time_ns() // 1000000}"

    try:
        req_path = log_thalamus_request(
            request_id, ENDPOINT_MESSAGES, "POST",
            payload,
            _headers_summary(request),
        )
    except Exception:
        req_path = ""

    logger.info(f"[{request_id}] POST /v1/messages model={payload.get('model', '?')} stream={payload.get('stream', False)} tools={len(payload.get('tools', []))}")

    req = normalize_anthropic(payload)

    result = await run_pipeline(
        req=req,
        request_id=request_id,
        auth_token=auth_token,
    )

    latency_ms = int((time.time() - start_time) * 1000)

    if not result.get("ok"):
        status = result.get("status", 500)
        body = result.get("body", {"type": "error", "error": {"type": "api_error", "message": "Internal error"}})
        try:
            res_path = log_thalamus_response(request_id, ENDPOINT_MESSAGES, status, body, latency_ms=latency_ms)
            log_thalamus_api_call(request_id, ENDPOINT_MESSAGES, "POST", status, latency_ms, req_path, res_path, error=body.get("error", {}).get("message"))
        except Exception:
            pass
        return JSONResponse(status_code=status, content=body)

    if result.get("stream"):
        stream_handler = result.get("stream_handler")
        if stream_handler:
            sse_chunks: list[str] = []

            async def wrapped_stream():
                nonlocal sse_chunks
                try:
                    async for chunk in stream_handler():
                        sse_chunks.append(chunk)
                        yield chunk
                finally:
                    try:
                        raw_sse = "".join(sse_chunks)
                        events = parse_sse_to_events(raw_sse)
                        res_path = log_thalamus_response(
                            request_id, ENDPOINT_MESSAGES, 200, events,
                            latency_ms=int((time.time() - start_time) * 1000),
                        )
                        log_thalamus_api_call(
                            request_id, ENDPOINT_MESSAGES, "POST", 200,
                            int((time.time() - start_time) * 1000),
                            req_path, res_path,
                        )
                    except Exception:
                        pass

            return StreamingResponse(
                wrapped_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    # Match OpenAI route: close after stream so clients see EOF right after
                    # the terminal SSE event; keep-alive can delay completion in some runtimes.
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                    "X-Thalamus-Request-Id": request_id,
                },
            )
        err_body = {"type": "error", "error": {"type": "api_error", "message": "No stream handler"}}
        try:
            res_path = log_thalamus_response(request_id, ENDPOINT_MESSAGES, 500, err_body, latency_ms=latency_ms)
            log_thalamus_api_call(request_id, ENDPOINT_MESSAGES, "POST", 500, latency_ms, req_path, res_path, error="No stream handler")
        except Exception:
            pass
        return JSONResponse(status_code=500, content=err_body)

    body = result.get("body", {})
    try:
        res_path = log_thalamus_response(request_id, ENDPOINT_MESSAGES, 200, body, latency_ms=latency_ms)
        log_thalamus_api_call(request_id, ENDPOINT_MESSAGES, "POST", 200, latency_ms, req_path, res_path)
    except Exception:
        pass
    return JSONResponse(content=body)


ENDPOINT_COUNT_TOKENS = "/v1/messages/count_tokens"


@router.post(ENDPOINT_COUNT_TOKENS)
async def count_tokens(request: Request):
    """Estimate token count for a messages request."""
    start_time = time.time()
    request_id = f"ct_{time.time_ns() // 1000000}"

    try:
        payload = await request.json()
    except Exception:
        err_body = {"input_tokens": 0}
        try:
            req_path = log_thalamus_request(request_id, ENDPOINT_COUNT_TOKENS, "POST", {"_parse_error": "Invalid JSON"}, _headers_summary(request))
            res_path = log_thalamus_response(request_id, ENDPOINT_COUNT_TOKENS, 400, err_body, latency_ms=0)
            log_thalamus_api_call(request_id, ENDPOINT_COUNT_TOKENS, "POST", 400, 0, req_path, res_path, error="Invalid JSON")
        except Exception:
            pass
        return JSONResponse(status_code=400, content=err_body)

    try:
        req_path = log_thalamus_request(
            request_id, ENDPOINT_COUNT_TOKENS, "POST",
            payload,
            _headers_summary(request),
        )
    except Exception:
        req_path = ""

    total_chars = 0
    if payload.get("system"):
        if isinstance(payload["system"], str):
            total_chars += len(payload["system"])
        elif isinstance(payload["system"], list):
            for block in payload["system"]:
                total_chars += len(block.get("text", ""))

    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                total_chars += len(block.get("text", ""))

    estimated_tokens = max(1, total_chars // 4)
    response_body = {"input_tokens": estimated_tokens}
    latency_ms = int((time.time() - start_time) * 1000)

    try:
        res_path = log_thalamus_response(request_id, ENDPOINT_COUNT_TOKENS, 200, response_body, latency_ms=latency_ms)
        log_thalamus_api_call(request_id, ENDPOINT_COUNT_TOKENS, "POST", 200, latency_ms, req_path, res_path)
    except Exception:
        pass

    return JSONResponse(content=response_body)
