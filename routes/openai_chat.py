import time
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from claude_code.normalizers import normalize_openai
from claude_code.pipeline import run_pipeline
from core.token_manager import capture_token_from_request
from utils.structured_logging import ThalamusStructuredLogger
from utils.thalamus_api_logger import (
    log_thalamus_request,
    log_thalamus_response,
    log_thalamus_api_call,
)

logger = ThalamusStructuredLogger.get_logger("openai-chat", "DEBUG")
router = APIRouter()

ENDPOINT = "/v1/chat/completions"


def _headers_summary(request: Request) -> dict:
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


@router.post(ENDPOINT)
async def chat_completions(request: Request):
    """Handle OpenAI Chat Completions API requests."""
    start_time = time.time()

    auth_token = request.headers.get("authorization", "") or request.headers.get("x-api-key", "")
    capture_token_from_request(request.headers.get("authorization", ""))

    try:
        payload = await request.json()
    except Exception as e:
        err_body = {"error": {"message": str(e), "type": "invalid_request_error"}}
        request_id = f"req_{time.time_ns() // 1000000}"
        try:
            req_path = log_thalamus_request(
                request_id, ENDPOINT, "POST",
                {"_parse_error": str(e)},
                _headers_summary(request),
            )
            res_path = log_thalamus_response(request_id, ENDPOINT, 400, err_body, latency_ms=0)
            log_thalamus_api_call(request_id, ENDPOINT, "POST", 400, 0, req_path, res_path, error=str(e))
        except Exception:
            pass
        return JSONResponse(status_code=400, content=err_body)

    request_id = f"req_{time.time_ns() // 1000000}"

    try:
        req_path = log_thalamus_request(
            request_id, ENDPOINT, "POST",
            payload,
            _headers_summary(request),
        )
    except Exception:
        req_path = ""

    logger.info(
        f"[{request_id}] POST {ENDPOINT} model={payload.get('model', '?')} "
        f"stream={payload.get('stream', False)} tools={len(payload.get('tools', []))}"
    )

    try:
        req = normalize_openai(payload)

        result = await run_pipeline(
            req=req,
            request_id=request_id,
            auth_token=auth_token,
        )
    except Exception as pipeline_exc:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"[{request_id}] pipeline_exception: {pipeline_exc}\n{tb_str}")
        err_body = {"error": {"message": f"Pipeline error: {pipeline_exc}", "type": "api_error", "traceback": tb_str}}
        return JSONResponse(status_code=500, content=err_body)

    latency_ms = int((time.time() - start_time) * 1000)

    if not result.get("ok"):
        status = result.get("status", 500)
        body = result.get("body", {"error": {"message": "Internal error", "type": "api_error"}})
        try:
            res_path = log_thalamus_response(request_id, ENDPOINT, status, body, latency_ms=latency_ms)
            log_thalamus_api_call(request_id, ENDPOINT, "POST", status, latency_ms, req_path, res_path,
                                 error=body.get("error", {}).get("message"))
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
                        res_path = log_thalamus_response(
                            request_id, ENDPOINT, 200,
                            {"_raw_sse_len": sum(len(c) for c in sse_chunks)},
                            latency_ms=int((time.time() - start_time) * 1000),
                        )
                        log_thalamus_api_call(
                            request_id, ENDPOINT, "POST", 200,
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
                    # Close after stream so OpenAI-compatible clients (e.g. OpenClaw / pi-ai)
                    # see EOF right after [DONE]. Keep-alive can leave some runtimes waiting
                    # for more bytes, delaying stream completion and leaving UIs stuck on "streaming".
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                    "X-Thalamus-Request-Id": request_id,
                },
            )
        err_body = {"error": {"message": "No stream handler", "type": "api_error"}}
        try:
            res_path = log_thalamus_response(request_id, ENDPOINT, 500, err_body, latency_ms=latency_ms)
            log_thalamus_api_call(request_id, ENDPOINT, "POST", 500, latency_ms, req_path, res_path, error="No stream handler")
        except Exception:
            pass
        return JSONResponse(status_code=500, content=err_body)

    body = result.get("body", {})
    try:
        res_path = log_thalamus_response(request_id, ENDPOINT, 200, body, latency_ms=latency_ms)
        log_thalamus_api_call(request_id, ENDPOINT, "POST", 200, latency_ms, req_path, res_path)
    except Exception:
        pass
    return JSONResponse(content=body)
