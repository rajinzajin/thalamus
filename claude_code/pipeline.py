from __future__ import annotations
"""
Main pipeline module — the heart of thalamus-py.

Pipeline flow (format-agnostic):
  1. Receive UnifiedRequest (from normalize_anthropic or normalize_openai)
  2. Inject tool prompts
  3. Call Cursor API via H2
  4. Parse tool calls from text response
  5. Assemble SSE output (Anthropic or OpenAI, chosen by original_format)
"""

import asyncio
import json
import os
import re
import time
import uuid
from typing import Any, AsyncIterator, Callable

from utils.structured_logging import ThalamusStructuredLogger

from core.token_manager import get_cursor_access_token
from core.bearer_token import strip_cursor_user_prefix
from core.unified_request import UnifiedRequest
from utils.llm_payload_logger import (
    log_llm_request,
    log_llm_response,
    log_llm_api_call,
)
from core.protobuf_builder import (
    build_gzip_framed_protobuf_chat_request_body,
    compute_sha256_hex_digest,
    generate_obfuscated_machine_id_checksum,
)
from core.protobuf_frame_parser import CURSOR_ABORT_ERROR_CODE, ProtobufFrameParser
from core.cursor_h2_client import open_streaming_h2_request
from claude_code.tool_prompt_builder import inject_tool_prompt_into_messages, _merge_consecutive_same_role
from config.system_prompt import THALAMUS_INSTRUCTION_SUPPLEMENT
from claude_code.tool_parser import try_parse_tool_calls_from_text
from claude_code.tool_lazy_loader import (
    is_task_complete_call,
    extract_task_complete_result,
    CONTINUATION_PROMPT,
    MAX_CONTINUATION_RETRIES,
)
from claude_code.sse_assembler import (
    StreamingAnthropicSession,
    build_unary_anthropic_response,
)
from claude_code.openai_sse_assembler import (
    StreamingOpenAISession,
    build_unary_openai_response,
)
from config.tool_registry import post_process_tool_calls
from config.fallback_config import load_fallback_config

logger = ThalamusStructuredLogger.get_logger("pipeline", "DEBUG")

FATAL_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(r"unable\s+to\s+reach\s+the\s+model\s+provider", re.I),
    re.compile(r"trouble\s+connecting", re.I),
    re.compile(r'code["\']?\s*:\s*["\']?unavailable', re.I),
    re.compile(r"ERROR_OPENAI", re.I),
    re.compile(r"service.*unavailable", re.I),
]

TOOL_JSON_START_MARKERS: list[str] = [
    '{"type":"tool_use"',
    '{"type": "tool_use"',
    '{"tool_calls"',
    '"tool_calls":',
    "```json",
    '{"function"',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_api_error_body(message: str, error_type: str = "api_error") -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _extract_raw_auth_token(value: Any) -> str:
    if not value:
        return ""
    raw = value[0] if isinstance(value, list) else value
    return re.sub(r"^Bearer\s+", "", str(raw), flags=re.I).strip()


# ---------------------------------------------------------------------------
# max_tokens limiter
# ---------------------------------------------------------------------------


def _parse_max_tokens(value: Any) -> dict:
    if value is None:
        return {"ok": True, "value": None}
    try:
        n = int(value)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"Invalid max_tokens: {value}"}
    if n < 1:
        return {"ok": False, "error": f"Invalid max_tokens: {value}"}
    return {"ok": True, "value": n}


class _OutputLimiter:
    """Approximate char-budget limiter (1 token ~ 4 chars)."""

    def __init__(self, max_tokens: int | None) -> None:
        if max_tokens and max_tokens > 0:
            self.has_limit = True
            self.char_budget = max_tokens * 4
        else:
            self.has_limit = False
            self.char_budget = None
        self._emitted = 0
        self._exhausted = False

    def emit_within_limit(self, text: str) -> str:
        if not text or self._exhausted:
            return ""
        if not self.has_limit:
            return text
        remaining = self.char_budget - self._emitted
        if remaining <= 0:
            self._exhausted = True
            return ""
        out = text[:remaining]
        self._emitted += len(out)
        if len(out) < len(text) or self._emitted >= self.char_budget:
            self._exhausted = True
        return out

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    @property
    def emitted_chars(self) -> int:
        return self._emitted


# ---------------------------------------------------------------------------
# Tool-JSON-aware text forwarder
# ---------------------------------------------------------------------------


def _find_first_tool_json_start_index(full_text: str) -> int:
    if not full_text:
        return -1
    first = -1
    for marker in TOOL_JSON_START_MARKERS:
        idx = full_text.find(marker)
        if idx >= 0 and (first < 0 or idx < first):
            first = idx
    return first


class ToolJsonAwareTextForwarder:
    """Buffer streaming text deltas and stop forwarding once tool JSON begins."""

    def __init__(
        self,
        emit_text_delta: Callable[[str], str | None],
        limiter: _OutputLimiter,
    ) -> None:
        self._emit = emit_text_delta
        self._limiter = limiter
        self.full_text_seen = ""
        self._pending_buffer = ""
        self._safe_text_consumed_len = 0
        self.stopped_due_to_tool_json = False
        self._tail_buffer_len = 15

    def _process_safe_chunk(self, chunk: str) -> str | None:
        if not chunk:
            return None
        self._safe_text_consumed_len += len(chunk)
        limited = self._limiter.emit_within_limit(chunk)
        if limited:
            return self._emit(limited)
        return None

    def on_delta(self, delta_text: str) -> str | None:
        """Feed a new text delta. Returns SSE string if text was forwarded."""
        delta = delta_text or ""
        if not delta:
            return None
        self.full_text_seen += delta
        if self.stopped_due_to_tool_json:
            return None

        self._pending_buffer += delta
        split_idx = _find_first_tool_json_start_index(self.full_text_seen)
        if split_idx >= 0:
            self.stopped_due_to_tool_json = True
            remaining_safe = max(0, split_idx - self._safe_text_consumed_len)
            if remaining_safe > 0:
                return self._process_safe_chunk(
                    self._pending_buffer[:remaining_safe]
                )
            self._pending_buffer = ""
            return None

        safe_flush_len = max(0, len(self._pending_buffer) - self._tail_buffer_len)
        if safe_flush_len > 0:
            result = self._process_safe_chunk(self._pending_buffer[:safe_flush_len])
            self._pending_buffer = self._pending_buffer[safe_flush_len:]
            return result
        return None

    def flush_using_final_safe_text(self, final_safe_text: str) -> str | None:
        """Flush remaining buffered text after the stream has ended."""
        final = final_safe_text or ""

        result_parts: list[str] = []
        if not self.stopped_due_to_tool_json and self._pending_buffer:
            r = self._process_safe_chunk(self._pending_buffer)
            if r:
                result_parts.append(r)
            self._pending_buffer = ""

        if len(final) > self._safe_text_consumed_len:
            r = self._process_safe_chunk(final[self._safe_text_consumed_len:])
            if r:
                result_parts.append(r)

        return "".join(result_parts) if result_parts else None


# ---------------------------------------------------------------------------
# Tool-call requirement detection
# ---------------------------------------------------------------------------


def is_tool_call_explicitly_required(messages: list[dict]) -> bool:
    """Check if the last user message explicitly demands a tool call."""
    if not messages:
        return False

    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        return False

    content = messages[last_user_idx].get("content", "")
    if isinstance(content, list):
        content = " ".join(
            (p.get("text", "") if isinstance(p, dict) else str(p)) for p in content
        )
    user_text = str(content).lower()
    if not user_text:
        return False

    general = [
        re.compile(r"must\s+call\s+at\s+least\s+one\s+tool"),
        re.compile(r"you\s+must\s+output\s+a\s+valid\s+tool_calls\s+json"),
        re.compile(r"必须.*至少.*调用.*工具"),
    ]
    if any(p.search(user_text) for p in general):
        return True

    first_msg = [
        re.compile(r"first\s+(assistant\s+)?message\s+must\s+be\s+tool-?call\s+json"),
        re.compile(r"第一条.*assistant.*消息.*必须.*tool-?call"),
        re.compile(r"第一条.*消息.*必须.*tool-?call"),
    ]
    if not any(p.search(user_text) for p in first_msg):
        return False

    has_assistant_before = any(
        m.get("role") == "assistant" for m in messages[:last_user_idx]
    )
    return not has_assistant_before


# ---------------------------------------------------------------------------
# Text-before-JSON extraction
# ---------------------------------------------------------------------------


def extract_text_before_json(full_text: str) -> str:
    """Extract text appearing before tool-call JSON in LLM output."""
    if not full_text:
        return ""
    patterns = [
        re.compile(r'\{"type"\s*:\s*"tool_use"'),
        re.compile(r'\{[\s\S]*?"tool_calls"\s*:\s*\['),
        re.compile(r'```(?:json)?\s*\{[\s\S]*?"tool_calls"'),
        re.compile(r"<tool_call>\s*\{"),
        re.compile(r"<<function=[^>]+>>\s*\{"),
        re.compile(r'\{[\s\S]*?"function_call"\s*:\s*\{'),
    ]
    for pattern in patterns:
        m = pattern.search(full_text)
        if m and m.start() > 0:
            before = full_text[: m.start()].strip()
            if before:
                return before
    return ""


# ---------------------------------------------------------------------------
# Fatal error detection
# ---------------------------------------------------------------------------


def _is_fatal_stream_error(error: Any) -> bool:
    text = (
        error if isinstance(error, str)
        else str(
            getattr(error, "detail", None)
            or getattr(error, "raw", None)
            or getattr(error, "message", None)
            or json.dumps(error if isinstance(error, dict) else {})
        )
    )
    return any(p.search(text) for p in FATAL_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Cursor stream plumbing
# ---------------------------------------------------------------------------


def build_cursor_stream_params(
    token: str, messages: list[dict], model: str
) -> tuple[str, dict[str, str], bytes]:
    """Build H2 path, headers, and protobuf body for a Cursor streaming request."""
    chosen_auth = strip_cursor_user_prefix(token)
    checksum = generate_obfuscated_machine_id_checksum(chosen_auth.strip())
    session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chosen_auth))
    client_key = compute_sha256_hex_digest(chosen_auth)
    client_version = os.environ.get("CURSOR_CLIENT_VERSION", "2.5.25")

    body = build_gzip_framed_protobuf_chat_request_body(
        messages, model, agent_mode=True
    )
    is_gzipped = body[0] == 0x01

    headers = {
        "authorization": f"Bearer {chosen_auth}",
        "connect-accept-encoding": "gzip",
        "connect-protocol-version": "1",
        "content-type": "application/connect+proto",
        "user-agent": "connect-es/1.6.1",
        "x-amzn-trace-id": f"Root={uuid.uuid4()}",
        "x-client-key": client_key,
        "x-cursor-checksum": checksum,
        "x-cursor-client-version": client_version,
        "x-cursor-config-version": str(uuid.uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-ghost-mode": "true",
        "x-request-id": str(uuid.uuid4()),
        "x-session-id": session_id,
        "Host": "api2.cursor.sh",
    }
    if is_gzipped:
        headers["connect-content-encoding"] = "gzip"

    path = "/aiserver.v1.ChatService/StreamUnifiedChatWithTools"
    return path, headers, body


class _ThinkTagSplitter:
    """Streaming splitter that separates <think>...</think> from regular text.

    Some models (e.g. composer-1.5) embed thinking inside the text field
    using <think> tags instead of the protobuf thinking field.  Claude Code
    expects thinking to arrive as proper ``thinking_delta`` SSE events, not
    as raw text — otherwise the response appears blank.
    """

    __slots__ = ("_inside_think", "_buf")

    def __init__(self) -> None:
        self._inside_think = False
        self._buf = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        """Return (thinking_part, text_part) extracted from *chunk*."""
        self._buf += chunk
        thinking_out: list[str] = []
        text_out: list[str] = []

        while self._buf:
            if self._inside_think:
                end = self._buf.find("</think>")
                if end == -1:
                    thinking_out.append(self._buf)
                    self._buf = ""
                else:
                    thinking_out.append(self._buf[:end])
                    self._buf = self._buf[end + len("</think>"):]
                    self._inside_think = False
            else:
                start = self._buf.find("<think>")
                if start == -1:
                    if len(self._buf) >= 7:
                        safe = len(self._buf) - 6
                        text_out.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                else:
                    text_out.append(self._buf[:start])
                    self._buf = self._buf[start + len("<think>"):]
                    self._inside_think = True

        return "".join(thinking_out), "".join(text_out)

    def flush(self) -> tuple[str, str]:
        """Flush any remaining buffered content."""
        remaining = self._buf
        self._buf = ""
        if self._inside_think:
            self._inside_think = False
            return remaining, ""
        return "", remaining


async def consume_stream(
    stream_iterator: AsyncIterator[bytes],
    on_text_delta: Callable[[str], Any] | None = None,
    on_thinking_delta: Callable[[str], Any] | None = None,
) -> dict:
    """Consume a Cursor protobuf stream, accumulating text/thinking/errors.

    Tool calls are extracted from text after the stream completes (prompt
    injection approach), not from protobuf wire-level tool call fields.

    Models that embed <think>...</think> in the text field (instead of the
    protobuf thinking field) are handled transparently: the tags are stripped
    and the content is routed to on_thinking_delta.
    """
    parser = ProtobufFrameParser()
    splitter = _ThinkTagSplitter()
    text = ""
    thinking = ""
    errors: list[Any] = []
    had_content = False
    has_fatal_error = False
    chunk_count = 0
    text_delta_count = 0
    thinking_delta_count = 0
    stream_start = time.monotonic()
    first_chunk_latency_ms: float | None = None

    async for chunk in stream_iterator:
        chunk_count += 1
        if first_chunk_latency_ms is None:
            first_chunk_latency_ms = (time.monotonic() - stream_start) * 1000
        logger.debug(f"[consume] chunk#{chunk_count} len={len(chunk)} hex_head={chunk[:20].hex()}")

        result = parser.parse(chunk)
        logger.debug(
            f"[consume] chunk#{chunk_count} parsed: text_len={len(result.text)} "
            f"thinking_len={len(result.thinking)} errors={len(result.errors)}"
        )

        if result.errors:
            errors.extend(result.errors)
            for err in result.errors:
                if _is_fatal_stream_error(err):
                    has_fatal_error = True

        if result.thinking:
            thinking += result.thinking
            thinking_delta_count += 1
            had_content = True
            if on_thinking_delta:
                on_thinking_delta(result.thinking)

        if result.text:
            think_part, text_part = splitter.feed(result.text)

            if think_part:
                thinking += think_part
                thinking_delta_count += 1
                had_content = True
                if on_thinking_delta:
                    on_thinking_delta(think_part)

            if text_part:
                text += text_part
                text_delta_count += 1
                had_content = True
                if on_text_delta:
                    on_text_delta(text_part)
                logger.debug(f"[consume] text so far ({len(text)} chars): ...{text[-200:]}")

    flush_think, flush_text = splitter.flush()
    if flush_think:
        thinking += flush_think
        thinking_delta_count += 1
        had_content = True
        if on_thinking_delta:
            on_thinking_delta(flush_think)
    if flush_text:
        text += flush_text
        text_delta_count += 1
        had_content = True
        if on_text_delta:
            on_text_delta(flush_text)

    errors = [
        e for e in errors
        if not (hasattr(e, "error_code") and e.error_code == CURSOR_ABORT_ERROR_CODE)
    ]

    return {
        "text": text,
        "thinking": thinking,
        "errors": errors,
        "had_content": had_content,
        "has_fatal_error": has_fatal_error,
        "metrics": {
            "stream_duration_ms": (time.monotonic() - stream_start) * 1000,
            "first_chunk_latency_ms": first_chunk_latency_ms if first_chunk_latency_ms is not None else -1,
            "chunk_count": chunk_count,
            "text_delta_count": text_delta_count,
            "thinking_delta_count": thinking_delta_count,
            "protocol_error_count": len(errors),
        },
    }


# ---------------------------------------------------------------------------
# Internal Cursor caller with fallback
# ---------------------------------------------------------------------------




async def _call_cursor_direct(
    messages: list[dict],
    model: str,
    tools: list[dict],
    valid_tool_names: list[str],
    auth_token: str,
    on_stream_delta: Callable[[str], Any] | None = None,
    on_thinking_delta: Callable[[str], Any] | None = None,
) -> dict:
    """Call Cursor API with tool schema injection, continuation retry, and fallback."""
    start_time = time.monotonic()
    request_id = f"cc_{uuid.uuid4().hex[:12]}"

    injected_base = inject_tool_prompt_into_messages(messages, tools)
    has_valid_tools = bool(valid_tool_names)
    all_valid_names = list(valid_tool_names) + (["task_complete"] if has_valid_tools else [])

    fallback_cfg = load_fallback_config()
    tried_models: list[str] = []
    current_model = model

    while len(tried_models) < fallback_cfg.max_attempts:
        logger.info(
            f"[{request_id}] Calling Cursor direct | model={current_model} | tools={len(valid_tool_names)} | msgs={len(injected_base)} | attempt={len(tried_models) + 1}"
        )

        attempt_start = time.monotonic()

        req_payload_path = log_llm_request(
            request_id, current_model, injected_base,
            extra={"tools": len(valid_tool_names), "attempt": len(tried_models) + 1},
        )

        try:
            path, headers, body = build_cursor_stream_params(
                auth_token, injected_base, current_model
            )
            async with open_streaming_h2_request(path, headers, body) as stream_iter:
                consumed = await consume_stream(
                    stream_iter,
                    on_text_delta=on_stream_delta,
                    on_thinking_delta=on_thinking_delta,
                )
        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"[{request_id}] Connection/stream error: {err_msg}")
            if fallback_cfg.should_fallback(err_msg):
                tried_models.append(current_model)
                next_model = fallback_cfg.select_next_model(
                    model, tried_models
                )
                if next_model:
                    logger.info(
                        f"[{request_id}] Fallback: {current_model} -> {next_model}"
                    )
                    current_model = next_model
                    continue
            latency = int((time.monotonic() - attempt_start) * 1000)
            res_payload_path = log_llm_response(
                request_id, current_model, "", error=err_msg, latency_ms=latency,
            )
            log_llm_api_call(
                request_id, current_model, "ERROR", latency,
                req_payload_path, res_payload_path, error=err_msg,
            )
            return {
                "error": err_msg,
                "status": 503,
                "fallback_attempts": len(tried_models),
                "model": current_model,
            }

        should_fallback_from_stream = (
            (consumed["errors"] and not consumed["had_content"])
            or consumed["has_fatal_error"]
        )

        if should_fallback_from_stream:
            err_detail = _first_error_detail(consumed["errors"])
            logger.warn(
                f"[{request_id}] Stream error: {err_detail} | had_content={consumed['had_content']} fatal={consumed['has_fatal_error']}"
            )
            if fallback_cfg.should_fallback(err_detail):
                tried_models.append(current_model)
                next_model = fallback_cfg.select_next_model(model, tried_models)
                if next_model:
                    logger.info(
                        f"[{request_id}] Fallback: {current_model} -> {next_model} (stream error)"
                    )
                    current_model = next_model
                    continue
            return {
                "error": err_detail,
                "status": 503,
                "fallback_attempts": len(tried_models),
                "model": current_model,
            }

        metrics = consumed["metrics"]
        attempt_latency = int((time.monotonic() - attempt_start) * 1000)
        logger.info(
            f"[{request_id}] Stream consumed | text_len={len(consumed['text'])} thinking_len={len(consumed['thinking'])} "
            f"errors={len(consumed['errors'])} "
            f"chunks={metrics['chunk_count']} first_token_ms={metrics['first_chunk_latency_ms']:.0f}"
        )

        converted_tcs: list[dict] = []
        if all_valid_names:
            parsed_tcs = try_parse_tool_calls_from_text(consumed["text"])
            if parsed_tcs:
                result = post_process_tool_calls(parsed_tcs, all_valid_names)
                converted_tcs = result.get("processed") or []
                if converted_tcs:
                    converted_tcs = _fix_garbled_paths_in_tool_calls(converted_tcs)
                logger.info(f"[{request_id}] Text-parsed tool calls: {len(converted_tcs)}")

        res_payload_path = log_llm_response(
            request_id, current_model, consumed["text"],
            tool_calls=converted_tcs or None,
            error=_first_error_detail(consumed["errors"]) if consumed["errors"] else None,
            latency_ms=attempt_latency,
            extra={
                "thinking_len": len(consumed["thinking"]),
                "chunks": metrics["chunk_count"],
            },
        )
        log_llm_api_call(
            request_id, current_model,
            "OK" if not consumed["errors"] else "STREAM_ERROR",
            attempt_latency, req_payload_path, res_payload_path,
            error=_first_error_detail(consumed["errors"]) if consumed["errors"] else None,
        )

        if not has_valid_tools:
            return {
                "text": consumed["text"],
                "thinking": consumed["thinking"],
                "model": current_model,
                "fallback_attempts": len(tried_models),
                "stats": {"passed": 0, "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
            }

        # Filter out task_complete from final tool_calls (it's a pseudo-tool)
        if converted_tcs:
            final_tcs = []
            task_complete_text = ""
            for tc in converted_tcs:
                if is_task_complete_call(tc):
                    task_complete_text = extract_task_complete_result(tc)
                    logger.info(f"[{request_id}] task_complete in final batch: {task_complete_text[:200]}")
                else:
                    final_tcs.append(tc)

            if task_complete_text and not final_tcs:
                text_before = extract_text_before_json(consumed["text"])
                return {
                    "text": text_before if text_before else task_complete_text,
                    "task_complete_text": task_complete_text,
                    "thinking": consumed["thinking"],
                    "model": current_model,
                    "fallback_attempts": len(tried_models),
                    "stats": {"passed": 0, "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
                }
            converted_tcs = final_tcs

        if converted_tcs:
            raw_names = [(tc.get("function") or {}).get("name", "?") for tc in converted_tcs]
            logger.info(f"[{request_id}] Tool calls: {json.dumps([{'name': n} for n in raw_names])}")

            return {
                "tool_calls": converted_tcs,
                "text": consumed["text"],
                "thinking": consumed["thinking"],
                "model": current_model,
                "stats": {"passed": len(converted_tcs), "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
                "fallback_attempts": len(tried_models),
            }

        # ── Continuation retry: no tool_calls AND no task_complete → not done ──
        # The ONLY way to signal "done" is task_complete. No task_complete = keep going.
        # This is purely mechanical — no heuristics, no "is it a tool_result" guessing.
        first_text = consumed["text"]
        first_thinking = consumed["thinking"]

        cont_retries = 0
        accumulated_text = first_text

        while cont_retries < MAX_CONTINUATION_RETRIES:
            cont_retries += 1
            logger.info(
                f"[{request_id}] Continuation retry {cont_retries}/{MAX_CONTINUATION_RETRIES} | "
                f"text_len={len(accumulated_text)} | no tool_calls, no task_complete"
            )

            injected_base.append({"role": "assistant", "content": accumulated_text})
            injected_base.append({"role": "user", "content": CONTINUATION_PROMPT})
            injected_base = _merge_consecutive_same_role(injected_base)

            try:
                path_c, headers_c, body_c = build_cursor_stream_params(
                    auth_token, injected_base, current_model
                )
                async with open_streaming_h2_request(path_c, headers_c, body_c) as stream_c:
                    consumed_c = await consume_stream(stream_c)
            except Exception as exc_c:
                logger.error(f"[{request_id}] Continuation retry stream error: {exc_c}")
                break

            if consumed_c["has_fatal_error"] or (consumed_c["errors"] and not consumed_c["had_content"]):
                logger.warn(f"[{request_id}] Continuation retry got fatal error")
                break

            retry_text = consumed_c["text"]
            logger.info(f"[{request_id}] Continuation retry response | text_len={len(retry_text)}")

            retry_tcs: list[dict] = []
            parsed_retry = try_parse_tool_calls_from_text(retry_text)
            if parsed_retry:
                result_retry = post_process_tool_calls(parsed_retry, all_valid_names)
                retry_tcs = result_retry.get("processed") or []
                if retry_tcs:
                    retry_tcs = _fix_garbled_paths_in_tool_calls(retry_tcs)

            if retry_tcs:
                # Check for task_complete
                tc_complete = [tc for tc in retry_tcs if is_task_complete_call(tc)]
                real_tcs = [tc for tc in retry_tcs if not is_task_complete_call(tc)]

                if tc_complete and not real_tcs:
                    tc_result = extract_task_complete_result(tc_complete[0])
                    logger.info(f"[{request_id}] Continuation → task_complete: {tc_result[:200]}")
                    return {
                        "text": first_text,
                        "task_complete_text": tc_result,
                        "thinking": first_thinking,
                        "model": current_model,
                        "fallback_attempts": len(tried_models),
                        "stats": {"passed": 0, "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
                    }

                if real_tcs:
                    retry_text_before = extract_text_before_json(retry_text)
                    merged_text = first_text
                    if retry_text_before and retry_text_before.strip():
                        merged_text = first_text + "\n" + retry_text_before

                    raw_names = [(tc.get("function") or {}).get("name", "?") for tc in real_tcs]
                    logger.info(
                        f"[{request_id}] Continuation → merged text+tool | "
                        f"text_len={len(merged_text)} tools={raw_names}"
                    )
                    return {
                        "tool_calls": real_tcs,
                        "text": merged_text,
                        "thinking": first_thinking,
                        "model": current_model,
                        "stats": {"passed": len(real_tcs), "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
                        "fallback_attempts": len(tried_models),
                    }

            accumulated_text = retry_text

        # Exhausted retries — model never called task_complete or any tool.
        # Fall through as end_turn (safety valve).
        logger.info(
            f"[{request_id}] Text response (after {cont_retries} continuation retries, no task_complete) | "
            f"len={len(first_text)} | {(time.monotonic() - start_time) * 1000:.0f}ms"
        )
        return {
            "text": first_text,
            "thinking": first_thinking,
            "model": current_model,
            "fallback_attempts": len(tried_models),
            "stats": {"passed": 0, "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
        }

    logger.error(f"[{request_id}] All fallback models exhausted")
    return {
        "error": "All available models are currently unavailable",
        "status": 503,
        "fallback_attempts": len(tried_models),
        "model": None,
    }


def _fix_garbled_paths_in_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Fix character corruption in paths by filesystem lookup — no hardcoded char maps.

    Walk each path segment top-down; when a segment doesn't exist, fuzzy-match
    against the real directory listing to find the closest real name.
    Works for ANY character corruption as long as the real file/dir exists on disk.

    Safety rules:
      - Only fix directory segments, not the final filename (Write may create new files)
      - Require same length + ≥80% char match to avoid false positives
      - For Bash commands: only fix paths that look like they reference existing trees
      - Don't swap quote styles if the path contains single-quotes or $variables
    """
    _listdir_cache: dict[str, list[str]] = {}

    def _cached_listdir(d: str) -> list[str]:
        if d not in _listdir_cache:
            try:
                _listdir_cache[d] = os.listdir(d)
            except OSError:
                _listdir_cache[d] = []
        return _listdir_cache[d]

    def _fuzzy_match_segment(parent: str, broken_seg: str) -> str | None:
        children = _cached_listdir(parent)
        if not children:
            return None
        if broken_seg in children:
            return broken_seg

        best, best_score = None, 0
        seg_len = len(broken_seg)
        for child in children:
            if len(child) != seg_len:
                continue
            score = sum(a == b for a, b in zip(child, broken_seg))
            if score > best_score:
                best_score = score
                best = child

        threshold = max(seg_len * 0.8, seg_len - 2)
        if best and best_score >= threshold:
            return best

        broken_lower = broken_seg.lower()
        for child in children:
            if child.lower() == broken_lower:
                return child
        return None

    def _fix_path(p: str, fix_last_segment: bool = True) -> str:
        """Fix garbled segments in an absolute path.

        fix_last_segment=False means the final component (filename) won't be
        fuzzy-matched — used for Write/Edit where the file may not exist yet.
        """
        if not p or not p.startswith("/") or os.path.exists(p):
            return p

        parts = p.split("/")
        rebuilt = ""
        fixed = False
        last_idx = len(parts) - 1
        for i, seg in enumerate(parts):
            if not seg:
                rebuilt += "/"
                continue
            candidate = rebuilt + seg
            if os.path.exists(candidate):
                rebuilt = candidate + ("/" if i < last_idx else "")
                continue
            if i == last_idx and not fix_last_segment:
                rebuilt += seg
                continue
            real = _fuzzy_match_segment(rebuilt if rebuilt else "/", seg)
            if real and real != seg:
                logger.info(f"[path-fix] segment '{seg}' → '{real}' in {rebuilt}")
                rebuilt += real + ("/" if i < last_idx else "")
                fixed = True
            else:
                rebuilt += seg + ("/" if i < last_idx else "")

        if fixed and rebuilt != p:
            return rebuilt.rstrip("/") if not p.endswith("/") else rebuilt
        return p

    def _fix_paths_in_string(s: str) -> str:
        """Find absolute paths in Bash commands and fix garbled segments.

        Only swaps double→single quotes when a path was actually fixed AND
        the fixed path contains shell-dangerous chars (!) AND it's safe
        (no single-quotes or $variables in the context).
        """
        patterns = [
            re.compile(r'"(/[^"]+)"'),
            re.compile(r"'(/[^']+)'"),
            re.compile(r'(?:^|[ =])(/[^\s"\']+(?:\\ [^\s"\']+)*)'),
            re.compile(r'(?<=\n)(/[^\s"\']+)'),
        ]
        result = s
        for pat in patterns:
            def _make_replacer(p: re.Pattern) -> callable:
                def _replacer(m: re.Match) -> str:
                    original = m.group(1)
                    if not original.startswith("/"):
                        return m.group(0)
                    fixed_p = _fix_path(original)
                    if fixed_p == original:
                        return m.group(0)
                    new_match = m.group(0).replace(original, fixed_p)
                    if "!" in fixed_p and '"' in m.group(0):
                        if "'" not in fixed_p and "$" not in m.group(0):
                            new_match = new_match.replace(f'"{fixed_p}"', f"'{fixed_p}'")
                    return new_match
                return _replacer
            result = pat.sub(_make_replacer(pat), result)
        return result

    WRITE_TOOLS = {"Write", "Edit", "MultiEdit"}

    result = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError, TypeError):
            result.append(tc)
            continue
        if not isinstance(args, dict):
            result.append(tc)
            continue

        changed = False
        is_write = name in WRITE_TOOLS

        for key in ("file_path", "path", "pattern"):
            if key in args and isinstance(args[key], str) and args[key].startswith("/"):
                fixed = _fix_path(args[key], fix_last_segment=not is_write)
                if fixed != args[key]:
                    args[key] = fixed
                    changed = True

        if name == "Bash" and "command" in args and isinstance(args["command"], str):
            fixed = _fix_paths_in_string(args["command"])
            if fixed != args["command"]:
                args["command"] = fixed
                changed = True

        if changed:
            tc = {
                **tc,
                "function": {**fn, "arguments": json.dumps(args)},
            }
        result.append(tc)
    return result



def _first_error_detail(errors: list) -> str:
    if not errors:
        return "Unknown stream error"
    err = errors[0]
    if isinstance(err, str):
        return err
    return str(
        getattr(err, "detail", None)
        or getattr(err, "raw", None)
        or getattr(err, "message", None)
        or err
    )


# ---------------------------------------------------------------------------
# Streaming delta granularity — split large Cursor chunks into small SSE events
# ---------------------------------------------------------------------------

DELTA_TARGET_SIZE = 8  # chars per SSE event, simulates token-level streaming
MIN_EVENT_DELAY = 0.003   # seconds — fast drain when queue is backlogged
MAX_EVENT_DELAY = 0.015   # seconds — smooth pacing when stream is trickling in


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_pipeline(
    req: UnifiedRequest,
    request_id: str,
    auth_token: str = "",
) -> dict:
    """Format-agnostic pipeline entry point.

    Accepts a UnifiedRequest (from normalize_anthropic or normalize_openai).
    Returns a dict with:
      - ok (bool)
      - stream (bool, if streaming)
      - body (dict, if non-streaming)
      - stream_handler (async generator, if streaming)
      - telemetry (dict)
    """
    pipeline_start = time.monotonic()

    messages = req.messages
    tools = req.tools
    stream = req.stream
    resolved_model = req.model
    max_tokens = req.max_tokens
    original_format = req.original_format
    valid_tool_names = [(t.get("function") or t).get("name", "") for t in tools]

    # Claude Code relay context only when the client registered tools.
    full_system = req.system or ""
    if tools:
        full_system += THALAMUS_INSTRUCTION_SUPPLEMENT
    if full_system:
        messages = [{"role": "system", "content": full_system}] + messages

    if req.metadata:
        logger.info(f"[{request_id}] CC metadata: {json.dumps(req.metadata, ensure_ascii=False)[:200]}")
    if req.thinking:
        logger.info(f"[{request_id}] CC thinking config: {req.thinking}")
    if req.context_management:
        logger.debug(f"[{request_id}] CC context_management: {req.context_management}")

    parsed_mt = _parse_max_tokens(max_tokens)
    if not parsed_mt["ok"]:
        return {
            "ok": False,
            "status": 400,
            "body": _to_api_error_body(parsed_mt["error"], "invalid_request_error"),
            "telemetry": {
                "request_id": request_id,
                "pipeline": "claude_code",
                "model_requested": resolved_model,
                "model_used": None,
                "latency_ms": _elapsed_ms(pipeline_start),
                "stream": stream,
            },
        }
    max_tokens = parsed_mt["value"]

    raw_req_token = _extract_raw_auth_token(auth_token)
    if raw_req_token and ("::" in raw_req_token or raw_req_token.startswith("eyJ")):
        token = raw_req_token
    else:
        token = get_cursor_access_token()

    logger.info(
        f"[{request_id}] pipeline=claude_code format={original_format} model={resolved_model} "
        f"stream={stream} tools={len(valid_tool_names)} msgs={len(messages)} max_tokens={max_tokens or '-'}"
    )
    if valid_tool_names and len(messages) <= 5:
        logger.info(f"[{request_id}] tool names: {valid_tool_names}")
        for t in tools[:5]:
            fn = t.get("function") or t
            tname = fn.get("name", "")
            tschema = fn.get("input_schema") or fn.get("parameters") or {}
            req_params = tschema.get("required", [])
            props = list((tschema.get("properties") or {}).keys())
            logger.info(f"[{request_id}] tool schema: {tname} props={props} required={req_params}")

    base_telemetry: dict[str, Any] = {
        "request_id": request_id,
        "pipeline": "claude_code",
        "original_format": original_format,
        "model_requested": resolved_model,
        "max_tokens": max_tokens,
        "stream": stream,
        "agent_mode": True,
    }

    if stream:
        return _build_streaming_result(
            req, request_id, messages, tools, valid_tool_names,
            resolved_model, max_tokens, token, original_format,
            pipeline_start, base_telemetry,
        )

    # --- Non-streaming (unary) path ---
    return await _build_unary_result(
        req, request_id, messages, tools, valid_tool_names,
        resolved_model, max_tokens, token, original_format,
        pipeline_start, base_telemetry,
    )


def _build_streaming_result(
    req: UnifiedRequest,
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    original_format: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
) -> dict:
    """Build the streaming result dict with an async generator."""

    if original_format == "openai":
        return _build_streaming_result_openai(
            request_id, messages, tools, valid_tool_names,
            resolved_model, max_tokens, token,
            pipeline_start, base_telemetry,
        )
    return _build_streaming_result_anthropic(
        request_id, messages, tools, valid_tool_names,
        resolved_model, max_tokens, token,
        pipeline_start, base_telemetry,
    )


def _build_streaming_result_anthropic(
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
) -> dict:
    message_id = f"msg_{uuid.uuid4().hex}"

    async def stream_handler() -> AsyncIterator[str]:
        session = StreamingAnthropicSession(message_id, resolved_model)
        yield session.emit_message_start()

        limiter = _OutputLimiter(max_tokens)
        sse_queue: asyncio.Queue[str | None] = asyncio.Queue()

        thinking_started = False
        thinking_ended = False

        def _enqueue_text_fragments(text: str) -> None:
            """Split text into DELTA_TARGET_SIZE chunks and enqueue as text_delta SSE."""
            for i in range(0, len(text), DELTA_TARGET_SIZE):
                fragment = text[i:i + DELTA_TARGET_SIZE]
                sse = session.emit_text_delta(fragment)
                if sse:
                    sse_queue.put_nowait(sse)

        def on_thinking_as_text(delta: str) -> None:
            nonlocal thinking_started
            if not delta:
                return
            if not thinking_started:
                thinking_started = True
                _enqueue_text_fragments("thinking: ")
            limited = limiter.emit_within_limit(delta)
            if not limited:
                return
            _enqueue_text_fragments(limited)

        def _emit_and_enqueue(text: str) -> str | None:
            """Emit callback for ToolJsonAwareTextForwarder — splits into fragments."""
            if text:
                _enqueue_text_fragments(text)
            return text

        forwarder = ToolJsonAwareTextForwarder(
            emit_text_delta=_emit_and_enqueue,
            limiter=limiter,
        )

        def on_text_delta(delta: str) -> None:
            nonlocal thinking_ended
            if not delta:
                return
            if thinking_started and not thinking_ended:
                thinking_ended = True
                _enqueue_text_fragments("\n\n")
            forwarder.on_delta(delta)

        async def run_cursor_call() -> dict:
            return await _call_cursor_direct(
                messages, resolved_model, tools, valid_tool_names, token,
                on_stream_delta=on_text_delta,
                on_thinking_delta=on_thinking_as_text,
            )

        cursor_task = asyncio.create_task(run_cursor_call())

        while not cursor_task.done() or not sse_queue.empty():
            try:
                sse = sse_queue.get_nowait()
                if sse:
                    yield sse
                    depth = sse_queue.qsize()
                    if depth > 5:
                        await asyncio.sleep(MIN_EVENT_DELAY)
                    elif depth <= 2:
                        await asyncio.sleep(MAX_EVENT_DELAY)
                    else:
                        frac = (depth - 2) / 3.0
                        await asyncio.sleep(MAX_EVENT_DELAY - frac * (MAX_EVENT_DELAY - MIN_EVENT_DELAY))
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)

        direct_result = cursor_task.result()

        if direct_result.get("error"):
            logger.error(
                f"[{request_id}] pipeline=claude_code stage=error(stream) error={direct_result['error']}"
            )
            yield session.finish(stop_reason="end_turn")
            return

        tool_calls = direct_result.get("tool_calls") or []
        task_complete_text = direct_result.get("task_complete_text", "")

        full_text = direct_result.get("text", "")
        final_safe = extract_text_before_json(full_text) if tool_calls else full_text
        if task_complete_text and not tool_calls:
            final_safe = task_complete_text
        forwarder.flush_using_final_safe_text(final_safe)
        while not sse_queue.empty():
            sse = sse_queue.get_nowait()
            if sse:
                yield sse

        yield session.close_open_blocks()

        if tool_calls:
            yield session.emit_tool_use_blocks(tool_calls)

        stop_reason: str
        if tool_calls:
            stop_reason = "tool_use"
        elif task_complete_text or limiter.is_exhausted:
            stop_reason = "end_turn" if task_complete_text else "max_tokens"
        else:
            stop_reason = "end_turn"

        yield session.finish(stop_reason=stop_reason)

        used_model = direct_result.get("model") or resolved_model
        latency = _elapsed_ms(pipeline_start)
        logger.info(
            f"[{request_id}] pipeline=claude_code stage=result(stream/anthropic) model={used_model} "
            f"tool_calls={len(tool_calls)} stop_reason={stop_reason} latency_ms={latency:.0f}"
        )

    return {
        "ok": True,
        "stream": True,
        "stream_handler": stream_handler,
        "telemetry": {**base_telemetry, "model_used": resolved_model},
    }


def _build_streaming_result_openai(
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
) -> dict:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    async def stream_handler() -> AsyncIterator[str]:
        session = StreamingOpenAISession(completion_id, resolved_model)
        yield session.emit_role_chunk()

        limiter = _OutputLimiter(max_tokens)
        sse_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_text_delta(delta: str) -> None:
            if not delta:
                return
            limited = limiter.emit_within_limit(delta)
            if not limited:
                return
            for i in range(0, len(limited), DELTA_TARGET_SIZE):
                fragment = limited[i:i + DELTA_TARGET_SIZE]
                sse = session.emit_text_delta(fragment)
                if sse:
                    sse_queue.put_nowait(sse)

        async def run_cursor_call() -> dict:
            return await _call_cursor_direct(
                messages, resolved_model, tools, valid_tool_names, token,
                on_stream_delta=on_text_delta,
            )

        cursor_task = asyncio.create_task(run_cursor_call())

        while not cursor_task.done() or not sse_queue.empty():
            try:
                sse = sse_queue.get_nowait()
                if sse:
                    yield sse
                    depth = sse_queue.qsize()
                    if depth > 5:
                        await asyncio.sleep(MIN_EVENT_DELAY)
                    elif depth <= 2:
                        await asyncio.sleep(MAX_EVENT_DELAY)
                    else:
                        frac = (depth - 2) / 3.0
                        await asyncio.sleep(MAX_EVENT_DELAY - frac * (MAX_EVENT_DELAY - MIN_EVENT_DELAY))
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)

        direct_result = cursor_task.result()

        if direct_result.get("error"):
            logger.error(
                f"[{request_id}] pipeline=claude_code stage=error(stream) error={direct_result['error']}"
            )
            yield session.finish(stop_reason="stop")
            return

        tool_calls = direct_result.get("tool_calls") or []

        if tool_calls:
            yield session.emit_tool_use_blocks(tool_calls)

        stop_reason: str
        if tool_calls:
            stop_reason = "tool_calls"
        elif limiter.is_exhausted:
            stop_reason = "length"
        else:
            stop_reason = "stop"

        yield session.finish(stop_reason=stop_reason)

        used_model = direct_result.get("model") or resolved_model
        latency = _elapsed_ms(pipeline_start)
        logger.info(
            f"[{request_id}] pipeline=claude_code stage=result(stream/openai) model={used_model} "
            f"tool_calls={len(tool_calls)} stop_reason={stop_reason} latency_ms={latency:.0f}"
        )

    return {
        "ok": True,
        "stream": True,
        "stream_handler": stream_handler,
        "telemetry": {**base_telemetry, "model_used": resolved_model},
    }


async def _build_unary_result(
    req: UnifiedRequest,
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    original_format: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
) -> dict:
    """Build a non-streaming result."""

    direct_result = await _call_cursor_direct(
        messages, resolved_model, tools, valid_tool_names, token,
    )

    logger.info(
        f"[{request_id}] DIRECT_RESULT(unary): text_len={len(direct_result.get('text', ''))} "
        f"tool_calls={len(direct_result.get('tool_calls') or [])} error={direct_result.get('error')}"
    )

    if direct_result.get("error"):
        return {
            "ok": False,
            "status": direct_result.get("status", 500),
            "body": _to_api_error_body(direct_result["error"]),
            "telemetry": {
                **base_telemetry,
                "model_used": direct_result.get("model"),
                "latency_ms": _elapsed_ms(pipeline_start),
            },
        }

    used_model = direct_result.get("model") or resolved_model
    tool_calls = direct_result.get("tool_calls") or []
    text = direct_result.get("text", "")
    thinking = direct_result.get("thinking", "")

    truncated = False
    if max_tokens and max_tokens > 0:
        char_budget = max_tokens * 4
        if len(text) > char_budget:
            text = text[:char_budget]
            truncated = True

    stop_reason_override = ""
    if not tool_calls and truncated:
        stop_reason_override = "max_tokens"

    telemetry = {
        **base_telemetry,
        "model_used": used_model,
        "fallback_attempts": direct_result.get("fallback_attempts", 0),
        "latency_ms": _elapsed_ms(pipeline_start),
        "stream": False,
        "output_truncated": truncated,
    }

    logger.info(
        f"[{request_id}] pipeline=claude_code stage=result model={used_model} "
        f"tool_calls={len(tool_calls)} text_len={len(text)} latency_ms={telemetry['latency_ms']:.0f}"
    )

    if original_format == "openai":
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        body = build_unary_openai_response(
            completion_id=completion_id,
            model=used_model,
            text=text,
            tool_calls=tool_calls,
            stop_reason_override=stop_reason_override,
        )
    else:
        message_id = f"msg_{uuid.uuid4().hex}"
        body = build_unary_anthropic_response(
            message_id=message_id,
            model=used_model,
            text=text,
            thinking="",
            tool_calls=tool_calls,
            stop_reason_override=stop_reason_override,
        )

    return {
        "ok": True,
        "stream": False,
        "body": body,
        "telemetry": telemetry,
    }


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
