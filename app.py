"""FastAPI interface for the bounded citizen-feedback RAG pipeline."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from llm_client import LLMClientError, LLMConfig, call_model
import pipeline
from ingest import IngestionError
from models import Result

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FALLBACK_UPLOAD_DIR = Path(tempfile.gettempdir()) / "wananchi-actions" / "uploads"
ALLOWED_SUFFIXES = {".txt", ".md"}
MAX_UPLOAD_FILES = 10
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_UPLOAD_BYTES = 5_000_000
MAX_QUERY_CHARS = 1_000
MODEL_REQUEST_BUDGET_SECONDS = 20.0
_knowledge_base_lock = threading.RLock()
_REQUIRED_MODEL_ENV = (
    "RAG_LLM_ENDPOINT",
    "RAG_LLM_MODEL",
    "RAG_LLM_API_KEY",
)


class ProcessRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=MAX_QUERY_CHARS)]

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


def _data_files() -> list[Path]:
    roots = [DATA_DIR]
    if FALLBACK_UPLOAD_DIR != DATA_DIR / "uploads":
        roots.append(FALLBACK_UPLOAD_DIR)
    paths = {
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    }
    return sorted(paths, key=lambda path: str(path).casefold())


def _upload_directory() -> Path:
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=UPLOAD_DIR):
            pass
        return UPLOAD_DIR
    except OSError as error:
        logger.warning(
            "Upload directory is not writable (%s); using temporary storage",
            type(error).__name__,
        )
        FALLBACK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        return FALLBACK_UPLOAD_DIR


def _model_configuration() -> tuple[bool, list[str], list[str]]:
    configured = [
        name for name in _REQUIRED_MODEL_ENV if os.environ.get(name, "").strip()
    ]
    missing = [name for name in _REQUIRED_MODEL_ENV if name not in configured]
    return not missing, configured, missing


def _bounded_model_client(deadline: float | None = None):
    if deadline is None:
        deadline = time.monotonic() + MODEL_REQUEST_BUDGET_SECONDS
    config: LLMConfig | None = None

    def request_model(prompt: str, *, system_prompt: str = "") -> str:
        nonlocal config
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LLMClientError(
                "timeout reason=request_budget_exhausted: model request budget exceeded",
                error_type="timeout",
                reason="request_budget_exhausted",
            )
        if config is None:
            config = LLMConfig.from_env()
        bounded_config = LLMConfig(
            endpoint=config.endpoint,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=min(config.timeout_seconds, remaining),
            # Pipeline schema retries share this deadline; transport retries stay disabled.
            max_retries=0,
        )
        return call_model(prompt, system_prompt=system_prompt, config=bounded_config)

    return request_model


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    model_configured, configured, missing = _model_configuration()
    if model_configured:
        logger.info("Model configuration ready; required variables set: %s", ", ".join(configured))
    else:
        logger.warning(
            "Model configuration incomplete; set: %s; missing: %s",
            ", ".join(configured) or "none",
            ", ".join(missing),
        )
    with _knowledge_base_lock:
        if not pipeline._KNOWLEDGE_BASE:
            paths = _data_files()
            if paths:
                try:
                    pipeline.ingest_documents(paths)
                    logger.info("Loaded knowledge base from %d files", len(paths))
                except IngestionError:
                    logger.exception("Could not load startup data; API remains available")
    yield


app = FastAPI(
    title="Bounded Citizen Feedback RAG API",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")


@app.get("/healthz")
@app.get("/health")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, object]:
    model_configured, _, missing = _model_configuration()
    return {
        "status": "ok",
        "model_configured": model_configured,
        "missing_variables": missing,
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(ROOT_DIR / "static" / "index.html", media_type="text/html")


def _safe_upload_name(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=422, detail="Every upload must have a filename")
    normalized = filename.replace("\\", "/")
    basename = PurePosixPath(normalized).name
    if basename in {"", ".", ".."} or basename != normalized.rstrip("/"):
        raise HTTPException(status_code=422, detail="Upload filename must not contain a path")
    if Path(basename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=422, detail="Only .txt and .md files are accepted")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", basename)
    if safe_name in {"", ".", ".."}:
        raise HTTPException(status_code=422, detail="Upload filename is invalid")
    return safe_name


async def _write_upload(upload: UploadFile, destination: Path) -> int:
    size = 0
    try:
        with destination.open("xb") as output:
            while True:
                block = await upload.read(64 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{upload.filename} exceeds the {MAX_FILE_BYTES}-byte limit",
                    )
                output.write(block)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"{upload.filename} is empty")
    return size


@app.post("/ingest")
async def ingest_uploads(
    files: Annotated[list[UploadFile], File(description="Text or Markdown feedback files")],
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"No more than {MAX_UPLOAD_FILES} files may be uploaded at once",
        )

    staged_paths: list[Path] = []
    total_bytes = 0
    try:
        upload_dir = _upload_directory()
        for upload in files:
            safe_name = _safe_upload_name(upload.filename)
            destination = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
            total_bytes += await _write_upload(upload, destination)
            staged_paths.append(destination)
            if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Combined upload exceeds {MAX_TOTAL_UPLOAD_BYTES} bytes",
                )

        with _knowledge_base_lock:
            chunk_count = pipeline.ingest_documents(_data_files())
    except HTTPException:
        for path in staged_paths:
            path.unlink(missing_ok=True)
        raise
    except (IngestionError, OSError, ValueError) as error:
        for path in staged_paths:
            path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": f"Ingestion failed: {error}"},
        )

    return {
        "status": "ok",
        "file_count": len(staged_paths),
        "chunk_count": chunk_count,
        "files": [path.name for path in staged_paths],
    }


def _safe_error_message(message: object) -> tuple[str, int, str, str]:
    """Return a client-safe error message without upstream bodies or secret values."""
    text = str(message)
    lowered = text.casefold()
    missing_prefix = "missing required environment variables:"
    if missing_prefix in lowered:
        variable_names = [
            name
            for name in _REQUIRED_MODEL_ENV
            if name in text[text.casefold().find(missing_prefix) + len(missing_prefix):]
        ]
        if variable_names:
            return (
                f"Model configuration is missing required variables: {', '.join(variable_names)}",
                502,
                "LLMConfigurationError",
                "missing_configuration",
            )
        return (
            "Model configuration is incomplete; check server-side model settings.",
            502,
            "LLMConfigurationError",
            "missing_configuration",
        )
    if "rag_llm_timeout" in lowered or "rag_llm_max_retries" in lowered:
        return (
            "Model configuration has an invalid timeout or retry setting.",
            502,
            "LLMConfigurationError",
            "invalid_model_settings",
        )

    error_match = re.search(
        r"\b(http_\d{3}|invalid_json|schema_mismatch|timeout|empty_content|network_error)\b",
        lowered,
    )
    reason_match = re.search(r"\breason=([a-z0-9_]+)", lowered)
    if error_match:
        error_type = error_match.group(1)
        reason = reason_match.group(1) if reason_match else error_type
        if error_type.startswith("http_"):
            status = int(error_type[5:])
            if status == 429:
                message_text = "Model failure: provider rate limit (HTTP 429); retry later."
                reason = "rate_limit"
            elif status == 404:
                message_text = "Model failure: model or endpoint was not found (HTTP 404)."
                reason = "model_not_found"
            elif status == 401:
                message_text = "Model failure: provider rejected authentication (HTTP 401)."
                reason = "authentication_failed"
            elif status == 400:
                message_text = "Model failure: provider rejected the request (HTTP 400)."
                reason = "bad_request"
            else:
                message_text = f"Model failure: provider returned HTTP {status}."
                reason = "timeout" if status == 408 else "provider_error"
            return message_text, 502, error_type, reason
        messages = {
            "invalid_json": "Model failure: provider returned invalid JSON.",
            "schema_mismatch": "Model failure: response did not match the required schema.",
            "timeout": "Model failure: request timed out.",
            "empty_content": "Model failure: response content was empty.",
            "network_error": "Model failure: could not connect to the model endpoint.",
        }
        return messages[error_type], 502, error_type, reason

    if any(marker in lowered for marker in ("model request failed", "llm client", "model endpoint")):
        status_match = re.search(r"\bHTTP\s+(\d{3})\b", text, flags=re.IGNORECASE)
        if status_match:
            provider_status = int(status_match.group(1))
            reason = {
                400: "bad_request",
                401: "authentication_failed",
                403: "permission_denied",
                404: "model_not_found",
                408: "timeout",
                425: "temporary_provider_failure",
                429: "rate_limit",
            }.get(provider_status, "provider_error")
            return (
                f"Model failure: endpoint returned HTTP {provider_status}.",
                502,
                f"http_{provider_status}",
                reason,
            )
        return (
            "Model failure: verify the server-side endpoint, model, API key, "
            "and network configuration.",
            502,
            "model_failure",
            "request_failed",
        )

    if any(marker in lowered for marker in ("ingestion failed", "retrieval failed")):
        return (
            "Retrieval failed; check the uploaded documents and server configuration.",
            503,
            "IngestionError",
            "retrieval_failed",
        )
    if "invalid json" in lowered or "valid json" in lowered:
        return "Model failure: provider returned invalid JSON.", 502, "invalid_json", "invalid_json"
    if any(
        marker in lowered
        for marker in (
            "response",
            "action_items",
            "citations",
            "status must",
            "status ok requires",
            "no action items had citations",
        )
    ):
        return (
            "Model failure: the endpoint returned an invalid response; verify the "
            "configured model and its OpenAI-compatible response format.",
            502,
            "schema_mismatch",
            "schema_mismatch",
        )
    return "Internal pipeline error.", 500, "PipelineError", "internal_pipeline_error"


@app.post("/process", response_model=None)
def process_query(payload: ProcessRequest, request: Request) -> JSONResponse | Result:
    model = getattr(request.app.state, "llm_client", None)
    if model is None:
        model = _bounded_model_client()
    try:
        with _knowledge_base_lock:
            result = pipeline.run_pipeline(payload.query, llm=model)
    except Exception as error:
        diagnostic_id = uuid.uuid4().hex
        error_type = type(error).__name__
        logger.exception(
            "Pipeline request failed (diagnostic_id=%s, exception_type=%s)",
            diagnostic_id,
            error_type,
            exc_info=(type(error), Exception(), error.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "Internal pipeline error.",
                "diagnostic_id": diagnostic_id,
                "error_type": error_type,
                "reason": "unexpected_exception",
            },
        )

    result_status = result.get("status") if isinstance(result, dict) else None
    if result_status in {"ok", "insufficient_evidence", "conflict"}:
        return result
    if result_status == "error":
        message, status_code, error_type, reason = _safe_error_message(
            result.get("error", "Pipeline failed")
        )
        content = {
            "status": "error",
            "error": message,
            "error_type": error_type,
            "reason": reason,
        }
        if status_code == 500:
            diagnostic_id = uuid.uuid4().hex
            logger.error(
                "Pipeline returned an internal error (diagnostic_id=%s, error_type=%s)",
                diagnostic_id,
                error_type,
            )
            content["diagnostic_id"] = diagnostic_id
        return JSONResponse(
            status_code=status_code,
            content=content,
        )
    diagnostic_id = uuid.uuid4().hex
    logger.error(
        "Pipeline returned an unknown status (diagnostic_id=%s, result_type=%s)",
        diagnostic_id,
        type(result).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal pipeline error.",
            "error_type": "PipelineError",
            "reason": "invalid_status",
            "diagnostic_id": diagnostic_id,
        },
    )
