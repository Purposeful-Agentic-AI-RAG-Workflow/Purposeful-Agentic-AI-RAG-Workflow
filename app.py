"""FastAPI interface for the bounded citizen-feedback RAG pipeline."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

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


def _safe_error_message(message: object) -> tuple[str, int, str]:
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
            )
        return (
            "Model configuration is incomplete; check server-side model settings.",
            502,
            "LLMConfigurationError",
        )

    if any(marker in lowered for marker in ("model request failed", "llm client", "model endpoint")):
        status_match = re.search(r"\bHTTP\s+(\d{3})\b", text, flags=re.IGNORECASE)
        if status_match:
            provider_status = int(status_match.group(1))
            return (
                f"Model failure: endpoint returned HTTP {provider_status}; verify the "
                "configured endpoint, model, and API key.",
                502,
                "LLMClientError",
            )
        return (
            "Model failure: verify the server-side endpoint, model, API key, "
            "and network configuration.",
            502,
            "LLMClientError",
        )

    if any(marker in lowered for marker in ("ingestion failed", "retrieval failed")):
        return (
            "Retrieval failed; check the uploaded documents and server configuration.",
            503,
            "IngestionError",
        )
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
            "LLMResponseError",
        )
    return "Internal pipeline error.", 500, "PipelineError"


@app.post("/process", response_model=None)
def process_query(payload: ProcessRequest, request: Request) -> JSONResponse | Result:
    model = getattr(request.app.state, "llm_client", None)
    try:
        with _knowledge_base_lock:
            result = (
                pipeline.run_pipeline(payload.query, llm=model)
                if model is not None
                else pipeline.run_pipeline(payload.query)
            )
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
            },
        )

    result_status = result.get("status") if isinstance(result, dict) else None
    if result_status in {"ok", "insufficient_evidence", "conflict"}:
        return result
    if result_status == "error":
        message, status_code, error_type = _safe_error_message(
            result.get("error", "Pipeline failed")
        )
        content = {"status": "error", "error": message, "error_type": error_type}
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
            "diagnostic_id": diagnostic_id,
        },
    )
