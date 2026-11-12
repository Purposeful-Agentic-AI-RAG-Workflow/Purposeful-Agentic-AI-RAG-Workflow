"""FastAPI interface for the bounded citizen-feedback RAG pipeline."""

from __future__ import annotations

import logging
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import pipeline
from ingest import IngestionError
from models import Result

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ALLOWED_SUFFIXES = {".txt", ".md"}
MAX_UPLOAD_FILES = 10
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_UPLOAD_BYTES = 5_000_000
MAX_QUERY_CHARS = 1_000
_knowledge_base_lock = threading.RLock()


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
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        (
            path
            for path in DATA_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
        ),
        key=lambda path: path.relative_to(DATA_DIR).as_posix().casefold(),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
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
def healthz() -> dict[str, str]:
    return {"status": "ok"}


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
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for upload in files:
            safe_name = _safe_upload_name(upload.filename)
            destination = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
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


def _error_status_code(message: str) -> int:
    lowered = message.casefold()
    if any(marker in lowered for marker in ("model request failed", "llm client", "model endpoint")):
        return 502
    if any(
        marker in lowered
        for marker in (
            "ingestion failed",
            "retrieval failed",
            "data directory",
            "no .txt or .md files",
        )
    ):
        return 503
    return 500


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
        logger.exception("Pipeline request failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(error)},
        )

    if result["status"] in {"ok", "insufficient_evidence", "conflict"}:
        return result
    if result["status"] == "error":
        message = result.get("error", "Pipeline failed")
        return JSONResponse(
            status_code=_error_status_code(message),
            content={"status": "error", "error": message},
        )
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": "Pipeline returned an unknown status"},
    )
