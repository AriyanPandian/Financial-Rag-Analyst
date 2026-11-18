from contextlib import asynccontextmanager
import hmac
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import AppError
from app.service import RAGService


settings = Settings.from_env()


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.rag = RAGService(settings)
    yield


app = FastAPI(
    title="Financial RAG Analyst",
    description="Ask evidence-grounded questions about indexed financial filings.",
    version="1.0.0",
    lifespan=lifespan,
)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.middleware("http")
async def protect_api(request: Request, call_next):
    if request.url.path == "/api/documents" and request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > settings.max_upload_bytes + 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "The uploaded file exceeds the configured size limit."},
                )
    if (
        settings.api_access_token
        and request.url.path.startswith("/api/")
        and request.url.path != "/api/health"
    ):
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401, content={"detail": "A valid API token is required."}
            )
        supplied = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied, settings.api_access_token):
            return JSONResponse(
                status_code=401, content={"detail": "A valid API token is required."}
            )
    return await call_next(request)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "openai_configured": bool(settings.openai_api_key),
        "authentication_enabled": bool(settings.api_access_token),
        "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
    }


@app.get("/api/documents")
async def list_documents(request: Request):
    return {"documents": request.app.state.rag.documents()}


@app.post("/api/documents", status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...)):
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise AppError("The uploaded file exceeds the configured size limit.", 413)
    return request.app.state.rag.ingest(file.filename or "", content)


@app.delete("/api/documents/{document_id}", status_code=204)
async def delete_document(document_id: str, request: Request):
    request.app.state.rag.delete_document(document_id)


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise AppError("Question must not be blank.")
    return request.app.state.rag.answer(
        question,
        [message.model_dump() for message in payload.history],
    )
