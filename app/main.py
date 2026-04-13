from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.config import MAX_UPLOAD_MB, UPLOAD_DIR
from app.services.pdf_service import PDFProcessingError, extract_text_from_pdf, validate_pdf_filename
from app.services.rag_service import document_manager

app = FastAPI(title="Chat con tu PDF", version="1.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

CURRENT_PDF_PATH = UPLOAD_DIR / "current.pdf"


def _state_payload(message: str | None = None) -> dict:
    state = document_manager.current()
    payload = state.to_public_dict()
    payload["message"] = message or (
        "PDF cargado y listo para preguntas." if state.ready else "Sube un PDF para comenzar."
    )
    return payload


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "initial_state": json.dumps(_state_payload(), ensure_ascii=False),
        },
    )


@app.get("/pdf")
async def get_pdf() -> FileResponse:
    if not CURRENT_PDF_PATH.exists():
        raise HTTPException(status_code=404, detail="No hay un PDF cargado.")
    headers = {
        "Content-Disposition": f'inline; filename="{CURRENT_PDF_PATH.name}"'
    }
    return FileResponse(
        CURRENT_PDF_PATH,
        media_type="application/pdf",
        headers=headers,
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile | None = File(None)) -> JSONResponse:
    if file is None or not file.filename:
        return JSONResponse(status_code=400, content={"ok": False, "error": "No se recibió ningún archivo."})

    try:
        validate_pdf_filename(file.filename)
    except PDFProcessingError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})

    contents = await file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"ok": False, "error": "El archivo está vacío."})

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_bytes:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"El PDF supera el límite de {MAX_UPLOAD_MB} MB."},
        )

    CURRENT_PDF_PATH.write_bytes(contents)

    try:
        text = extract_text_from_pdf(CURRENT_PDF_PATH)
        state = document_manager.ingest_text(
            filename=file.filename,
            pdf_path=str(CURRENT_PDF_PATH),
            text=text,
        )
    except RuntimeError as exc:
        document_manager.reset()
        if CURRENT_PDF_PATH.exists():
            CURRENT_PDF_PATH.unlink()
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})
    except PDFProcessingError as exc:
        document_manager.reset()
        if CURRENT_PDF_PATH.exists():
            CURRENT_PDF_PATH.unlink()
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    except Exception as exc:
        document_manager.reset()
        if CURRENT_PDF_PATH.exists():
            CURRENT_PDF_PATH.unlink()
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Error procesando el PDF: {exc}"},
        )

    return JSONResponse(
        content={
            "ok": True,
            "message": f"PDF '{state.filename}' cargado correctamente.",
            "state": state.to_public_dict(),
            "pdf_url": "/pdf",
        }
    )


@app.post("/ask")
async def ask_question(question: str = Form("")) -> JSONResponse:
    question = question.strip()
    if not question:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Debes escribir una pregunta."})

    if not document_manager.has_document():
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "Primero debes subir y procesar un PDF."},
        )

    try:
        answer, chunks, out_of_context = document_manager.answer(question)
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Error de API al generar la respuesta: {exc}"},
        )

    return JSONResponse(
        content={
            "ok": True,
            "answer": answer,
            "out_of_context": out_of_context,
            "chunks": [] if out_of_context else [
                {"index": chunk.index, "content": chunk.content}
                for chunk in chunks[:3]
            ],
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"ok": True})
