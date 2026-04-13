from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader


class PDFProcessingError(Exception):
    pass


def validate_pdf_filename(filename: str) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("El archivo debe tener extensión .pdf.")


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PDFProcessingError("No se pudo abrir el PDF. Puede estar dañado.") from exc

    if not reader.pages:
        raise PDFProcessingError("El PDF está vacío o no contiene páginas.")

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            pages_text.append(page_text.strip())

    text = "\n\n".join(pages_text).strip()
    if not text:
        raise PDFProcessingError(
            "No se encontró texto extraíble en el PDF. Puede estar vacío o ser solo imagen."
        )
    return text

