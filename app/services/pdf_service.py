from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader


class PDFProcessingError(Exception):
    pass


def validate_pdf_filename(filename: str) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("El archivo debe tener extensión .pdf.")


def extract_text_from_pdf(pdf_path: Path) -> str:
    pages = extract_text_from_pdf_pages(pdf_path)
    return "\n\n".join(page_text for _, page_text in pages).strip()


def extract_text_from_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PDFProcessingError("No se pudo abrir el PDF. Puede estar dañado.") from exc

    if not reader.pages:
        raise PDFProcessingError("El PDF está vacío o no contiene páginas.")

    pages_text: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            pages_text.append((page_number, page_text.strip()))

    if not pages_text:
        raise PDFProcessingError(
            "No se encontró texto extraíble en el PDF. Puede estar vacío o ser solo imagen."
        )
    return pages_text
