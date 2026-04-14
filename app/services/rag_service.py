from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MIN_CHUNK_DISPLAY_SCORE,
    MIN_RELEVANCE_SCORE,
    OPENAI_API_KEY_PRESENT,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    TEMPERATURE,
    TOP_K_RESULTS,
)

OUT_OF_CONTEXT_MESSAGE = "No hay información suficiente."

STOPWORDS = {
    "a",
    "acerca",
    "al",
    "algo",
    "ante",
    "antes",
    "cual",
    "cuales",
    "como",
    "con",
    "contra",
    "de",
    "del",
    "desde",
    "donde",
    "dame",
    "dime",
    "el",
    "en",
    "entre",
    "ese",
    "eso",
    "esta",
    "este",
    "esto",
    "explica",
    "la",
    "las",
    "lo",
    "los",
    "mas",
    "muy",
    "menciona",
    "no",
    "para",
    "por",
    "resume",
    "resumen",
    "que",
    "quien",
    "quienes",
    "indica",
    "lista",
    "se",
    "sin",
    "sobre",
    "su",
    "sus",
    "un",
    "una",
    "uno",
    "y",
}


@dataclass
class ChunkRecord:
    index: int
    page: int | None
    content: str
    score: float | None = None


@dataclass
class DocumentState:
    filename: str | None = None
    pdf_path: str | None = None
    full_text: str = ""
    chunks: list[ChunkRecord] = field(default_factory=list)
    vectorstore: FAISS | None = None
    ready: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "ready": self.ready,
            "chunks": [
                {
                    "index": chunk.index,
                    "page": chunk.page,
                    "content": chunk.content,
                    "score": chunk.score,
                }
                for chunk in self.chunks
            ],
        }


class DocumentManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state = DocumentState()
        self._embeddings: OpenAIEmbeddings | None = None

    def _ensure_openai_ready(self) -> None:
        if not OPENAI_API_KEY_PRESENT:
            raise RuntimeError("Falta configurar OPENAI_API_KEY en el archivo .env.")

    def _get_embeddings(self) -> OpenAIEmbeddings:
        self._ensure_openai_ready()
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        return self._embeddings

    def reset(self) -> None:
        with self._lock:
            self._state = DocumentState()

    def current(self) -> DocumentState:
        with self._lock:
            return self._state

    def has_document(self) -> bool:
        return self.current().ready

    def ingest_text(
        self,
        *,
        filename: str,
        pdf_path: str,
        text: str,
        page_texts: list[tuple[int, str]] | None = None,
    ) -> DocumentState:
        self._ensure_openai_ready()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        documents: list[Document] = []
        chunk_records: list[ChunkRecord] = []

        if page_texts:
            chunk_index = 1
            for page_number, page_text in page_texts:
                page_chunks = splitter.split_text(page_text)
                for chunk in page_chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"index": chunk_index, "page": page_number},
                        )
                    )
                    chunk_records.append(
                        ChunkRecord(
                            index=chunk_index,
                            page=page_number,
                            content=chunk,
                        )
                    )
                    chunk_index += 1
        else:
            chunks = splitter.split_text(text)
            for index, chunk in enumerate(chunks, start=1):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"index": index, "page": None},
                    )
                )
                chunk_records.append(ChunkRecord(index=index, page=None, content=chunk))

        if not documents:
            raise ValueError("No se generaron chunks a partir del texto extraído.")

        vectorstore = FAISS.from_documents(documents, embedding=self._get_embeddings())
        state = DocumentState(
            filename=filename,
            pdf_path=pdf_path,
            full_text=text,
            chunks=chunk_records,
            vectorstore=vectorstore,
            ready=True,
        )
        with self._lock:
            self._state = state
        return state

    def retrieve(self, question: str, top_k: int = TOP_K_RESULTS) -> list[ChunkRecord]:
        retrieved, _, _ = self.retrieve_with_scores(question, top_k=top_k)
        return retrieved

    def _normalize_terms(self, text: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", text.lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        return {
            token
            for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) > 2 and token not in STOPWORDS
        }

    def _term_overlap(self, reference_terms: set[str], candidate_terms: set[str]) -> float:
        if not reference_terms or not candidate_terms:
            return 0.0
        return len(reference_terms & candidate_terms) / len(reference_terms)

    def _chunk_heading_terms(self, chunk: ChunkRecord) -> set[str]:
        preview = " ".join(chunk.content.split()[:12])
        preview = re.sub(r"^\d+(?:\.\d+)*\s*", "", preview)
        return self._normalize_terms(preview)

    def _build_query_variants(self, question: str) -> list[str]:
        normalized = " ".join(question.replace("\n", " ").split()).strip()
        if not normalized:
            return []

        variants: list[str] = [normalized]
        split_patterns = [
            r"\s+(?:y|e|ademas|además|tambien|también)\s+",
            r"[;,]\s+",
        ]

        for pattern in split_patterns:
            pieces = re.split(pattern, normalized, flags=re.IGNORECASE)
            for piece in pieces:
                piece = piece.strip(" .;:?!,-")
                if len(self._normalize_terms(piece)) < 2:
                    continue
                if piece not in variants:
                    variants.append(piece)

        return variants[:4]

    def _split_fragments(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        raw_fragments: list[str] = []

        for line in lines:
            match = re.match(r"^(?:\d+[.)]|[-*•])\s*(.+)$", line)
            if match:
                raw_fragments.append(match.group(1).strip())

        if not raw_fragments:
            normalized = " ".join(text.replace("\n", " ").split())
            raw_fragments = re.split(r"(?<=[\.\?!;:])\s+|\s+\-\s+", normalized)

        fragments: list[str] = []
        for fragment in raw_fragments:
            fragment = fragment.strip(" .;:?!-")
            if not fragment:
                continue
            pieces = [fragment]
            lowered = fragment.lower()
            if " y " in lowered and len(fragment.split()) >= 8:
                candidate_pieces = [part.strip(" .;:?!-") for part in re.split(r"\s+y\s+", fragment)]
                if len(candidate_pieces) >= 2 and all(len(self._normalize_terms(part)) >= 3 for part in candidate_pieces):
                    pieces = candidate_pieces

            for piece in pieces:
                if len(self._normalize_terms(piece)) < 3:
                    continue
                fragments.append(piece)
        return fragments

    def _split_answer_items(self, text: str) -> list[str]:
        items: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            match = re.match(r"^(?:\d+[.)]|[-*•])\s*(.+)$", line)
            if not match:
                continue
            item = match.group(1).strip(" .;:?!-")
            if len(self._normalize_terms(item)) < 3:
                continue
            items.append(item)
        return items

    def _split_answer_sentences(self, text: str) -> list[str]:
        normalized = " ".join(text.replace("\n", " ").split())
        if not normalized:
            return []

        fragments = re.split(r"(?<=[\.\?!;:])\s+", normalized)
        sentences: list[str] = []
        for fragment in fragments:
            fragment = fragment.strip(" .;:?!-")
            if not fragment:
                continue
            if len(self._normalize_terms(fragment)) < 3:
                continue
            sentences.append(fragment)
        return sentences

    def _score_fragment_chunk(self, fragment: str, question_terms: set[str], chunk: ChunkRecord) -> float:
        fragment_terms = self._normalize_terms(fragment)
        chunk_terms = self._normalize_terms(chunk.content)
        heading_terms = self._chunk_heading_terms(chunk)
        overlap = self._term_overlap(fragment_terms, chunk_terms)
        question_overlap = self._term_overlap(question_terms, chunk_terms)
        heading_overlap = self._term_overlap(fragment_terms, heading_terms)
        base_score = chunk.score or 0.0
        return (0.45 * base_score) + (0.30 * overlap) + (0.15 * question_overlap) + (0.10 * heading_overlap)

    def _select_display_chunks(
        self,
        chunks: list[ChunkRecord],
        *,
        question: str,
        answer_text: str,
        best_score: float,
        limit: int = 4,
    ) -> list[ChunkRecord]:
        if not chunks:
            return []

        ordered_chunks = sorted(
            [chunk for chunk in chunks if chunk.score is not None],
            key=lambda chunk: chunk.score or 0.0,
            reverse=True,
        )
        if not ordered_chunks:
            return []

        question_terms = self._normalize_terms(question)
        question_fragments = self._split_fragments(question)
        answer_fragments = self._split_answer_sentences(answer_text) if answer_text.strip() else []
        fragments = question_fragments[:]

        selected: list[ChunkRecord] = []
        seen_indices: set[int] = set()

        for fragment in fragments:
            fragment_terms = self._normalize_terms(fragment)
            if len(fragment_terms) < 3:
                continue

            best_chunk: ChunkRecord | None = None
            best_fragment_score = 0.0

            for chunk in ordered_chunks:
                if chunk.index in seen_indices:
                    continue
                chunk_terms = self._normalize_terms(chunk.content)
                overlap = self._term_overlap(fragment_terms, chunk_terms)
                question_overlap = self._term_overlap(question_terms, chunk_terms)
                heading_overlap = self._term_overlap(fragment_terms, self._chunk_heading_terms(chunk))
                score = self._score_fragment_chunk(fragment, question_terms, chunk)

                if overlap < 0.08 and question_overlap < 0.08 and heading_overlap < 0.08:
                    continue
                if question_fragments and question_overlap < 0.08 and heading_overlap < 0.08:
                    continue
                if score > best_fragment_score:
                    best_fragment_score = score
                    best_chunk = chunk

            if best_chunk is not None and best_chunk.index not in seen_indices:
                selected.append(best_chunk)
                seen_indices.add(best_chunk.index)

            if len(selected) >= limit:
                break

        if not selected:
            selected = [ordered_chunks[0]]
            seen_indices.add(ordered_chunks[0].index)

        if len(selected) < limit and len(answer_fragments) > 1:
            supplemental_score_floor = max(MIN_CHUNK_DISPLAY_SCORE, best_score - 0.30)
            for fragment in answer_fragments:
                fragment_terms = self._normalize_terms(fragment)
                if len(fragment_terms) < 3:
                    continue

                best_chunk: ChunkRecord | None = None
                best_fragment_score = 0.0

                for chunk in ordered_chunks:
                    if chunk.index in seen_indices:
                        continue
                    if (chunk.score or 0.0) < supplemental_score_floor:
                        continue

                    chunk_terms = self._normalize_terms(chunk.content)
                    fragment_overlap = self._term_overlap(fragment_terms, chunk_terms)
                    if fragment_overlap < 0.22:
                        continue

                    question_overlap = self._term_overlap(question_terms, chunk_terms)
                    if question_overlap < 0.08:
                        continue

                    candidate_score = (
                        (0.60 * fragment_overlap)
                        + (0.25 * question_overlap)
                        + (0.10 * (chunk.score or 0.0))
                    )
                    if candidate_score > best_fragment_score:
                        best_fragment_score = candidate_score
                        best_chunk = chunk

                if best_chunk is not None:
                    selected.append(best_chunk)
                    seen_indices.add(best_chunk.index)

                if len(selected) >= limit:
                    break

        return selected[:limit]

    def retrieve_with_scores(
        self,
        question: str,
        top_k: int = TOP_K_RESULTS,
    ) -> tuple[list[ChunkRecord], bool, float]:
        state = self.current()
        if not state.ready or state.vectorstore is None:
            return [], False, 0.0

        try:
            query_variants = self._build_query_variants(question)
            if not query_variants:
                return [], False, 0.0

            per_query_k = max(8, top_k)
            best_by_index: dict[int, ChunkRecord] = {}

            for query in query_variants:
                scored_results = state.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=per_query_k,
                )
                for i, (doc, score) in enumerate(scored_results):
                    index = int(doc.metadata.get("index", i + 1))
                    chunk = ChunkRecord(
                        index=index,
                        page=int(doc.metadata["page"]) if doc.metadata.get("page") is not None else None,
                        content=doc.page_content,
                        score=float(score),
                    )
                    existing = best_by_index.get(index)
                    if existing is None or (chunk.score or 0.0) > (existing.score or 0.0):
                        best_by_index[index] = chunk

            chunks = sorted(best_by_index.values(), key=lambda chunk: chunk.score or 0.0, reverse=True)
            best_score = max((chunk.score or 0.0) for chunk in chunks) if chunks else 0.0
            is_relevant = bool(chunks) and best_score >= MIN_RELEVANCE_SCORE
            return chunks, is_relevant, best_score
        except Exception:
            docs = state.vectorstore.similarity_search(question, k=top_k)
            chunks = [
                ChunkRecord(
                    index=int(doc.metadata.get("index", i + 1)),
                    page=int(doc.metadata["page"]) if doc.metadata.get("page") is not None else None,
                    content=doc.page_content,
                    score=None,
                )
                for i, doc in enumerate(docs)
            ]
            return chunks, False, 0.0

    def answer(self, question: str, top_k: int = TOP_K_RESULTS) -> tuple[str, list[ChunkRecord], bool]:
        self._ensure_openai_ready()

        candidate_top_k = max(top_k * 4, 12)
        retrieved_chunks, has_context, best_score = self.retrieve_with_scores(
            question,
            top_k=candidate_top_k,
        )
        usable_chunks = [chunk for chunk in retrieved_chunks if chunk.content.strip()]

        if not usable_chunks or not has_context:
            return OUT_OF_CONTEXT_MESSAGE, [], True

        answer_context_chunks = usable_chunks[:candidate_top_k]
        context = "\n\n".join(
            [
                f"Chunk {chunk.index}"
                + (f" (Página {chunk.page})" if chunk.page is not None else "")
                + f":\n{chunk.content}"
                for chunk in answer_context_chunks
            ]
        )

        system_prompt = (
            "Eres un asistente académico en español. Responde usando únicamente el contexto recuperado del PDF. "
            "Si el contexto no alcanza para responder de forma precisa, responde exactamente con: "
            f"\"{OUT_OF_CONTEXT_MESSAGE}\" "
            "No inventes datos, no adivines y no uses conocimiento externo."
        )
        user_prompt = (
            f"Contexto recuperado:\n{context}\n\n"
            f"Pregunta del usuario: {question}\n\n"
            "Redacta una respuesta clara, breve y útil basada solo en el contexto recuperado."
        )

        llm = ChatOpenAI(
            model=OPENAI_CHAT_MODEL,
            temperature=TEMPERATURE,
        )

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        answer_text = response.content.strip()
        if not answer_text:
            return OUT_OF_CONTEXT_MESSAGE, [], True

        supplemental_chunks, supplemental_relevant, supplemental_best_score = self.retrieve_with_scores(
            answer_text,
            top_k=candidate_top_k,
        )
        merged_chunks: dict[int, ChunkRecord] = {}
        for chunk in usable_chunks + supplemental_chunks:
            existing = merged_chunks.get(chunk.index)
            if existing is None or (chunk.score or 0.0) > (existing.score or 0.0):
                merged_chunks[chunk.index] = chunk

        display_chunks = self._select_display_chunks(
            sorted(merged_chunks.values(), key=lambda chunk: chunk.score or 0.0, reverse=True),
            question=question,
            answer_text=answer_text,
            best_score=max(best_score, supplemental_best_score if supplemental_relevant else 0.0),
            limit=4,
        )

        return answer_text, display_chunks, False


document_manager = DocumentManager()
