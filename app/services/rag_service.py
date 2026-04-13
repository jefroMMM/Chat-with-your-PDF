from __future__ import annotations

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
    MIN_RELEVANCE_SCORE,
    OPENAI_API_KEY_PRESENT,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    TEMPERATURE,
    TOP_K_RESULTS,
)

OUT_OF_CONTEXT_MESSAGE = "No hay información suficiente."


@dataclass
class ChunkRecord:
    index: int
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

    def ingest_text(self, *, filename: str, pdf_path: str, text: str) -> DocumentState:
        self._ensure_openai_ready()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        if not chunks:
            raise ValueError("No se generaron chunks a partir del texto extraído.")

        documents = [
            Document(page_content=chunk, metadata={"index": i + 1})
            for i, chunk in enumerate(chunks)
        ]
        vectorstore = FAISS.from_documents(documents, embedding=self._get_embeddings())
        state = DocumentState(
            filename=filename,
            pdf_path=pdf_path,
            full_text=text,
            chunks=[ChunkRecord(index=i + 1, content=chunk) for i, chunk in enumerate(chunks)],
            vectorstore=vectorstore,
            ready=True,
        )
        with self._lock:
            self._state = state
        return state

    def retrieve(self, question: str, top_k: int = TOP_K_RESULTS) -> list[ChunkRecord]:
        retrieved, _, _ = self.retrieve_with_scores(question, top_k=top_k)
        return retrieved

    def retrieve_with_scores(
        self,
        question: str,
        top_k: int = TOP_K_RESULTS,
    ) -> tuple[list[ChunkRecord], bool, float]:
        state = self.current()
        if not state.ready or state.vectorstore is None:
            return [], False, 0.0

        try:
            scored_results = state.vectorstore.similarity_search_with_relevance_scores(
                question,
                k=top_k,
            )
            chunks = [
                ChunkRecord(
                    index=int(doc.metadata.get("index", i + 1)),
                    content=doc.page_content,
                    score=float(score),
                )
                for i, (doc, score) in enumerate(scored_results)
            ]
            best_score = max((chunk.score or 0.0) for chunk in chunks) if chunks else 0.0
            is_relevant = bool(chunks) and best_score >= MIN_RELEVANCE_SCORE
            return chunks, is_relevant, best_score
        except Exception:
            docs = state.vectorstore.similarity_search(question, k=top_k)
            chunks = [
                ChunkRecord(
                    index=int(doc.metadata.get("index", i + 1)),
                    content=doc.page_content,
                    score=None,
                )
                for i, doc in enumerate(docs)
            ]
            return chunks, False, 0.0

    def answer(self, question: str, top_k: int = TOP_K_RESULTS) -> tuple[str, list[ChunkRecord], bool]:
        self._ensure_openai_ready()

        retrieved_chunks, has_context, _best_score = self.retrieve_with_scores(
            question,
            top_k=top_k,
        )
        usable_chunks = [chunk for chunk in retrieved_chunks if chunk.content.strip()]

        if not usable_chunks or not has_context:
            return OUT_OF_CONTEXT_MESSAGE, [], True

        context = "\n\n".join(
            [f"Chunk {chunk.index}:\n{chunk.content}" for chunk in usable_chunks]
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

        return answer_text, usable_chunks, False


document_manager = DocumentManager()
