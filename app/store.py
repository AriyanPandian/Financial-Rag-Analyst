from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any

import chromadb


class DocumentStore:
    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name="financial_documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._db = sqlite3.connect(data_dir / "documents.sqlite3", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    def has_document(self, document_id: str) -> bool:
        with self._lock:
            return self._db.execute(
                "SELECT 1 FROM documents WHERE id = ?", (document_id,)
            ).fetchone() is not None

    def add_document(
        self,
        document_id: str,
        name: str,
        content_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ids = [f"{document_id}_{index}" for index in range(len(chunks))]
        with self._lock:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
            try:
                with self._db:
                    self._db.execute(
                        """INSERT INTO documents
                        (id, name, content_hash, chunk_count, created_at)
                        VALUES (?, ?, ?, ?, ?)""",
                        (
                            document_id,
                            name,
                            content_hash,
                            len(chunks),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
            except Exception:
                self._collection.delete(where={"document_id": document_id})
                raise

    def query(
        self, embedding: list[float], count: int
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self._collection.count() == 0:
                return []
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(count, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {"text": text, "metadata": metadata, "distance": distance}
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """SELECT id, name, chunk_count, created_at
                FROM documents ORDER BY created_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            exists = self._db.execute(
                "SELECT 1 FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if exists is None:
                return False
            self._collection.delete(where={"document_id": document_id})
            with self._db:
                self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return True
