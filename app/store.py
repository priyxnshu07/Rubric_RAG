"""ChromaDB wrapper: store chunks for a rubric and retrieve the most relevant ones."""
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app import config
from app.chunking import chunk_notes, chunk_rubric
from app.schemas import Chunk

COLLECTION = "rubricrag"


def _to_metadata(rubric_id: str, c: Chunk) -> dict:
    meta = {"rubric_id": rubric_id, "chunk_id": c.id, "doc_type": c.doc_type, "source": c.source,
            "criterion": c.criterion, "start_line": c.start_line, "end_line": c.end_line}
    if c.points is not None:  # Chroma metadata cannot hold None
        meta["points"] = c.points
    return meta


def _from_result(doc: str, m: dict) -> Chunk:
    return Chunk(id=m["chunk_id"], text=doc, doc_type=m["doc_type"], start_line=m["start_line"],
                 end_line=m["end_line"], source=m.get("source", ""), criterion=m.get("criterion", ""),
                 points=m.get("points"))


class Store:
    def __init__(self, path: str = config.CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            COLLECTION,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL),
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, rubric_id: str, rubric_text: str, notes_text: str = "",
               rubric_source: str = "rubric", notes_source: str = "notes") -> tuple[int, int]:
        notes = [(notes_source, notes_text)] if notes_text.strip() else []
        return self.ingest_documents(rubric_id, (rubric_source, rubric_text), notes)

    def ingest_documents(self, rubric_id: str, rubric: tuple[str, str],
                         notes: list[tuple[str, str]]) -> tuple[int, int]:
        """Replace any existing chunks for this rubric_id. rubric/notes are (file name, text)."""
        self.collection.delete(where={"rubric_id": rubric_id})
        rubric_chunks = chunk_rubric(rubric[1], source=rubric[0])
        notes_chunks = []
        for name, text in notes:
            # With several notes files, prefix ids with the file name so they stay unique.
            prefix = "N" if len(notes) == 1 else f"{Path(name).stem}:N"
            notes_chunks += chunk_notes(text, source=name, prefix=prefix)
        chunks = rubric_chunks + notes_chunks
        if chunks:
            self.collection.add(
                ids=[f"{rubric_id}:{c.id}" for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[_to_metadata(rubric_id, c) for c in chunks],
            )
        return len(rubric_chunks), len(notes_chunks)

    def list_rubrics(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["rubric_id"] for m in metas})

    def get_chunks(self, rubric_id: str, doc_type: str) -> list[Chunk]:
        res = self.collection.get(where={"$and": [{"rubric_id": rubric_id}, {"doc_type": doc_type}]})
        chunks = [_from_result(d, m) for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: c.start_line)

    def retrieve(self, rubric_id: str, query: str, doc_type: str, k: int) -> list[Chunk]:
        where = {"$and": [{"rubric_id": rubric_id}, {"doc_type": doc_type}]}
        available = len(self.collection.get(where=where)["ids"])
        if available == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, available), where=where)
        chunks = [_from_result(d, m) for d, m in zip(res["documents"][0], res["metadatas"][0])]
        # Present chunks in document order so the LLM sees the rubric as written.
        return sorted(chunks, key=lambda c: (c.source, c.start_line))
