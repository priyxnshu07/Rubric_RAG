"""ChromaDB wrapper: store chunks for a rubric and retrieve the most relevant ones."""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app import config
from app.chunking import chunk_notes, chunk_rubric
from app.schemas import Chunk

COLLECTION = "rubricrag"


class Store:
    def __init__(self, path: str = config.CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            COLLECTION,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL),
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, rubric_id: str, rubric_text: str, notes_text: str = "") -> tuple[int, int]:
        """Replace any existing chunks for this rubric_id with fresh ones."""
        self.collection.delete(where={"rubric_id": rubric_id})
        rubric_chunks = chunk_rubric(rubric_text)
        notes_chunks = chunk_notes(notes_text)
        chunks = rubric_chunks + notes_chunks
        if chunks:
            self.collection.add(
                ids=[f"{rubric_id}:{c.id}" for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[{"rubric_id": rubric_id, "chunk_id": c.id, "doc_type": c.doc_type,
                            "start_line": c.start_line, "end_line": c.end_line} for c in chunks],
            )
        return len(rubric_chunks), len(notes_chunks)

    def has_rubric(self, rubric_id: str) -> bool:
        return bool(self.collection.get(where={"rubric_id": rubric_id}, limit=1)["ids"])

    def list_rubrics(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["rubric_id"] for m in metas})

    def retrieve(self, rubric_id: str, query: str, doc_type: str, k: int) -> list[Chunk]:
        where = {"$and": [{"rubric_id": rubric_id}, {"doc_type": doc_type}]}
        available = len(self.collection.get(where=where)["ids"])
        if available == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, available), where=where)
        chunks = [
            Chunk(id=m["chunk_id"], text=doc, doc_type=m["doc_type"],
                  start_line=m["start_line"], end_line=m["end_line"])
            for doc, m in zip(res["documents"][0], res["metadatas"][0])
        ]
        # Present chunks in document order so the LLM sees the rubric as written.
        return sorted(chunks, key=lambda c: c.start_line)
