"""
Embeddings & retrieval layer — embeds table schema chunks into a
local vector store so the agent can retrieve only relevant tables
for a given user question, instead of stuffing the full schema
into every prompt.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from sql_copilot.schema import TableInfo, table_to_text_chunk

_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for schema text
_COLLECTION_NAME = "schema_chunks"


class SchemaRetriever:
    def __init__(self, persist_path: str = "./chroma_store"):
        self.model = SentenceTransformer(_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_path)
        # Reset the collection each time we index, so stale schema info
        # from a previous DB never lingers.
        try:
            self.client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.create_collection(_COLLECTION_NAME)

    def index_schema(self, tables: list[TableInfo]) -> None:
        """Embeds every table's text chunk and stores it in the vector DB."""
        chunks = [table_to_text_chunk(t) for t in tables]
        ids = [t.name for t in tables]
        embeddings = self.model.encode(chunks).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"table_name": t.name} for t in tables],
        )

    def retrieve_relevant_tables(self, user_query: str, top_k: int = 3) -> list[str]:
        """
        Given a natural-language question, returns the text chunks
        of the top_k most relevant tables to include in the LLM prompt.
        """
        query_embedding = self.model.encode([user_query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        return results["documents"][0] if results["documents"] else []