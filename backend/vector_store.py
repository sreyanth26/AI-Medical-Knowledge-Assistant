"""
Step 5: ChromaDB Vector Store
Stores document embeddings and performs semantic similarity search.

Why ChromaDB over FAISS:
  - Built-in persistent storage (SQLite under the hood)
  - Metadata filtering support (filter by source, date, etc.)
  - No manual save/load needed — auto-persists on every write
  - Better suited for growing document collections
  - Rich query API with where-clause filtering

Run standalone test:
    python vector_store.py
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from embeddings import get_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = "models/chroma_db"
DEFAULT_COLLECTION  = "medical_knowledge"


class MedicalVectorStore:
    """
    Wrapper around ChromaDB for storing and retrieving medical document embeddings.

    Key differences from FAISS:
      - No manual save() needed — ChromaDB auto-persists every write to disk
      - Supports metadata filtering: search only within specific documents
      - Uses SQLite internally — human-readable, inspectable database
      - Collection-based: documents are grouped into named collections

    Usage:
        store = MedicalVectorStore()
        store.add_documents(chunks)            # index + auto-persist
        results = store.search("fever meds")   # semantic search
        results = store.search(               # filtered search
            "fever meds",
            filter={"source": "who_dengue.txt"}
        )
        store.clear()                          # wipe collection
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        """
        Args:
            persist_directory:  Folder where ChromaDB stores its SQLite database.
            collection_name:    Name of the ChromaDB collection (like a table name).
            embedding_model_name: HuggingFace model for generating embeddings.
            top_k:              Default number of results to return per search.
        """
        self.persist_directory = persist_directory
        self.collection_name   = collection_name
        self.top_k             = top_k
        self.embedding_model   = get_embedding_model(embedding_model_name)

        # Ensure storage directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize persistent ChromaDB client
        self._client = chromadb.PersistentClient(path=persist_directory)

        # Initialize LangChain Chroma wrapper
        self.vectorstore: Chroma = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )

        count = self._doc_count()
        if count > 0:
            logger.info(
                f"ChromaDB loaded — collection='{collection_name}' "
                f"| {count} chunks | path='{persist_directory}'"
            )
        else:
            logger.info(
                f"ChromaDB ready (empty) — collection='{collection_name}' "
                f"| path='{persist_directory}'"
            )

    # ──────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add document chunks to ChromaDB.
        ChromaDB auto-persists — no manual save() needed.

        Args:
            documents: List of LangChain Document objects.
        """
        if not documents:
            logger.warning("No documents provided to add_documents().")
            return

        logger.info(f"Indexing {len(documents)} chunks into ChromaDB...")

        # Add in batches of 100 to avoid memory issues with large uploads
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(f"  Indexed batch {i // batch_size + 1} ({len(batch)} chunks)")

        logger.info(
            f"Indexing complete. Total chunks in DB: {self._doc_count()}"
        )

    # ──────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Retrieve the most semantically similar chunks for a query.

        Args:
            query:  Natural-language question.
            top_k:  Number of results (defaults to self.top_k).
            filter: Optional ChromaDB metadata filter.
                    Example: {"source": "who_dengue.txt"}
                    Example: {"$and": [{"source": "drug_manual.txt"}]}

        Returns:
            List of Document objects ordered by relevance.

        Raises:
            RuntimeError: If the collection is empty.
        """
        if self.is_empty:
            raise RuntimeError(
                "Vector store is empty. Add documents or upload files first."
            )

        k      = top_k or self.top_k
        kwargs = {"k": k}
        if filter:
            kwargs["filter"] = filter

        logger.info(
            f"ChromaDB search | query='{query[:60]}' | top_k={k}"
            + (f" | filter={filter}" if filter else "")
        )

        results = self.vectorstore.similarity_search(query, **kwargs)
        return results

    def search_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Same as search() but also returns relevance scores.

        Returns:
            List of (Document, score) tuples.
            ChromaDB returns distance scores — lower = more similar.
        """
        if self.is_empty:
            raise RuntimeError("Vector store is empty.")

        k      = top_k or self.top_k
        kwargs = {"k": k}
        if filter:
            kwargs["filter"] = filter

        return self.vectorstore.similarity_search_with_score(query, **kwargs)

    def search_by_source(
        self, query: str, source_filename: str, top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Search within a specific document only.
        This is a ChromaDB advantage over FAISS — metadata filtering.

        Args:
            query:           Search query.
            source_filename: Exact filename to filter by (e.g. 'drug_manual.txt').
            top_k:           Number of results.

        Example:
            results = store.search_by_source(
                "paracetamol dosage",
                source_filename="drug_reference.txt"
            )
        """
        return self.search(
            query,
            top_k=top_k,
            filter={"source": source_filename},
        )

    # ──────────────────────────────────────────────────────────────
    # Persistence (ChromaDB auto-persists — these are convenience methods)
    # ──────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        """
        ChromaDB auto-persists on every write — no manual save needed.
        This method exists for API compatibility with the old FAISS store.
        """
        logger.info(
            "ChromaDB auto-persists on every write. "
            f"Data is already saved at '{self.persist_directory}'."
        )

    def load(self, path: Optional[str] = None) -> None:
        """
        ChromaDB auto-loads on init from persist_directory.
        This method exists for API compatibility with the old FAISS store.
        """
        logger.info("ChromaDB auto-loads on startup. No manual load needed.")

    def clear(self) -> None:
        """
        Delete all documents from the collection.
        The collection itself remains — it just becomes empty.
        """
        try:
            self._client.delete_collection(self.collection_name)
            # Re-create empty collection
            self.vectorstore = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' cleared.")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")
            raise

    def delete_by_source(self, source_filename: str) -> int:
        """
        Delete all chunks from a specific source document.
        ChromaDB advantage: can surgically remove one document's chunks.

        Args:
            source_filename: The filename to delete (e.g. 'drug_manual.txt').

        Returns:
            Number of chunks deleted.
        """
        try:
            collection = self._client.get_collection(self.collection_name)
            results    = collection.get(where={"source": source_filename})
            ids        = results.get("ids", [])

            if ids:
                collection.delete(ids=ids)
                logger.info(
                    f"Deleted {len(ids)} chunks for source='{source_filename}'"
                )
            else:
                logger.warning(f"No chunks found for source='{source_filename}'")

            return len(ids)
        except Exception as e:
            logger.error(f"delete_by_source failed: {e}")
            return 0

    def wipe_storage(self) -> None:
        """
        Completely delete the ChromaDB folder from disk.
        Use with caution — all data is permanently lost.
        """
        path = Path(self.persist_directory)
        if path.exists():
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"ChromaDB storage wiped: '{self.persist_directory}'")
        # Re-initialize
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self.vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
        )

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    def _doc_count(self) -> int:
        """Return number of chunks currently in the collection."""
        try:
            collection = self._client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        """
        Return a list of all unique source filenames in the collection.
        ChromaDB advantage: can query metadata directly.
        """
        try:
            collection = self._client.get_collection(self.collection_name)
            results    = collection.get(include=["metadatas"])
            sources    = list({
                m.get("source", "unknown")
                for m in results.get("metadatas", [])
                if m
            })
            return sorted(sources)
        except Exception as e:
            logger.error(f"list_sources failed: {e}")
            return []

    @property
    def is_empty(self) -> bool:
        """True if no documents have been indexed yet."""
        return self._doc_count() == 0

    @property
    def doc_count(self) -> int:
        """Total number of chunks in the collection."""
        return self._doc_count()


# ──────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    from langchain.schema import Document

    sample_docs = [
        Document(
            page_content="Paracetamol reduces fever by acting on the hypothalamus.",
            metadata={"source": "drug_manual.txt", "chunk_index": 0},
        ),
        Document(
            page_content="Dengue fever treatment is mainly supportive: rest, fluids, paracetamol.",
            metadata={"source": "who_guidelines.txt", "chunk_index": 1},
        ),
        Document(
            page_content="Amoxicillin treats bacterial infections like pneumonia.",
            metadata={"source": "drug_manual.txt", "chunk_index": 2},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        store = MedicalVectorStore(persist_directory=tmp)

        print("Adding documents...")
        store.add_documents(sample_docs)
        print(f"Total chunks: {store.doc_count}")
        print(f"Sources: {store.list_sources()}")

        print("\nSearch: 'treatment for dengue'")
        for doc in store.search("treatment for dengue", top_k=2):
            print(f"  [{doc.metadata['source']}] {doc.page_content[:80]}")

        print("\nFiltered search: only drug_manual.txt")
        for doc in store.search_by_source("fever medicine", "drug_manual.txt"):
            print(f"  [{doc.metadata['source']}] {doc.page_content[:80]}")

        print("\nTest passed ✅")