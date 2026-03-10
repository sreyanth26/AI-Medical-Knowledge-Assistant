"""
Step 2 & 3: Document Loader and Chunker
Loads PDFs and splits them into chunks for embedding.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, DirectoryLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDocumentLoader:
    """
    Loads medical documents (PDFs, TXT) and splits them into
    overlapping chunks suitable for semantic search.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        Args:
            chunk_size:    Target token count per chunk (500–1000 recommended).
            chunk_overlap: Overlap between consecutive chunks to preserve context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF file and return chunked Documents."""
        logger.info(f"Loading PDF: {file_path}")
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        return self._chunk_and_tag(pages, source=file_path)

    def load_text(self, file_path: str) -> List[Document]:
        """Load a plain-text file and return chunked Documents."""
        logger.info(f"Loading text file: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        return self._chunk_and_tag(docs, source=file_path)

    def load_directory(self, directory: str) -> List[Document]:
        """
        Recursively load all PDFs and .txt files from a directory.
        Returns a flat list of chunked Documents.
        """
        logger.info(f"Loading directory: {directory}")
        all_docs: List[Document] = []

        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory not found: {directory}")
            return all_docs

        # PDFs
        for pdf_file in path.rglob("*.pdf"):
            try:
                chunks = self.load_pdf(str(pdf_file))
                all_docs.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to load {pdf_file}: {e}")

        # Plain text
        for txt_file in path.rglob("*.txt"):
            try:
                chunks = self.load_text(str(txt_file))
                all_docs.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to load {txt_file}: {e}")

        logger.info(f"Loaded {len(all_docs)} chunks from {directory}")
        return all_docs

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Load a document from raw bytes (e.g., uploaded via API).
        Saves a temporary file, processes it, then deletes it.
        """
        import tempfile

        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                chunks = self.load_pdf(tmp_path)
            elif suffix in (".txt", ".md"):
                chunks = self.load_text(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            # Rewrite source metadata to the original filename
            for chunk in chunks:
                chunk.metadata["source"] = filename

            return chunks
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_and_tag(self, docs: List[Document], source: str) -> List[Document]:
        """Split documents and enrich metadata."""
        chunks = self.splitter.split_documents(docs)
        filename = Path(source).name

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "source": filename,
                    "full_path": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )

        logger.info(f"Created {len(chunks)} chunks from '{filename}'")
        return chunks


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    loader = MedicalDocumentLoader(chunk_size=800, chunk_overlap=150)
    sample_dir = "../data/medical_guidelines"

    if Path(sample_dir).exists():
        docs = loader.load_directory(sample_dir)
        print(f"\nTotal chunks loaded: {len(docs)}")
        if docs:
            print("\nSample chunk:")
            print(docs[0].page_content[:300])
            print("\nMetadata:", docs[0].metadata)
    else:
        print(f"Directory '{sample_dir}' not found. Place PDFs there and re-run.")
