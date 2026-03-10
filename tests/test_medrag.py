"""
Automated Test Suite — MedRAG (ChromaDB edition)
=================================================
Tests cover:
  - Document loading + chunking
  - ChromaDB vector store operations
  - RAG pipeline logic
  - FastAPI endpoints
  - Input validation
  - Error handling edge cases

Run:
    pytest tests/test_medrag.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from langchain.schema import Document


# ══════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=(
                "Dengue fever treatment is primarily supportive. "
                "Patients should rest and drink plenty of fluids. "
                "Paracetamol is recommended for fever — avoid NSAIDs."
            ),
            metadata={"source": "who_dengue.txt", "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Paracetamol adult dose: 500mg–1g every 4–6 hours. "
                "Maximum daily dose is 4g. Overdose can cause liver damage."
            ),
            metadata={"source": "drug_manual.txt", "chunk_index": 1},
        ),
        Document(
            page_content=(
                "Amoxicillin is a penicillin-type antibiotic. "
                "Adult dose: 250–500mg every 8 hours. "
                "Contraindicated in penicillin allergy."
            ),
            metadata={"source": "drug_manual.txt", "chunk_index": 2},
        ),
    ]


@pytest.fixture
def tmp_text_file(tmp_path):
    content = (
        "MALARIA TREATMENT GUIDELINES\n\n"
        "Malaria is caused by Plasmodium parasites transmitted by Anopheles mosquitoes.\n"
        "First-line treatment for uncomplicated P. falciparum: Artemether-Lumefantrine (AL).\n"
        "Severe malaria requires IV Artesunate and hospitalization.\n"
        "Prevention: insecticide-treated nets, indoor spraying, chemoprophylaxis.\n"
    )
    f = tmp_path / "malaria_guidelines.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def vector_store_with_docs(sample_documents, tmp_path):
    """ChromaDB store pre-loaded with sample documents."""
    from vector_store import MedicalVectorStore
    store = MedicalVectorStore(
        persist_directory=str(tmp_path / "test_chroma"),
        collection_name="test_collection",
    )
    store.add_documents(sample_documents)
    return store


# ══════════════════════════════════════════════════════════════════
# 1. DOCUMENT LOADER TESTS
# ══════════════════════════════════════════════════════════════════

class TestDocumentLoader:

    def test_load_text_file(self, tmp_text_file):
        from document_loader import MedicalDocumentLoader
        loader = MedicalDocumentLoader(chunk_size=200, chunk_overlap=20)
        chunks = loader.load_text(str(tmp_text_file))
        assert len(chunks) >= 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunk_metadata_populated(self, tmp_text_file):
        from document_loader import MedicalDocumentLoader
        loader = MedicalDocumentLoader(chunk_size=200, chunk_overlap=20)
        chunks = loader.load_text(str(tmp_text_file))
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["source"] == "malaria_guidelines.txt"

    def test_chunk_size_respected(self, tmp_text_file):
        from document_loader import MedicalDocumentLoader
        max_size = 100
        loader = MedicalDocumentLoader(chunk_size=max_size, chunk_overlap=10)
        chunks = loader.load_text(str(tmp_text_file))
        for chunk in chunks:
            assert len(chunk.page_content) <= max_size * 1.5

    def test_load_from_bytes_txt(self, tmp_text_file):
        from document_loader import MedicalDocumentLoader
        loader = MedicalDocumentLoader()
        content = tmp_text_file.read_bytes()
        chunks = loader.load_from_bytes(content, "test_doc.txt")
        assert len(chunks) >= 1
        assert all(c.metadata["source"] == "test_doc.txt" for c in chunks)

    def test_load_from_bytes_unsupported_type(self):
        from document_loader import MedicalDocumentLoader
        loader = MedicalDocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_from_bytes(b"data", "file.docx")

    def test_load_directory_empty(self, tmp_path):
        from document_loader import MedicalDocumentLoader
        loader = MedicalDocumentLoader()
        chunks = loader.load_directory(str(tmp_path))
        assert chunks == []

    def test_load_directory_with_files(self, tmp_path):
        from document_loader import MedicalDocumentLoader
        (tmp_path / "a.txt").write_text("Dengue treatment: rest and fluids.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Paracetamol dose: 500mg every 6 hours.", encoding="utf-8")
        loader = MedicalDocumentLoader()
        chunks = loader.load_directory(str(tmp_path))
        assert len(chunks) >= 2

    def test_load_nonexistent_directory(self, tmp_path):
        from document_loader import MedicalDocumentLoader
        loader = MedicalDocumentLoader()
        result = loader.load_directory(str(tmp_path / "does_not_exist"))
        assert result == []


# ══════════════════════════════════════════════════════════════════
# 2. CHROMADB VECTOR STORE TESTS
# ══════════════════════════════════════════════════════════════════

class TestVectorStore:

    def test_add_and_search(self, vector_store_with_docs):
        results = vector_store_with_docs.search("dengue treatment", top_k=2)
        assert len(results) <= 2
        assert any("dengue" in r.page_content.lower() for r in results)

    def test_search_returns_documents(self, vector_store_with_docs):
        results = vector_store_with_docs.search("paracetamol dose")
        assert all(isinstance(r, Document) for r in results)

    def test_empty_store_raises_on_search(self, tmp_path):
        from vector_store import MedicalVectorStore
        empty_store = MedicalVectorStore(
            persist_directory=str(tmp_path / "empty_chroma"),
            collection_name="empty_col",
        )
        with pytest.raises(RuntimeError, match="empty"):
            empty_store.search("any query")

    def test_is_empty_flag(self, tmp_path):
        from vector_store import MedicalVectorStore
        store = MedicalVectorStore(
            persist_directory=str(tmp_path / "flag_test"),
            collection_name="flag_col",
        )
        assert store.is_empty is True

    def test_is_not_empty_after_adding(self, vector_store_with_docs):
        assert vector_store_with_docs.is_empty is False

    def test_doc_count(self, vector_store_with_docs):
        assert vector_store_with_docs.doc_count == 3

    def test_clear_resets_store(self, vector_store_with_docs):
        assert not vector_store_with_docs.is_empty
        vector_store_with_docs.clear()
        assert vector_store_with_docs.is_empty

    def test_list_sources(self, vector_store_with_docs):
        sources = vector_store_with_docs.list_sources()
        assert "who_dengue.txt"  in sources
        assert "drug_manual.txt" in sources

    def test_search_by_source_filter(self, vector_store_with_docs):
        """ChromaDB advantage: filter results to a specific source file."""
        results = vector_store_with_docs.search_by_source(
            "antibiotic medicine", "drug_manual.txt", top_k=3
        )
        assert all(r.metadata["source"] == "drug_manual.txt" for r in results)

    def test_delete_by_source(self, vector_store_with_docs):
        deleted = vector_store_with_docs.delete_by_source("who_dengue.txt")
        assert deleted >= 1
        sources = vector_store_with_docs.list_sources()
        assert "who_dengue.txt" not in sources

    def test_chroma_auto_persists(self, sample_documents, tmp_path):
        """ChromaDB should reload data automatically on re-init — no manual save needed."""
        from vector_store import MedicalVectorStore

        persist_dir = str(tmp_path / "persist_test")

        # First instance — add documents
        store1 = MedicalVectorStore(
            persist_directory=persist_dir,
            collection_name="persist_col",
        )
        store1.add_documents(sample_documents)
        assert store1.doc_count == 3

        # Second instance — should auto-load from disk
        store2 = MedicalVectorStore(
            persist_directory=persist_dir,
            collection_name="persist_col",
        )
        assert store2.doc_count == 3, "ChromaDB should auto-persist and reload"
        results = store2.search("dengue", top_k=1)
        assert len(results) == 1

    def test_search_with_scores(self, vector_store_with_docs):
        import numbers
        results = vector_store_with_docs.search_with_scores("fever medication", top_k=2)
        assert all(
            isinstance(doc, Document) and isinstance(score, numbers.Real)
            for doc, score in results
        )

    def test_add_empty_documents_no_crash(self, tmp_path):
        from vector_store import MedicalVectorStore
        store = MedicalVectorStore(
            persist_directory=str(tmp_path / "empty_add"),
            collection_name="empty_add_col",
        )
        store.add_documents([])
        assert store.is_empty

    def test_save_is_noop(self, vector_store_with_docs):
        """save() should not crash — it's a no-op for ChromaDB."""
        vector_store_with_docs.save()  # should not raise


# ══════════════════════════════════════════════════════════════════
# 3. RAG PIPELINE TESTS
# ══════════════════════════════════════════════════════════════════

class TestRAGPipeline:

    def test_answer_returns_response_object(self, vector_store_with_docs):
        from rag_pipeline import RAGPipeline, RAGResponse
        pipeline = RAGPipeline(vector_store=vector_store_with_docs)
        with patch.object(pipeline, "_call_ollama", return_value="Supportive care with fluids."):
            result = pipeline.answer("What is the treatment for dengue?")
        assert isinstance(result, RAGResponse)
        assert result.answer == "Supportive care with fluids."
        assert result.retrieved_chunks > 0

    def test_answer_empty_store_returns_message(self, tmp_path):
        from rag_pipeline import RAGPipeline
        from vector_store import MedicalVectorStore
        empty_store = MedicalVectorStore(
            persist_directory=str(tmp_path / "empty_rag"),
            collection_name="empty_rag_col",
        )
        pipeline = RAGPipeline(vector_store=empty_store)
        result = pipeline.answer("Any question")
        assert "No medical documents" in result.answer or "upload" in result.answer.lower()

    def test_sources_populated(self, vector_store_with_docs):
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(vector_store=vector_store_with_docs)
        with patch.object(pipeline, "_call_ollama", return_value="Answer here."):
            result = pipeline.answer("paracetamol dose")
        assert len(result.sources) > 0
        assert all(hasattr(s, "document") for s in result.sources)

    def test_ingest_bytes_txt(self, tmp_path):
        from rag_pipeline import RAGPipeline
        from vector_store import MedicalVectorStore
        pipeline = RAGPipeline(
            vector_store=MedicalVectorStore(
                persist_directory=str(tmp_path / "ingest_test"),
                collection_name="ingest_col",
            )
        )
        content = b"Malaria is treated with artemisinin-based combination therapy."
        chunks = pipeline.ingest_bytes(content, "test.txt")
        assert chunks >= 1

    def test_ollama_connection_error_handled(self, vector_store_with_docs):
        import requests as req
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(vector_store=vector_store_with_docs)
        with patch("requests.post", side_effect=req.exceptions.ConnectionError):
            result = pipeline.answer("dengue symptoms")
        assert "Ollama" in result.answer or "connect" in result.answer.lower()

    def test_summarize_returns_string(self, vector_store_with_docs):
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(vector_store=vector_store_with_docs)
        with patch.object(pipeline, "_call_ollama", return_value="Summary text here."):
            summary = pipeline.summarize("Long medical text about dengue treatment protocols.")
        assert isinstance(summary, str)
        assert len(summary) > 0


# ══════════════════════════════════════════════════════════════════
# 4. API ENDPOINT TESTS
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def api_client(vector_store_with_docs):
    from fastapi.testclient import TestClient
    import api as api_module

    api_module.vector_store = vector_store_with_docs
    api_module.pipeline     = MagicMock()
    api_module.pipeline.vector_store = vector_store_with_docs
    api_module.pipeline.top_k = 5

    from rag_pipeline import RAGResponse, SourceCitation
    api_module.pipeline.answer.return_value = RAGResponse(
        answer="Supportive care: rest, fluids, paracetamol.",
        sources=[SourceCitation(document="who_dengue.txt", chunk_index=0, snippet="Dengue treatment...")],
        retrieved_chunks=3,
        model_used="llama3",
    )
    api_module.pipeline.summarize.return_value = "Summary of document."
    api_module.pipeline.ingest_bytes.return_value = 5
    api_module._doc_registry.clear()

    return TestClient(api_module.app)


class TestAPIEndpoints:

    def test_root_returns_200(self, api_client):
        r = api_client.get("/")
        assert r.status_code == 200
        assert "MedRAG" in r.json()["service"]

    def test_health_endpoint(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_ready_endpoint(self, api_client):
        r = api_client.get("/ready")
        assert r.status_code == 200

    def test_status_endpoint(self, api_client):
        r = api_client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert "llm_model" in data
        assert "document_count" in data

    def test_upload_txt_file(self, api_client):
        content = b"Dengue fever treatment guidelines: rest, hydration, paracetamol."
        r = api_client.post(
            "/upload",
            files={"file": ("test_guidelines.txt", content, "text/plain")},
        )
        assert r.status_code == 201
        data = r.json()
        assert "doc_id" in data
        assert data["filename"] == "test_guidelines.txt"

    def test_upload_empty_file_returns_400(self, api_client):
        r = api_client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert r.status_code == 400

    def test_upload_unsupported_type_returns_415(self, api_client):
        r = api_client.post(
            "/upload",
            files={"file": ("report.docx", b"fake content", "application/vnd.openxmlformats")},
        )
        assert r.status_code == 415

    def test_list_documents_empty(self, api_client):
        r = api_client.get("/documents")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_documents_after_upload(self, api_client):
        api_client.post(
            "/upload",
            files={"file": ("a.txt", b"Some medical content for testing.", "text/plain")},
        )
        r = api_client.get("/documents")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_delete_nonexistent_returns_404(self, api_client):
        r = api_client.delete("/documents/nonexistent-id")
        assert r.status_code == 404

    def test_delete_document_success(self, api_client):
        upload_r = api_client.post(
            "/upload",
            files={"file": ("delete_me.txt", b"Content to delete from index.", "text/plain")},
        )
        doc_id = upload_r.json()["doc_id"]
        r = api_client.delete(f"/documents/{doc_id}")
        assert r.status_code == 200

    def test_ask_question_success(self, api_client):
        r = api_client.post("/ask", json={"question": "What is the treatment for dengue?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "sources" in data
        assert "retrieved_chunks" in data

    def test_ask_empty_question_returns_422(self, api_client):
        r = api_client.post("/ask", json={"question": "   "})
        assert r.status_code == 422

    def test_ask_too_short_returns_422(self, api_client):
        r = api_client.post("/ask", json={"question": "hi"})
        assert r.status_code == 422

    def test_ask_top_k_out_of_range(self, api_client):
        r = api_client.post("/ask", json={"question": "dengue treatment", "top_k": 99})
        assert r.status_code == 422

    def test_summarize_success(self, api_client):
        long_text = "Dengue fever is a viral illness. " * 10
        r = api_client.post("/summarize", json={"text": long_text})
        assert r.status_code == 200
        assert "summary" in r.json()

    def test_summarize_too_short_returns_422(self, api_client):
        r = api_client.post("/summarize", json={"text": "Too short."})
        assert r.status_code == 422

    def test_request_id_in_headers(self, api_client):
        r = api_client.get("/health")
        assert "x-request-id" in r.headers

    def test_response_time_in_headers(self, api_client):
        r = api_client.get("/health")
        assert "x-response-time" in r.headers

    def test_clear_index(self, api_client):
        r = api_client.delete("/index")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════
# 5. INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════

class TestInputValidation:

    def test_question_strips_whitespace(self):
        from api import QuestionRequest
        q = QuestionRequest(question="  dengue symptoms  ")
        assert q.question == "dengue symptoms"

    def test_question_max_length(self):
        from pydantic import ValidationError
        from api import QuestionRequest
        with pytest.raises(ValidationError):
            QuestionRequest(question="a" * 1001)

    def test_top_k_minimum(self):
        from pydantic import ValidationError
        from api import QuestionRequest
        with pytest.raises(ValidationError):
            QuestionRequest(question="valid question here", top_k=0)

    def test_top_k_maximum(self):
        from pydantic import ValidationError
        from api import QuestionRequest
        with pytest.raises(ValidationError):
            QuestionRequest(question="valid question here", top_k=21)

    def test_summarize_min_length(self):
        from pydantic import ValidationError
        from api import SummarizeRequest
        with pytest.raises(ValidationError):
            SummarizeRequest(text="short")


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short", "--no-header"],
        capture_output=False,
    )
    sys.exit(result.returncode)