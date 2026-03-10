"""
Step 6 & 7: RAG Pipeline
Combines retrieval + LLM generation to answer medical questions
from uploaded documents.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain.schema import Document

from document_loader import MedicalDocumentLoader
from vector_store import MedicalVectorStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------

@dataclass
class SourceCitation:
    document: str
    chunk_index: int
    snippet: str


@dataclass
class RAGResponse:
    answer: str
    sources: List[SourceCitation] = field(default_factory=list)
    retrieved_chunks: int = 0
    model_used: str = ""  # not a Pydantic model — dataclass, no namespace conflict


# ------------------------------------------------------------------
# Prompt Templates
# ------------------------------------------------------------------

MEDICAL_QA_PROMPT = """You are a knowledgeable medical information assistant.
Your role is to provide accurate, helpful answers STRICTLY based on the provided medical documents.

IMPORTANT RULES:
1. Answer ONLY using information from the context below.
2. If the answer is not in the context, say: "I could not find relevant information in the uploaded documents."
3. Always recommend consulting a qualified healthcare professional for medical decisions.
4. Be concise, clear, and use plain language.
5. If dosages or treatments are mentioned, include any warnings present in the context.

---
CONTEXT FROM MEDICAL DOCUMENTS:
{context}

---
QUESTION: {question}

ANSWER:"""

SUMMARIZATION_PROMPT = """You are a medical document summarizer.
Provide a clear, structured summary of the following medical document content.
Include: key topics, important medications/treatments mentioned, and any critical warnings.

DOCUMENT CONTENT:
{content}

SUMMARY:"""


# ------------------------------------------------------------------
# Core Pipeline
# ------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline for medical Q&A.

    Flow:
        User Question
          → Embed question
          → FAISS similarity search
          → Build prompt with retrieved context
          → LLM generates answer
          → Return answer + source citations
    """

    def __init__(
        self,
        vector_store: Optional[MedicalVectorStore] = None,
        llm_model: str = "phi3",
        top_k: int = 5,
        ollama_host: str = "http://localhost:11434",
    ):
        self.vector_store = vector_store or MedicalVectorStore()
        self.llm_model = llm_model
        self.top_k = top_k
        self.ollama_host = ollama_host
        self._loader = MedicalDocumentLoader()

    # ------------------------------------------------------------------
    # Document Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> int:
        """
        Load a PDF/text file, chunk it, and add it to the vector store.

        Returns:
            Number of chunks indexed.
        """
        ext = file_path.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            chunks = self._loader.load_pdf(file_path)
        elif ext in ("txt", "md"):
            chunks = self._loader.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        self.vector_store.add_documents(chunks)
        self.vector_store.save()
        return len(chunks)

    def ingest_bytes(self, file_bytes: bytes, filename: str) -> int:
        """
        Ingest a file from raw bytes (API upload).

        Returns:
            Number of chunks indexed.
        """
        chunks = self._loader.load_from_bytes(file_bytes, filename)
        self.vector_store.add_documents(chunks)
        self.vector_store.save()
        return len(chunks)

    def ingest_directory(self, directory: str) -> int:
        """Ingest all documents in a directory."""
        chunks = self._loader.load_directory(directory)
        self.vector_store.add_documents(chunks)
        self.vector_store.save()
        return len(chunks)

    # ------------------------------------------------------------------
    # Query Processing
    # ------------------------------------------------------------------

    def answer(self, question: str) -> RAGResponse:
        """
        Main entry point: answer a medical question using RAG.

        Steps:
          1. Retrieve relevant chunks from FAISS
          2. Build prompt
          3. Call local LLM via Ollama
          4. Return structured response with citations
        """
        if self.vector_store.is_empty:
            return RAGResponse(
                answer=(
                    "No medical documents have been uploaded yet. "
                    "Please upload PDF or text files first."
                )
            )

        # Step 1: Retrieve
        retrieved: List[Document] = self.vector_store.search(question, top_k=self.top_k)

        if not retrieved:
            return RAGResponse(
                answer="I could not find relevant information in the uploaded documents."
            )

        # Step 2: Build context string
        # Limit each chunk to 400 chars to keep total prompt within Ollama context window
        MAX_CHARS_PER_CHUNK = 400
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            src     = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            if len(content) > MAX_CHARS_PER_CHUNK:
                content = content[:MAX_CHARS_PER_CHUNK] + "..."
            context_parts.append(f"[{i}] Source: {src}\n{content}")
        context = "\n\n".join(context_parts)

        logger.info(f"Context built | chunks={len(retrieved)} | total_chars={len(context)}")

        # Step 3: Build prompt
        prompt = MEDICAL_QA_PROMPT.format(context=context, question=question)
        logger.info(f"Prompt built | total_len={len(prompt)} chars")

        # Step 4: Generate answer via Ollama
        answer_text = self._call_ollama(prompt)

        # Step 5: Build citations
        sources = [
            SourceCitation(
                document=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", -1),
                snippet=doc.page_content[:200],
            )
            for doc in retrieved
        ]

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            retrieved_chunks=len(retrieved),
            model_used=self.llm_model,
        )

    def summarize(self, text: str) -> str:
        """Summarize a block of medical text using the LLM."""
        prompt = SUMMARIZATION_PROMPT.format(content=text[:4000])
        return self._call_ollama(prompt)

    # ------------------------------------------------------------------
    # LLM Interface (Ollama)
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        """
        Send a prompt to the locally running Ollama LLM and return the response.

        Requires Ollama to be running:  ollama serve
        Pull the model first:           ollama pull phi3
        """
        try:
            import requests

            # Trim prompt if too long — prevents Ollama 500 context overflow errors
            # Llama3 default context window is 4096 tokens (~3500 chars safe limit)
            MAX_PROMPT_CHARS = 3500
            if len(prompt) > MAX_PROMPT_CHARS:
                logger.warning(
                    f"Prompt too long ({len(prompt)} chars) — trimming to {MAX_PROMPT_CHARS}"
                )
                prompt = prompt[:MAX_PROMPT_CHARS] + "\n...[context trimmed for length]"

            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,   # low temp = more factual answers
                    "num_predict": 512,   # max tokens in response
                    "num_ctx": 4096,      # explicitly set context window size
                },
            }

            logger.info(f"Calling Ollama | model={self.llm_model} | prompt_len={len(prompt)}")

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "No response from model.")
            logger.info(f"Ollama response received | answer_len={len(answer)}")
            return answer

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            return (
                "⚠️  Could not connect to the local LLM (Ollama). "
                "Please run: ollama serve\n"
                "Then pull a model: ollama pull phi3"
            )
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after 120s.")
            return (
                "⚠️  The LLM took too long to respond (timeout 120s). "
                "Try reducing the Top-K slider to 2 or 3, or use a smaller model like phi3."
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            logger.error(f"Ollama HTTP error: {status} — {e.response.text[:300]}")
            return (
                f"⚠️  Ollama returned error {status}. "
                f"This usually means the model name is wrong or the prompt is too long. "
                f"Check that model \"{self.llm_model}\" is installed: run `ollama list`"
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"⚠️  LLM error: {str(e)}"

# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    from langchain.schema import Document

    # Use a temp directory for ChromaDB during smoke-test
    with tempfile.TemporaryDirectory() as tmp:
        store = MedicalVectorStore(persist_directory=tmp, collection_name="smoke_test")
        store.add_documents([
            Document(
                page_content=(
                    "Dengue fever treatment is primarily supportive. Patients should rest, "
                    "maintain adequate fluid intake, and take paracetamol for fever. "
                    "Avoid aspirin and NSAIDs as they may increase bleeding risk."
                ),
                metadata={"source": "who_dengue_guidelines.txt", "chunk_index": 0},
            ),
            Document(
                page_content=(
                    "Paracetamol dosage for adults: 500mg-1g every 4-6 hours as needed. "
                    "Maximum daily dose: 4g. Overdose can cause serious liver damage."
                ),
                metadata={"source": "drug_database.txt", "chunk_index": 1},
            ),
        ])

        pipeline = RAGPipeline(vector_store=store)
        result   = pipeline.answer("What is the treatment for dengue fever?")

        print("\n=== RAG RESPONSE ===")
        print(result.answer)
        print("\n=== SOURCES ===")
        for src in result.sources:
            print(f"  📄 {src.document}  (chunk #{src.chunk_index})")
            print(f"     {src.snippet[:100]}...")