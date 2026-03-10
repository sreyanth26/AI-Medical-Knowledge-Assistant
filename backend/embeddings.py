"""
Step 4: Embedding Generator
Converts text chunks into vector embeddings using a local HuggingFace model.
Model: sentence-transformers/all-MiniLM-L6-v2  (free, runs locally)
"""

import logging
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Default embedding model — lightweight, fast, and effective for semantic search
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding model.

    The first call downloads the model weights (~90 MB).
    Subsequent calls load from the local cache.

    Args:
        model_name: Any sentence-transformers compatible model on HuggingFace.

    Returns:
        A LangChain-compatible HuggingFaceEmbeddings instance.
    """
    logger.info(f"Loading embedding model: {model_name}")

    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU is available
        encode_kwargs={
            "normalize_embeddings": True,  # cosine similarity works better with L2-norm
            "batch_size": 32,
        },
    )

    logger.info("Embedding model loaded successfully.")
    return model


def embed_texts(texts: List[str], model_name: str = DEFAULT_MODEL) -> List[List[float]]:
    """
    Convenience function: embed a list of raw strings.

    Args:
        texts:      List of text strings to embed.
        model_name: HuggingFace model identifier.

    Returns:
        List of embedding vectors (each vector is a list of floats).
    """
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    """
    Embed a single user query for similarity search.

    Args:
        query:      The question or search string.
        model_name: HuggingFace model identifier.

    Returns:
        A single embedding vector.
    """
    model = get_embedding_model(model_name)
    return model.embed_query(query)


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "Paracetamol is used to treat fever and mild to moderate pain.",
        "Dengue fever is caused by a virus transmitted by Aedes mosquitoes.",
        "Amoxicillin is a penicillin-type antibiotic used to treat bacterial infections.",
    ]

    print("Generating embeddings for sample medical texts...\n")
    vectors = embed_texts(sample_texts)

    for text, vec in zip(sample_texts, vectors):
        print(f"Text   : {text[:60]}...")
        print(f"Vector : [{vec[0]:.4f}, {vec[1]:.4f}, ... ] (dim={len(vec)})\n")
