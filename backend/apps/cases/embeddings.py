"""Text embedding for the Case KB and for RAG queries against it.

Loaded lazily and cached: the model is only pulled into memory the first time
something actually needs an embedding (management command, ingestion script,
API request), not at Django boot or during migrations.
"""
from functools import lru_cache

from django.conf import settings


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()
