import hashlib
import logging
import os
import socket
import uuid
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import chromadb

from vectorcode.cli_utils import Config, expand_path

logger = logging.getLogger(name=__name__)


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class StandaloneOllamaEmbeddingFunction:
    """Standalone Ollama embedding function that doesn't require ChromaDB."""

    def __init__(self, url: str, model_name: str, **kwargs: Any):
        self.url = url
        self.model_name = model_name

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import requests

        embeddings = []
        for text in texts:
            response = requests.post(
                self.url,
                json={"model": self.model_name, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings


class StandaloneSentenceTransformerEmbeddingFunction:
    """Standalone Sentence Transformer embedding function that doesn't require ChromaDB."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs: Any):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


# Mapping of embedding function names to standalone implementations
_STANDALONE_EMBEDDING_FUNCTIONS: dict[str, type] = {
    "OllamaEmbeddingFunction": StandaloneOllamaEmbeddingFunction,
    "SentenceTransformerEmbeddingFunction": StandaloneSentenceTransformerEmbeddingFunction,
}


def hash_str(string: str) -> str:
    """Return the sha-256 hash of a string."""
    return hashlib.sha256(string.encode()).hexdigest()


def hash_file(path: str) -> str:
    """return the sha-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(8192)
            if chunk:
                hasher.update(chunk)
            else:
                break
    return hasher.hexdigest()


def get_uuid() -> str:
    return uuid.uuid4().hex


def get_collection_id(full_path: str) -> str:
    full_path = str(expand_path(full_path, absolute=True))
    hasher = hashlib.sha256()
    plain_collection_name = f"{os.environ.get('USER', os.environ.get('USERNAME', 'DEFAULT_USER'))}@{socket.gethostname()}:{full_path}"
    hasher.update(plain_collection_name.encode())
    collection_id = hasher.hexdigest()[:63]
    logger.debug(
        f"Hashing {plain_collection_name} as the collection name for {full_path}."
    )
    return collection_id


@cache
def get_embedding_function(
    configs: Config,
) -> EmbeddingFunction:  # pragma: nocover
    """
    Get an embedding function based on the configuration.

    First tries standalone implementations to avoid ChromaDB dependency,
    then falls back to ChromaDB's embedding functions if needed.
    """
    ef_name = configs.embedding_function
    ef_params = configs.embedding_params

    # Try standalone implementation first (avoids ChromaDB dependency)
    if ef_name in _STANDALONE_EMBEDDING_FUNCTIONS:
        try:
            ef = _STANDALONE_EMBEDDING_FUNCTIONS[ef_name](**ef_params)
            logger.debug(f"Using standalone {ef_name}")
            return ef
        except Exception as e:
            logger.warning(
                f"Failed to create standalone {ef_name}: {e}. Trying ChromaDB fallback."
            )

    # Fall back to ChromaDB's embedding functions
    try:
        from chromadb.utils import embedding_functions

        ef = getattr(embedding_functions, ef_name)(**ef_params)
        if ef is None:  # pragma: nocover
            raise AttributeError()
        return ef
    except ImportError:
        # ChromaDB not available or incompatible, use standalone fallback
        logger.warning(
            f"ChromaDB not available. Using standalone SentenceTransformer."
        )
        return StandaloneSentenceTransformerEmbeddingFunction()
    except AttributeError:
        logger.warning(
            f"Failed to use {ef_name}. Falling back to Sentence Transformer.",
        )
        return StandaloneSentenceTransformerEmbeddingFunction()
    except Exception as e:
        e.add_note(
            "\nFor errors caused by missing dependency, consult the documentation of pipx (or whatever package manager that you installed VectorCode with) for instructions to inject libraries into the virtual environment."
        )
        logger.error(
            f"Failed to use {ef_name} with following error.",
        )
        raise
