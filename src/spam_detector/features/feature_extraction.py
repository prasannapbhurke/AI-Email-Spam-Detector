from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    # Keep TF-IDF deterministic.
    lowercase: bool = False  # preprocessing already lowercases


def create_tfidf_vectorizer(config: Optional[TfidfConfig] = None) -> TfidfVectorizer:
    cfg = config or TfidfConfig()
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        lowercase=cfg.lowercase,
    )


class SpacyEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Optional dense feature extractor using spaCy token vectors.

    Notes:
    - For most meaningful embeddings, prefer a spaCy model with vectors
      (e.g., `en_core_web_md` / `en_core_web_lg`), not only `en_core_web_sm`.
    - This transformer expects *preprocessed* text (lowercased/lemmatized tokens
      are ideal). Input is a sequence of strings.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        vector_size: Optional[int] = None,
        pooling: str = "mean",
        non_negative: bool = True,
        batch_size: int = 64,
        raise_on_missing_vectors: bool = True,
    ):
        if pooling not in {"mean"}:
            raise ValueError("Only pooling='mean' is supported in this implementation.")

        self.spacy_model = spacy_model
        self.vector_size = vector_size
        self.pooling = pooling
        self.non_negative = non_negative
        self.batch_size = batch_size
        self.raise_on_missing_vectors = raise_on_missing_vectors

        self._nlp = None

    def _load_nlp(self) -> None:
        if self._nlp is not None:
            return

        # Disable everything except tokenizer + vocabulary vectors.
        disable = ["ner", "parser", "tagger", "attribute_ruler", "lemmatizer", "textcat"]
        self._nlp = spacy.load(self.spacy_model, disable=disable)

        if self.vector_size is None:
            # Use the current model's vector dimension (may be 0 for models without vectors).
            self.vector_size = int(getattr(self._nlp.vocab, "vectors_length", 0) or 0)

    def fit(self, X: Iterable[str], y=None):
        self._load_nlp()
        if self.raise_on_missing_vectors and int(self.vector_size or 0) <= 0:
            raise ValueError(
                f"spaCy model '{self.spacy_model}' has no vectors (vectors_length=0). "
                "Install a model with vectors (e.g. en_core_web_md / en_core_web_lg) or "
                "train with TF-IDF instead."
            )
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        self._load_nlp()

        texts = list(X)
        dim = int(self.vector_size or 0)

        if dim <= 0:
            # Model has no vectors. Return a stable zero matrix.
            return np.zeros((len(texts), 0), dtype=np.float32)

        out = np.zeros((len(texts), dim), dtype=np.float32)

        for i, doc in enumerate(self._nlp.pipe(texts, batch_size=self.batch_size)):
            vectors: List[np.ndarray] = []
            for token in doc:
                if token.has_vector:
                    vectors.append(token.vector)
            if vectors:
                vec = np.mean(np.stack(vectors, axis=0), axis=0)
            else:
                vec = np.zeros((dim,), dtype=np.float32)

            if self.non_negative:
                vec = np.maximum(vec, 0)

            out[i] = vec.astype(np.float32, copy=False)

        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_nlp"] = None
        return state

