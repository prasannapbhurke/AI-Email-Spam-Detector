from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import nltk
import spacy
from sklearn.base import BaseEstimator, TransformerMixin


_NON_WORD_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def ensure_nltk_stopwords(lang: str = "english", download_if_missing: bool = True) -> List[str]:
    """
    Ensure NLTK stopwords are available and return them.

    In production, prefer downloading during image build or startup (not per request).
    """

    try:
        from nltk.corpus import stopwords

        return stopwords.words(lang)
    except LookupError:
        if not download_if_missing:
            raise RuntimeError(
                "NLTK stopwords not found. Run: python -c \"import nltk; nltk.download('stopwords')\""
            )
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return stopwords.words(lang)


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for deterministic text normalization.
    """

    spacy_model: str = "en_core_web_sm"
    nltk_stopwords_lang: str = "english"
    additional_stopwords: Optional[List[str]] = None
    # Keep this small to avoid slow token filtering.
    min_token_len: int = 2
    # spaCy pipeline disable list for performance.
    spacy_disable: Optional[List[str]] = None
    batch_size: int = 64
    # In production, prefer pre-downloading NLTK data at build/startup.
    download_nltk_stopwords_if_missing: bool = False


class SpaCyPreprocessTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for:
      - lowercasing
      - punctuation removal
      - tokenization (via spaCy)
      - stopword removal (via NLTK)
      - lemmatization (via spaCy)

    Returns a list[str] where each string is a whitespace-joined lemma sequence.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

        # Lazily loaded to keep joblib serialization smaller and to avoid pickling spaCy objects.
        self._nlp = None
        self._stopwords = None

    def _load_nlp(self) -> None:
        if self._nlp is not None:
            return

        disable = self.config.spacy_disable or ["ner", "parser", "textcat"]
        self._nlp = spacy.load(self.config.spacy_model, disable=disable)

        # Some spaCy models have case-sensitive tokenization; ensure we're lowercasing ourselves.
        # (We do explicit lowercasing in _clean_text.)

    def _load_stopwords(self) -> None:
        if self._stopwords is not None:
            return

        download_if_missing = getattr(self.config, "download_nltk_stopwords_if_missing", False)
        stop_words = set(
            ensure_nltk_stopwords(self.config.nltk_stopwords_lang, download_if_missing=download_if_missing)
        )
        if self.config.additional_stopwords:
            stop_words.update(w.lower() for w in self.config.additional_stopwords)
        self._stopwords = stop_words

    def _clean_text(self, text: str) -> str:
        # Lowercasing
        text = text.lower()
        # Punctuation removal (replace with spaces so tokens stay separated)
        text = _NON_WORD_RE.sub(" ", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text

    def fit(self, X: Iterable[str], y=None):
        self._load_stopwords()
        # Do not eagerly load spaCy here; it's heavy. It will load on first transform call.
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        self._load_stopwords()
        self._load_nlp()

        texts = list(X)
        cleaned_texts = [self._clean_text(t if t is not None else "") for t in texts]

        # Use nlp.pipe for batch processing performance.
        results: List[str] = []
        for doc in self._nlp.pipe(cleaned_texts, batch_size=self.config.batch_size):
            lemmas: List[str] = []
            for token in doc:
                if token.is_space:
                    continue
                lemma = (token.lemma_ or "").strip().lower()
                if not lemma:
                    continue
                # Stopword removal
                if lemma in self._stopwords:
                    continue
                # Optional lightweight token length filter
                if len(lemma) < self.config.min_token_len:
                    continue
                lemmas.append(lemma)
            results.append(" ".join(lemmas))

        return results

    def __getstate__(self):
        """
        Make the transformer joblib-serializable by dropping spaCy model objects.
        """

        state = self.__dict__.copy()
        state["_nlp"] = None
        return state

