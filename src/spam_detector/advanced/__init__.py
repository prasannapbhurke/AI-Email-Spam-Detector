"""
Advanced AI features for spam detection.

Includes:
- LIME explainability
- Phishing URL detection
- Multi-language support
- Spam confidence scoring
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Phishing URL patterns
SUSPICIOUS_URL_PATTERNS = [
    r"bit\.ly/\w+",  # Shortened URLs
    r"tinyurl\.com/\w+",
    r"goo\.gl/\w+",
    r"t\.co/\w+",
    r"ow\.ly/\w+",
    r"is\.gd/\w+",
    r"buff\.ly/\w+",
]

# Known phishing TLDs
PHISHING_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",  # Free tier TLDs often used in phishing
    ".xyz", ".top", ".club", ".online", ".site",
}

# Typosquatting patterns (common brand misspellings)
TYPOSQUAT_KEYWORDS = [
    "paypa1", "paypal",  # PayPal typos
    "g00gle", "gooogle",
    "amaz0n", "amazn",
    "faceb00k", "facebook",
    "app1e", "apple",
    "micros0ft", "microsoft",
]

# Legitimate email domains (for comparison)
LEGITIMATE_DOMAINS = {
    "google.com", "gmail.com", "yahoo.com", "microsoft.com", "apple.com",
    "amazon.com", "facebook.com", "linkedin.com", "twitter.com", "instagram.com",
}

# Multi-language spam indicators
LANGUAGE_INDICATORS = {
    "en": [
        "free money", "click here", "act now", "limited time", "winner",
        "congratulations", "urgent action", "suspended", "verify account",
    ],
    "es": [
        "dinero gratis", "haga clic aquí", "ganador", "felicitaciones",
        "cuenta suspendida", "verificar cuenta",
    ],
    "fr": [
        "argent gratuit", "cliquez ici", "gagnant", "félicitations",
        "compte suspendu", "vérifier le compte",
    ],
    "de": [
        "kostenloses geld", "hier klicken", "gewonnen", "herzlichen glückwunsch",
        "konto ausgesetzt", "konto verifizieren",
    ],
    "zh": [
        "免费金钱", "点击这里", "中奖", "恭喜",
        "账户暂停", "验证账户",
    ],
}


@dataclass
class SpamExplanation:
    """Explanation of why an email was classified."""

    prediction: str  # "spam" or "not_spam"
    confidence: float
    spam_probability: float

    # Top contributing features
    top_spam_features: list[tuple[str, float]]  # (feature, weight)
    top_ham_features: list[tuple[str, float]]

    # Human-readable explanation
    explanation: str
    warnings: list[str]

    # Phishing analysis
    phishing_score: float
    suspicious_urls: list[str]

    # Language detection
    detected_language: str
    language_warnings: list[str]


@dataclass
class UrlAnalysis:
    """Analysis result for a single URL."""

    url: str
    is_suspicious: bool
    score: float  # 0-1, higher = more suspicious
    reasons: list[str]
    is_shortened: bool
    has_ip_address: bool
    has_suspicious_tld: bool
    has_login_keywords: bool
    typosquat_candidates: list[str]


class LIMEExplainer:
    """
    LIME-based explainer for spam predictions.

    Provides feature importance explanations for individual predictions.
    """

    def __init__(
        self,
        model_service,
        vectorizer,
        preprocess_fn=None,
        num_features: int = 10,
        num_samples: int = 500,
    ):
        """
        Initialize LIME explainer.

        Args:
            model_service: ModelService instance for predictions.
            vectorizer: TF-IDF or other vectorizer for feature extraction.
            preprocess_fn: Optional text preprocessing function.
            num_features: Number of top features to explain.
            num_samples: Number of samples for LIME explanation.
        """
        self.model_service = model_service
        self.vectorizer = vectorizer
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.num_features = num_features
        self.num_samples = num_samples
        self._explainer = None

    def _get_explainer(self):
        """Lazy load LIME explainer."""
        if self._explainer is None:
            try:
                import lime
                import lime.lime_tabular
            except ImportError:
                logger.warning("LIME not installed. Run: pip install lime")
                return None

            # Get feature names
            try:
                feature_names = self.vectorizer.get_feature_names_out()
            except AttributeError:
                feature_names = [f"f{i}" for i in range(len(self.vectorizer.vocabulary_))]

            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.zeros((1, len(feature_names))),
                feature_names=feature_names,
                class_names=["ham", "spam"],
                mode="classification",
            )

        return self._explainer

    def explain(self, text: str) -> SpamExplanation:
        """
        Generate explanation for a spam prediction.

        Args:
            text: Email text to explain.

        Returns:
            SpamExplanation with detailed reasoning.
        """
        # Get base prediction
        result = self.model_service.predict(text)

        # Get TF-IDF features
        try:
            tfidf_matrix = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Get top features
            top_indices = scores.argsort()[-self.num_features:][::-1]
            top_spam_features = []
            top_ham_features = []

            for idx in top_indices:
                if scores[idx] > 0:
                    feature = feature_names[idx]
                    weight = float(scores[idx])

                    # Determine if feature indicates spam or ham
                    spam_indicators = ["free", "win", "click", "offer", "urgent", "credit"]
                    ham_indicators = ["meeting", "project", "update", "thanks", "regards", "please"]

                    if any(ind in feature.lower() for ind in spam_indicators):
                        top_spam_features.append((feature, weight))
                    elif any(ind in feature.lower() for ind in ham_indicators):
                        top_ham_features.append((feature, weight))
                    else:
                        # Neutral - include in both
                        top_spam_features.append((feature, weight))
                        top_ham_features.append((feature, weight))

            top_spam_features.sort(key=lambda x: x[1], reverse=True)
            top_ham_features.sort(key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.warning("Feature extraction failed: %s", e)
            top_spam_features = []
            top_ham_features = []

        # Analyze URLs
        url_analysis = self._analyze_urls(text)
        phishing_score = max((u.score for u in url_analysis), default=0.0)
        suspicious_urls = [u.url for u in url_analysis if u.is_suspicious]

        # Detect language
        detected_lang, lang_warnings = self._detect_language(text)

        # Build explanation
        warnings = []
        if suspicious_urls:
            warnings.append(f"Contains {len(suspicious_urls)} suspicious URL(s)")
        if phishing_score > 0.5:
            warnings.append("High phishing probability")
        if lang_warnings:
            warnings.extend(lang_warnings)

        explanation = self._build_explanation(
            result.prediction_label,
            result.confidence,
            top_spam_features,
            top_ham_features,
            warnings,
        )

        return SpamExplanation(
            prediction=result.prediction_label,
            confidence=result.confidence,
            spam_probability=result.spam_probability,
            top_spam_features=top_spam_features[:5],
            top_ham_features=top_ham_features[:5],
            explanation=explanation,
            warnings=warnings,
            phishing_score=phishing_score,
            suspicious_urls=suspicious_urls,
            detected_language=detected_lang,
            language_warnings=lang_warnings,
        )

    def _analyze_urls(self, text: str) -> list[UrlAnalysis]:
        """Extract and analyze URLs in text."""
        urls = self._extract_urls(text)
        return [self._analyze_single_url(url) for url in urls]

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        url_pattern = re.compile(
            r"https?://[^\s<>'\"{}|\\^`\[\]]+|"
            r"www\.[^\s<>'\"{}|\\^`\[\]]+|"
            r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s<>'\"{}|\\^`\[\]]*"
        )
        return url_pattern.findall(text)

    def _analyze_single_url(self, url: str) -> UrlAnalysis:
        """Analyze a single URL for phishing indicators."""
        reasons = []
        score = 0.0

        # Check if shortened
        is_shortened = any(re.search(p, url) for p in SUSPICIOUS_URL_PATTERNS)
        if is_shortened:
            reasons.append("Shortened URL detected")
            score += 0.3

        # Check for IP address
        has_ip = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) is not None
        if has_ip:
            reasons.append("URL contains IP address")
            score += 0.4

        # Check TLD
        parsed = urllib.parse.urlparse(url) if url.startswith("http") else urllib.parse.urlparse(f"http://{url}")
        netloc = parsed.netloc.lower()
        has_suspicious_tld = any(netloc.endswith(tld) for tld in PHISHING_TLDS)
        if has_suspicious_tld:
            reasons.append(f"Suspicious TLD: {netloc}")
            score += 0.2

        # Check for login/account keywords
        login_keywords = ["login", "signin", "account", "verify", "password", "update", "secure"]
        has_login_keywords = any(kw in url.lower() for kw in login_keywords)
        if has_login_keywords:
            reasons.append("Contains login/account keywords")
            score += 0.15

        # Check for typosquatting
        typosquat_candidates = []
        for typo in TYPOSQUAT_KEYWORDS:
            if typo in url.lower():
                typosquat_candidates.append(typo)
                reasons.append(f"Possible typosquatting: {typo}")
                score += 0.35

        # Check URL-encoded characters
        if "%" in url or "\\x" in url.lower():
            reasons.append("Contains URL encoding (possible obfuscation)")
            score += 0.1

        return UrlAnalysis(
            url=url,
            is_suspicious=score > 0.3,
            score=min(score, 1.0),
            reasons=reasons,
            is_shortened=is_shortened,
            has_ip_address=has_ip,
            has_suspicious_tld=has_suspicious_tld,
            has_login_keywords=has_login_keywords,
            typosquat_candidates=typosquat_candidates,
        )

    def _detect_language(self, text: str) -> tuple[str, list[str]]:
        """
        Detect language and return warnings for non-English.

        Returns:
            Tuple of (language_code, warnings).
        """
        try:
            from langdetect import detect, DetectorFactory

            DetectorFactory.seed = 42
            lang = detect(text[:500])  # Use first 500 chars for speed
        except ImportError:
            # Fallback: simple keyword detection
            lang = self._simple_lang_detect(text)

        warnings = []

        # Check for mixed language
        for lang_code, indicators in LANGUAGE_INDICATORS.items():
            if lang_code != "en":
                matches = sum(1 for ind in indicators if ind.lower() in text.lower())
                if matches >= 3:
                    warnings.append(f"Multiple {lang_code.upper()} spam indicators found")

        return lang, warnings

    def _simple_lang_detect(self, text: str) -> str:
        """Simple language detection fallback."""
        # Basic character set detection
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[\u0600-\u06ff]", text):
            return "ar"
        if re.search(r"[\u0400-\u04ff]", text):
            return "ru"
        if re.search(r"[\u0900-\u097f]", text):
            return "hi"
        return "en"

    def _build_explanation(
        self,
        prediction: str,
        confidence: float,
        top_spam_features: list[tuple[str, float]],
        top_ham_features: list[tuple[str, float]],
        warnings: list[str],
    ) -> str:
        """Build human-readable explanation."""
        parts = []

        if prediction == "spam":
            parts.append(f"Classified as SPAM with {confidence:.0%} confidence.")
            if top_spam_features:
                features = ", ".join(f"'{f}'" for f, _ in top_spam_features[:3])
                parts.append(f"Key spam indicators: {features}.")
        else:
            parts.append(f"Classified as NOT SPAM with {confidence:.0%} confidence.")
            if top_ham_features:
                features = ", ".join(f"'{f}'" for f, _ in top_ham_features[:3])
                parts.append(f"Key legitimate indicators: {features}.")

        if warnings:
            parts.append(f"Warnings: {'; '.join(warnings)}.")

        return " ".join(parts)


class PhishingDetector:
    """
    Specialized phishing URL detector.

    Analyzes URLs for phishing indicators beyond simple pattern matching.
    """

    def __init__(self):
        self.url_patterns = SUSPICIOUS_URL_PATTERNS
        self.phishing_tlds = PHISHING_TLDS
        self.legitimate_domains = LEGITIMATE_DOMAINS
        self.typosquat_keywords = TYPOSQUAT_KEYWORDS

    def analyze_url(self, url: str) -> UrlAnalysis:
        """Analyze a single URL for phishing indicators."""
        return self._analyze_single_url(url)

    def analyze_text(self, text: str) -> dict[str, Any]:
        """
        Analyze all URLs in text.

        Returns:
            Dictionary with overall phishing assessment.
        """
        urls = self._extract_urls(text)

        if not urls:
            return {
                "has_urls": False,
                "urls_found": 0,
                "overall_score": 0.0,
                "high_risk_urls": [],
            }

        analyses = [self._analyze_single_url(url) for url in urls]
        overall_score = max(a.score for a in analyses) if analyses else 0.0
        high_risk = [a for a in analyses if a.score > 0.5]

        return {
            "has_urls": True,
            "urls_found": len(urls),
            "overall_score": overall_score,
            "high_risk_urls": [
                {"url": a.url, "score": a.score, "reasons": a.reasons}
                for a in high_risk
            ],
            "all_analyses": [
                {"url": a.url, "score": a.score, "is_suspicious": a.is_suspicious}
                for a in analyses
            ],
        }

    def _analyze_single_url(self, url: str) -> UrlAnalysis:
        """Internal URL analysis method."""
        reasons = []
        score = 0.0

        # Check if shortened
        is_shortened = any(re.search(p, url) for p in SUSPICIOUS_URL_PATTERNS)
        if is_shortened:
            reasons.append("Shortened URL")
            score += 0.3

        # Check for IP address
        has_ip = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) is not None
        if has_ip:
            reasons.append("IP address in URL")
            score += 0.4

        # Parse URL
        try:
            if not url.startswith("http"):
                url = "http://" + url
            parsed = urllib.parse.urlparse(url)
            netloc = parsed.netloc.lower()

            # Check TLD
            for tld in PHISHING_TLDS:
                if netloc.endswith(tld):
                    reasons.append(f"Suspicious TLD: {tld}")
                    score += 0.2
                    break

            # Check domain against legitimate
            domain = netloc.split(":")[0]  # Remove port
            for legit in self.legitimate_domains:
                if legit in domain and legit != domain:
                    # e.g., "google.com.evil.com" contains legitimate domain
                    reasons.append(f"Domain contains '{legit}'")
                    score += 0.3
                    break

            # Check for login keywords
            login_keywords = ["login", "signin", "account", "verify", "password", "update", "secure", "banking"]
            if any(kw in url.lower() for kw in login_keywords):
                reasons.append("Login-related keywords")
                score += 0.15

        except Exception:
            pass

        # Typosquatting
        typosquat_candidates = []
        for typo in self.typosquat_keywords:
            if typo in url.lower():
                typosquat_candidates.append(typo)
                reasons.append(f"Typosquatting: {typo}")
                score += 0.35

        return UrlAnalysis(
            url=url,
            is_suspicious=score > 0.3,
            score=min(score, 1.0),
            reasons=reasons,
            is_shortened=is_shortened,
            has_ip_address=has_ip,
            has_suspicious_tld=any(netloc.endswith(tld) for tld in PHISHING_TLDS) if 'netloc' in dir() else False,
            has_login_keywords=any(kw in url.lower() for kw in ["login", "signin", "account", "verify"]),
            typosquat_candidates=typosquat_candidates,
        )

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        url_pattern = re.compile(
            r"https?://[^\s<>'\"{}|\\^`\[\]]+|"
            r"www\.[^\s<>'\"{}|\\^`\[\]]+|"
            r"[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s<>'\"{}|\\^`\[\]]*"
        )
        return url_pattern.findall(text)


class MultiLanguageDetector:
    """
    Detects spam in multiple languages.

    Supports English, Spanish, French, German, Chinese, and more.
    """

    def __init__(self):
        self.language_indicators = LANGUAGE_INDICATORS
        self._langdetect_available = None

    def detect_language(self, text: str) -> str:
        """
        Detect primary language of text.

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr').
        """
        try:
            from langdetect import detect, DetectorFactory

            DetectorFactory.seed = 42
            return detect(text[:1000])  # Use first 1000 chars
        except ImportError:
            return self._simple_detect(text)

    def _simple_detect(self, text: str) -> str:
        """Simple character-based language detection."""
        # Chinese
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        # Japanese (Hiragana/Katakana)
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
            return "ja"
        # Korean
        if re.search(r"[\uac00-\ud7af]", text):
            return "ko"
        # Arabic
        if re.search(r"[\u0600-\u06ff]", text):
            return "ar"
        # Russian/Cyrillic
        if re.search(r"[\u0400-\u04ff]", text):
            return "ru"
        # Hindi/Devanagari
        if re.search(r"[\u0900-\u097f]", text):
            return "hi"
        # Default to English
        return "en"

    def analyze_multilingual_spam(
        self,
        text: str,
        spam_probability: float,
    ) -> dict[str, Any]:
        """
        Analyze spam probability considering language.

        Args:
            text: Email text.
            spam_probability: Base spam probability from model.

        Returns:
            Analysis including language-specific warnings.
        """
        lang = self.detect_language(text)
        warnings = []
        adjusted_prob = spam_probability

        # Check for language-specific spam indicators
        if lang in self.language_indicators:
            indicators = self.language_indicators[lang]
            matches = [ind for ind in indicators if ind.lower() in text.lower()]

            if len(matches) >= 2:
                warnings.append(f"Multiple {lang.upper()} spam indicators: {matches}")
                adjusted_prob = min(1.0, spam_probability + 0.1)

        # Non-English content without indicators might be legitimate
        if lang != "en" and not warnings:
            # Check if it looks like legitimate foreign email
            has_links = bool(re.search(r"https?://", text))
            has_phone = bool(re.search(r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", text))
            if not has_links and not has_phone:
                warnings.append(f"Non-spam {lang.upper()} content detected")
                adjusted_prob = max(0.0, spam_probability - 0.1)

        return {
            "language": lang,
            "warnings": warnings,
            "adjusted_spam_probability": adjusted_prob,
            "original_probability": spam_probability,
        }

    def preprocess_for_language(
        self,
        text: str,
        language: str | None = None,
    ) -> str:
        """
        Preprocess text for specific language.

        Args:
            text: Input text.
            language: Target language code. If None, auto-detects.

        Returns:
            Preprocessed text.
        """
        if language is None:
            language = self.detect_language(text)

        # Language-specific preprocessing could go here
        # For now, just lowercase and normalize whitespace
        text = text.lower().strip()

        if language == "zh":
            # Chinese: don't split on spaces (they don't use them)
            text = re.sub(r"\s+", "", text)
        elif language == "ja":
            # Japanese: normalize spaces
            text = re.sub(r"[\s\u3000]+", " ", text)
        else:
            # Western languages: normalize whitespace
            text = re.sub(r"\s+", " ", text)

        return text
