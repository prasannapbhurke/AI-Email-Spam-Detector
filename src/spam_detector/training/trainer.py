from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from spam_detector.features.feature_extraction import (
    SpacyEmbeddingVectorizer,
    TfidfConfig,
    create_tfidf_vectorizer,
)
from spam_detector.models.model_factory import ModelFactoryConfig, ModelName, get_model
from spam_detector.nlp.preprocessing import PreprocessConfig, SpaCyPreprocessTransformer
from spam_detector.training.evaluation import classification_metrics, find_best_f1_threshold
from spam_detector.training.model_io import ModelMetadata, SpamDetectorModel


FeatureType = Literal["tfidf", "embeddings"]


@dataclass(frozen=True)
class TrainingConfig:
    feature_type: FeatureType = "tfidf"
    use_random_state: int = 42

    # Data split: train/val/test
    test_size: float = 0.2
    val_size_within_train: float = 0.2

    # Threshold tuning grid
    threshold_grid_points: int = 101

    # RandomForest tuning (only used when feature_type='tfidf')
    random_forest_svd_components: int = 200


def _get_spam_class_index(pipeline) -> int:
    # Most pipelines store classifier in pipeline.named_steps['clf'].
    clf = getattr(pipeline, "named_steps", {}).get("clf", None)
    classes = getattr(clf, "classes_", None)
    if classes is None:
        classes = getattr(pipeline, "classes_", None)
    if classes is None:
        raise RuntimeError("Could not find classifier classes_ to map predict_proba columns.")
    classes_arr = np.asarray(classes)
    matches = np.where(classes_arr == 1)[0]
    if matches.size == 0:
        raise RuntimeError("Classifier does not include spam class label '1' in classes_.")
    return int(matches[0])


def _build_pipeline(
    *,
    feature_type: FeatureType,
    preprocess_cfg: PreprocessConfig,
    tfidf_cfg: TfidfConfig,
    embeddings_spacy_model: str,
    model_name: ModelName,
    model_factory_cfg: Optional[ModelFactoryConfig],
    random_forest_svd_components: int,
) -> Pipeline:
    preprocess = SpaCyPreprocessTransformer(config=preprocess_cfg)
    clf = get_model(model_name, config=model_factory_cfg)

    if feature_type == "tfidf":
        tfidf = create_tfidf_vectorizer(tfidf_cfg)
        if model_name == "random_forest":
            # RandomForest expects dense features; convert sparse TF-IDF via SVD.
            svd = TruncatedSVD(n_components=random_forest_svd_components, random_state=model_factory_cfg.random_state if model_factory_cfg else 42)
            return Pipeline([("preprocess", preprocess), ("tfidf", tfidf), ("svd", svd), ("clf", clf)])
        return Pipeline([("preprocess", preprocess), ("tfidf", tfidf), ("clf", clf)])

    if feature_type == "embeddings":
        embeddings = SpacyEmbeddingVectorizer(
            spacy_model=embeddings_spacy_model,
            non_negative=True,
        )
        # NB assumes non-negative features; embedding vectorizer is configured non-negative by default.
        return Pipeline([("preprocess", preprocess), ("embeddings", embeddings), ("clf", clf)])

    raise ValueError(f"Unknown feature_type: {feature_type}")


@dataclass(frozen=True)
class ModelTrainingResult:
    model_name: ModelName
    pipeline: Pipeline
    threshold: float
    metrics: Dict[str, float]


def train_and_select_best_model(
    X_texts: Sequence[str],
    y: np.ndarray,
    *,
    training_cfg: Optional[TrainingConfig] = None,
    preprocess_cfg: Optional[PreprocessConfig] = None,
    tfidf_cfg: Optional[TfidfConfig] = None,
    embeddings_spacy_model: str = "en_core_web_md",
    model_factory_cfg: Optional[ModelFactoryConfig] = None,
    candidate_models: Optional[List[ModelName]] = None,
) -> tuple[SpamDetectorModel, ModelMetadata, Dict[str, ModelTrainingResult]]:
    cfg = training_cfg or TrainingConfig()
    preprocess_cfg = preprocess_cfg or PreprocessConfig()
    tfidf_cfg = tfidf_cfg or TfidfConfig()
    model_factory_cfg = model_factory_cfg or ModelFactoryConfig(random_state=cfg.use_random_state)

    if candidate_models is None:
        candidate_models = ["naive_bayes", "logistic_regression", "random_forest"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        list(X_texts),
        y,
        test_size=cfg.test_size,
        random_state=cfg.use_random_state,
        stratify=y,
    )

    # Further split trainval into train and validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=cfg.val_size_within_train,
        random_state=cfg.use_random_state,
        stratify=y_trainval,
    )

    results: Dict[str, ModelTrainingResult] = {}

    for model_name in candidate_models:
        pipeline = _build_pipeline(
            feature_type=cfg.feature_type,
            preprocess_cfg=preprocess_cfg,
            tfidf_cfg=tfidf_cfg,
            embeddings_spacy_model=embeddings_spacy_model,
            model_name=model_name,
            model_factory_cfg=model_factory_cfg,
            random_forest_svd_components=cfg.random_forest_svd_components,
        )

        # Fit on training split.
        pipeline.fit(X_train, y_train)

        # Tune threshold on validation split.
        spam_idx = _get_spam_class_index(pipeline)
        val_proba_all = pipeline.predict_proba(X_val)
        y_val_proba_spam = val_proba_all[:, spam_idx]

        threshold_res = find_best_f1_threshold(
            y_val,
            y_val_proba_spam,
            thresholds=cfg.threshold_grid_points,
        )

        # Evaluate on test split using tuned threshold.
        test_proba_all = pipeline.predict_proba(X_test)
        y_test_proba_spam = test_proba_all[:, spam_idx]
        y_test_pred = (y_test_proba_spam >= threshold_res.threshold).astype(int)

        metrics = classification_metrics(y_test, y_test_pred)

        results[model_name] = ModelTrainingResult(
            model_name=model_name,
            pipeline=pipeline,
            threshold=threshold_res.threshold,
            metrics=metrics,
        )

    # Select best by F1-score (spam=1).
    best_model_name = max(results.keys(), key=lambda k: (results[k].metrics["f1"], results[k].metrics["recall"]))
    best = results[best_model_name]

    # Refit the best pipeline on train+val (keep threshold tuned on val).
    best_refit = _build_pipeline(
        feature_type=cfg.feature_type,
        preprocess_cfg=preprocess_cfg,
        tfidf_cfg=tfidf_cfg,
        embeddings_spacy_model=embeddings_spacy_model,
        model_name=best_model_name,
        model_factory_cfg=model_factory_cfg,
        random_forest_svd_components=cfg.random_forest_svd_components,
    )
    best_refit.fit(X_trainval, y_trainval)

    # Convert to wrapper for easy thresholded prediction.
    detector = SpamDetectorModel(pipeline=best_refit, threshold=best.threshold, spam_label=1)

    metadata = ModelMetadata(
        model_name=best_model_name,
        vectorizer_type=cfg.feature_type,
        preprocess_spacy_model=preprocess_cfg.spacy_model,
        embeddings_spacy_model=embeddings_spacy_model if cfg.feature_type == "embeddings" else None,
        threshold=best.threshold,
        created_at_unix=float(time.time()),
        metrics=best.metrics,
    )

    return detector, metadata, results

