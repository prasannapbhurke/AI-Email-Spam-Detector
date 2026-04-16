from __future__ import annotations

import argparse
import sys
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spam_detector.features.feature_extraction import TfidfConfig
from spam_detector.nlp.preprocessing import PreprocessConfig, ensure_nltk_stopwords
from spam_detector.training.dataset import DatasetConfig, load_dataset_from_csv, load_sample_dataset
from spam_detector.training.model_io import save_model
from spam_detector.training.trainer import TrainingConfig, train_and_select_best_model
from spam_detector.training.model_io import load_model as load_saved_model


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and persist an email spam detector.")

    p.add_argument("--data", type=str, default="", help="Path to CSV dataset. Must contain text + label columns.")
    p.add_argument("--text-col", type=str, default="text")
    p.add_argument("--label-col", type=str, default="label")

    p.add_argument("--output", type=str, default="models/spam_detector.joblib")

    p.add_argument(
        "--feature-type",
        type=str,
        choices=["tfidf", "embeddings"],
        default="tfidf",
        help="Feature extraction strategy.",
    )
    p.add_argument("--preprocess-spacy-model", type=str, default="en_core_web_sm")
    p.add_argument("--embeddings-spacy-model", type=str, default="en_core_web_md")

    p.add_argument("--tfidf-max-features", type=int, default=50000)
    p.add_argument("--tfidf-ngram-min", type=int, default=1)
    p.add_argument("--tfidf-ngram-max", type=int, default=2)
    p.add_argument("--tfidf-min-df", type=int, default=2)
    p.add_argument("--tfidf-max-df", type=float, default=0.95)

    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-within-train", type=float, default=0.2)
    p.add_argument("--threshold-grid-points", type=int, default=101)

    p.add_argument("--random-forest-svd-components", type=int, default=200)
    p.add_argument("--random-state", type=int, default=42)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Ensure required NLTK resources are present early.
    ensure_nltk_stopwords(download_if_missing=True)

    # Sanity-check spaCy model availability for lemmatization.
    # (The actual training pipeline will lazy-load too.)
    try:
        spacy.load(args.preprocess_spacy_model, disable=["ner", "parser", "textcat", "tagger"])
    except OSError as e:
        raise RuntimeError(
            f"Could not load spaCy preprocess model: '{args.preprocess_spacy_model}'. "
            "Install it (e.g. python -m spacy download en_core_web_sm)."
        ) from e

    if args.data:
        dataset_cfg = DatasetConfig(text_col=args.text_col, label_col=args.label_col)
        X_texts, y = load_dataset_from_csv(
            args.data,
            config=dataset_cfg,
            label_to_int=None,
            encoding="utf-8",
        )
    else:
        X_texts, y = load_sample_dataset()

    preprocess_cfg = PreprocessConfig(spacy_model=args.preprocess_spacy_model)

    tfidf_cfg = TfidfConfig(
        max_features=args.tfidf_max_features,
        ngram_range=(args.tfidf_ngram_min, args.tfidf_ngram_max),
        min_df=args.tfidf_min_df,
        max_df=args.tfidf_max_df,
    )

    training_cfg = TrainingConfig(
        feature_type=args.feature_type,
        use_random_state=args.random_state,
        test_size=args.test_size,
        val_size_within_train=args.val_within_train,
        threshold_grid_points=args.threshold_grid_points,
        random_forest_svd_components=args.random_forest_svd_components,
    )

    detector, metadata, results = train_and_select_best_model(
        X_texts,
        y,
        training_cfg=training_cfg,
        preprocess_cfg=preprocess_cfg,
        tfidf_cfg=tfidf_cfg,
        embeddings_spacy_model=args.embeddings_spacy_model,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[1] / out_path

    save_model(detector, metadata, out_path)

    print("Training complete. Metrics (test split, tuned threshold):")
    for model_name, res in results.items():
        print(f"  - {model_name}: {res.metrics} threshold={res.threshold:.3f}")

    # Quick smoke test: load and run one prediction.
    loaded_detector, _ = load_saved_model(out_path)
    example = X_texts[0]
    pred = loaded_detector.predict([example])[0]
    print(f"Loaded model smoke test: first-sample predicted={pred} (spam=1, ham=0).")


if __name__ == "__main__":
    main()

