from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = "models/spam_detector.joblib"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "spam_detector"
    mongo_collection: str = "predictions"

    log_level: str = "INFO"

    # Mongo writes can be disabled for local development by setting this to false.
    enable_db_writes: bool = True

    # Control how many hours of history stats consider (basic analytics).
    stats_last_hours: int = 24

    model_spam_label: int = 1  # spam class id in the trained model

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

