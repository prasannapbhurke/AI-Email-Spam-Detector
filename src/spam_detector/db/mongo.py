from __future__ import annotations

from typing import Optional

from pymongo import MongoClient


def create_mongo_client(mongo_uri: str) -> MongoClient:
    # `MongoClient` maintains internal connection pooling.
    return MongoClient(mongo_uri)


def get_collection():
    """
    Get MongoDB collection using settings from environment or defaults.
    Compatible with retrain_model.py usage.
    """
    import os
    from spam_detector.core.settings import Settings
    settings = Settings()
    client = create_mongo_client(settings.mongo_uri)
    return client[settings.mongo_db][settings.mongo_collection]

