"""
Chroma DB Cloud Connection Module
Provides FastAPI dependency injection for Chroma collections
"""

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from fastapi import Depends
from typing import Optional

from ..config import get_settings
from ..core.logger import get_logger

# Global client and collections
_client: Optional[ClientAPI] = None
_tm_collection: Optional[Collection] = None
_glossary_collection: Optional[Collection] = None

logger = get_logger("chroma_connection", "database")


def get_chroma_client() -> ClientAPI:
    """Get or create Chroma Cloud client"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.CloudClient(
            api_key=settings.chroma_cloud_api_key,
            tenant=settings.chroma_cloud_tenant,
            database=settings.chroma_cloud_database
        )
        logger.info(f"Chroma Cloud client initialized: tenant={settings.chroma_cloud_tenant}, database={settings.chroma_cloud_database}")
    return _client


def get_tm_collection(client: ClientAPI = Depends(get_chroma_client)) -> Collection:
    """Get or create Translation Memory collection"""
    global _tm_collection
    if _tm_collection is None:
        settings = get_settings()
        _tm_collection = client.get_or_create_collection(
            name=settings.tm_collection_name,
            metadata={"description": "Translation memory for source-target text pairs"}
        )
        logger.info(f"Translation Memory collection '{settings.tm_collection_name}' ready")
    return _tm_collection


def get_glossary_collection(client: ClientAPI = Depends(get_chroma_client)) -> Collection:
    """Get or create Glossaries collection"""
    global _glossary_collection
    if _glossary_collection is None:
        settings = get_settings()
        _glossary_collection = client.get_or_create_collection(
            name=settings.glossary_collection_name,
            metadata={"description": "Glossaries and terminology definitions"}
        )
        logger.info(f"Glossaries collection '{settings.glossary_collection_name}' ready")
    return _glossary_collection


def get_both_collections(
    tm_collection: Collection = Depends(get_tm_collection),
    glossary_collection: Collection = Depends(get_glossary_collection)
) -> dict:
    """Get both collections as a dictionary"""
    return {
        "translation_memory": tm_collection,
        "glossaries": glossary_collection
    }
