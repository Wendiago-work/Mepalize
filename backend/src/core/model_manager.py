#!/usr/bin/env python3
"""
Centralized Model Manager for Localized Translator
Manages all ML models to avoid loading them multiple times
"""

import logging
from typing import Optional, Dict, Any
from ..config.config import get_settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized manager for all ML models"""
    
    def __init__(self):
        self.settings = get_settings()
        self._embedding_model = None
        self._sparse_model = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all models"""
        try:
            logger.info("🤖 Initializing centralized model manager...")
            
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Initialize sparse model
            await self._initialize_sparse_model()
            
            self._initialized = True
            logger.info("✅ Model manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model manager initialization failed: {e}")
            return False
    
    async def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            from fastembed import TextEmbedding
            
            # Get embedding model from config service
            from ..config.config import get_preprocessing_config
            preprocessing_config = get_preprocessing_config()
            embedding_model = preprocessing_config.embedding_model
            
            self._embedding_model = TextEmbedding(embedding_model)
            logger.info(f"✅ Embedding model initialized: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    async def _initialize_sparse_model(self):
        """Initialize the sparse model"""
        try:
            from fastembed import SparseTextEmbedding
            
            # Get sparse model from config service
            from ..config.config import get_preprocessing_config
            preprocessing_config = get_preprocessing_config()
            sparse_model = preprocessing_config.sparse_model
            
            self._sparse_model = SparseTextEmbedding(sparse_model)
            logger.info(f"✅ Sparse model initialized: {sparse_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sparse model: {e}")
            raise
    
    def get_embedding(self, text: str) -> list:
        """Get embedding for text"""
        if not self._initialized:
            raise RuntimeError("Model manager not initialized")
        
        # FastEmbed returns list of embeddings
        embeddings = list(self._embedding_model.embed([text]))
        return embeddings[0].tolist()
    
    def get_sparse_vector(self, text: str) -> Dict[str, Any]:
        """Get sparse vector for text"""
        if not self._initialized:
            raise RuntimeError("Model manager not initialized")
        
        try:
            # Generate sparse embedding
            sparse_embeddings = list(self._sparse_model.embed([text]))
            sparse_embedding = sparse_embeddings[0]
            
            # Convert to Qdrant's SparseVector format
            sparse_vector = {
                "indices": sparse_embedding.indices.tolist(),
                "values": sparse_embedding.values.tolist()
            }
            
            return sparse_vector
            
        except Exception as e:
            logger.error(f"Failed to generate sparse vector: {e}")
            return {"indices": [], "values": []}
    
    async def close(self):
        """Close model manager and cleanup resources"""
        try:
            # Models don't need explicit cleanup in most cases
            self._initialized = False
            logger.info("✅ Model manager closed")
        except Exception as e:
            logger.error(f"❌ Error closing model manager: {e}")

# Global model manager instance
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
