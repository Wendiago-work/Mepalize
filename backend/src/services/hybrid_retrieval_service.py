#!/usr/bin/env python3
"""
Hybrid Retrieval Service for Translation Memory and Glossaries
Searches translation memory and glossaries using SPLADE hybrid search.
Style guides and cultural notes are handled by MongoDB Context Service.
"""

from typing import List, Dict, Any, Optional
from ..database.qdrant_client import QdrantVectorStore
from ..core.logger import get_logger

logger = get_logger("hybrid_retrieval_service", "rag_operations")

class HybridRetrievalService:
    """Hybrid retrieval service for translation memory and glossaries using SPLADE sparse vectors"""

    def __init__(self, qdrant_vector_store: QdrantVectorStore):
        self.qdrant_vector_store = qdrant_vector_store
    
    def initialize_services(self):
        """Initialize services using centralized model manager"""
        try:
            from ..core.model_manager import get_model_manager
            
            self.model_manager = get_model_manager()
            
            logger.info("✅ Hybrid retrieval services initialized with centralized model manager")
        except Exception as e:
            logger.error(f"❌ Failed to initialize hybrid retrieval services: {e}")
    
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        limit: int = 10,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search on translation memory and glossaries using SPLADE sparse vectors
        
        Args:
            query_embedding: Vector embedding for similarity search
            query_text: Text query for sparse vector generation
            limit: Maximum number of results
            source_language: Filter by source language
            target_language: Filter by target language
            domain: Filter by domain (e.g., gaming, entertainment)
            
        Returns:
            List of results from translation memory and glossaries with hybrid scores
        """
        try:
            # Initialize services if not already done
            if not hasattr(self, 'model_manager'):
                self.initialize_services()
            
            # Build additional filters for metadata filtering
            additional_filters = {}
            if source_language:
                additional_filters["source_language"] = source_language
            if target_language:
                additional_filters["target_language"] = target_language
            if domain:
                additional_filters["domain"] = domain
            
            # Calculate per-dataset limit to ensure we don't get too many results
            # If we have 2 datasets and want 4 total results, we need 2 per dataset
            per_dataset_limit = max(1, limit // 2)  # Ensure at least 1 per dataset
            
            # Perform multi-dataset hybrid search if query_text is provided
            if query_text and self.model_manager:
                # Generate sparse vector for hybrid search
                sparse_vector = self.model_manager.get_sparse_vector(query_text)
                
                # Use multi-dataset hybrid search with RRF fusion and metadata filtering
                results = await self.qdrant_vector_store.multi_dataset_hybrid_search(
                    query_vector=query_embedding,
                    sparse_vector=sparse_vector,
                    datasets=["translation_memory", "glossaries"],  # Only TM and glossaries in Qdrant
                    top_k=per_dataset_limit,  # Limit per dataset
                    additional_filters=additional_filters
                )
            else:
                # Fallback to multi-dataset dense search only
                results = await self.qdrant_vector_store.multi_dataset_hybrid_search(
                    query_vector=query_embedding,
                    sparse_vector=None,
                    datasets=["translation_memory", "glossaries"],  # Only TM and glossaries in Qdrant
                    top_k=per_dataset_limit,  # Limit per dataset
                    additional_filters=additional_filters
                )
            
            # Format results
            all_results = []
            for i, result in enumerate(results):
                all_results.append({
                    "id": result.id,
                    "score": result.score,
                    "rank": i + 1,
                    "payload": result.metadata,  # RetrievedDocument uses metadata, not payload
                    "content": result.content,
                    "search_type": "hybrid" if query_text else "dense"
                })
            
            # Sort all results by score (highest first)
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top results up to limit
            final_results = all_results[:limit]
            
            # Re-rank results
            for i, result in enumerate(final_results):
                result["rank"] = i + 1
            
            search_type = "hybrid" if query_text else "dense"
            logger.info(f"✅ {search_type} search completed: {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics for translation memory and glossaries"""
        try:
            # Get collection info using our corrected implementation
            stats = await self.qdrant_vector_store.get_collection_info()
            
            return {
                "translation_memory_and_glossaries": stats
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get retrieval stats: {e}")
            return {}

