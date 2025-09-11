"""
Chroma Retrieval Service for Translation Memory and Glossaries
Searches translation memory and glossaries using Chroma DB with separate collections.
"""

from typing import List, Dict, Any, Optional
from ..database.chroma_client import ChromaVectorStore
from ..core.types import RetrievedDocument
from ..core.logger import get_logger


class ChromaRetrievalService:
    """Retrieval service for translation memory and glossaries using Chroma DB"""
    
    def __init__(self, chroma_vector_store: ChromaVectorStore):
        self.chroma_vector_store = chroma_vector_store
        self.logger = get_logger("chroma_retrieval_service", "retrieval")
    
    async def search_collections(self, 
                                query: str,
                                source_language: str = None,
                                target_language: str = None,
                                domain: str = None,
                                top_k: int = 10,
                                similarity_threshold: float = 0.3) -> Dict[str, List[RetrievedDocument]]:
        """
        Search both translation memory and glossaries collections using Chroma DB
        
        Args:
            query: Search query text
            source_language: Filter by source language (e.g., "en", "ja")
            target_language: Filter by target language (e.g., "en", "ja")
            domain: Filter by domain (e.g., "Game - Music", "Entertainment")
            top_k: Maximum number of results per collection
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary with results from translation memory and glossaries
        """
        try:
            # Build filters for language and domain
            filters = {}
            if source_language:
                filters["source_language"] = source_language
            if target_language:
                filters["target_language"] = target_language
            if domain:
                filters["domain"] = domain
            
            # Search both collections
            results = await self.chroma_vector_store.multi_collection_search(
                query=query,
                datasets=["translation_memory", "glossaries"],
                top_k=top_k,
                additional_filters=filters
            )
            
            # Filter results by similarity threshold
            filtered_results = {}
            for collection, docs in results.items():
                filtered_docs = [doc for doc in docs if doc.score >= similarity_threshold]
                filtered_results[collection] = filtered_docs
            
            self.logger.info(
                f"Collection search completed: {len(filtered_results.get('translation_memory', []))} TM, "
                f"{len(filtered_results.get('glossaries', []))} glossaries",
                extra={"context": {
                    "query": query,
                    "source_language": source_language,
                    "target_language": target_language,
                    "domain": domain,
                    "similarity_threshold": similarity_threshold
                }}
            )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Collection search failed: {str(e)}")
            raise
    
    async def search_translation_memory(self, 
                                      query: str,
                                      source_language: str = None,
                                      target_language: str = None,
                                      domain: str = None,
                                      top_k: int = 10,
                                      similarity_threshold: float = 0.7) -> List[RetrievedDocument]:
        """Search only translation memory collection"""
        try:
            # Build filters
            filters = {}
            if source_language:
                filters["source_language"] = source_language
            if target_language:
                filters["target_language"] = target_language
            if domain:
                filters["domain"] = domain
            
            # Search translation memory
            results = await self.chroma_vector_store.search_translation_memory(
                query=query,
                top_k=top_k,
                filters=filters,
                similarity_threshold=similarity_threshold
            )
            
            # Filter by similarity threshold
            filtered_results = [doc for doc in results if doc.score >= similarity_threshold]
            
            self.logger.info(
                f"Translation memory search: {len(filtered_results)} results",
                extra={"context": {
                    "query": query,
                    "source_language": source_language,
                    "target_language": target_language,
                    "domain": domain
                }}
            )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Translation memory search failed: {str(e)}")
            raise
    
    async def search_glossaries(self, 
                               query: str,
                               source_language: str = None,
                               target_language: str = None,
                               domain: str = None,
                               top_k: int = 10,
                               similarity_threshold: float = 0.7) -> List[RetrievedDocument]:
        """Search only glossaries collection"""
        try:
            # Build filters
            filters = {}
            if source_language:
                filters["source_language"] = source_language
            if target_language:
                filters["target_language"] = target_language
            if domain:
                filters["domain"] = domain
            
            # Search glossaries
            results = await self.chroma_vector_store.search_glossaries(
                query=query,
                top_k=top_k,
                filters=filters,
                similarity_threshold=similarity_threshold
            )
            
            # Filter by similarity threshold
            filtered_results = [doc for doc in results if doc.score >= similarity_threshold]
            
            self.logger.info(
                f"Glossaries search: {len(filtered_results)} results",
                extra={"context": {
                    "query": query,
                    "source_language": source_language,
                    "target_language": target_language,
                    "domain": domain
                }}
            )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Glossaries search failed: {str(e)}")
            raise
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics for translation memory and glossaries"""
        try:
            stats = await self.chroma_vector_store.get_collection_info()
            return {
                "translation_memory_and_glossaries": stats
            }
        except Exception as e:
            self.logger.error(f"Failed to get retrieval stats: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Lightweight health check that doesn't query data"""
        try:
            health = await self.chroma_vector_store.health_check()
            return {
                "retrieval_service": health.get("status", "unknown"),
                "message": "Chroma retrieval service active"
            }
        except Exception as e:
            return {
                "retrieval_service": "unhealthy",
                "message": f"Retrieval service health check failed: {str(e)}"
            }


def create_chroma_retrieval_service(chroma_vector_store: ChromaVectorStore) -> ChromaRetrievalService:
    """Factory function to create Chroma retrieval service"""
    return ChromaRetrievalService(chroma_vector_store)
