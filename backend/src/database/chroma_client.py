"""
Chroma DB cloud client for Translation RAG Pipeline
Handles vector storage, similarity search, and collection management with separate collections
for translation memory and glossaries.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from uuid import uuid4
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult

from ..config import get_settings
from ..core.exceptions import DatabaseError, ErrorSeverity
from ..core.types import (
    DocumentId, Embedding, Score, RetrievedDocument
)
from ..core.logger import get_logger


@dataclass
class TranslationMemoryEntry:
    """Translation memory entry for Chroma storage"""
    id: str
    source_text: str
    target_text: str
    metadata: Dict[str, Any]
    
    def to_chroma_format(self) -> Dict[str, Any]:
        """Convert to Chroma document format"""
        return {
            "id": self.id,
            "document": self.source_text,  # Embed source text
            "metadata": {
                **self.metadata,
                "target_text": self.target_text  # Store target in metadata
            }
        }


@dataclass
class GlossaryEntry:
    """Glossary entry for Chroma storage"""
    id: str
    term: str
    definition: str
    metadata: Dict[str, Any]
    
    def to_chroma_format(self) -> Dict[str, Any]:
        """Convert to Chroma document format"""
        # Embed both term and definition together
        combined_text = f"{self.term}: {self.definition}"
        return {
            "id": self.id,
            "document": combined_text,
            "metadata": {
                **self.metadata,
                "term": self.term,
                "definition": self.definition
            }
        }


class ChromaVectorStore:
    """Chroma DB cloud client with separate collections for translation memory and glossaries"""
    
    def __init__(self, 
                 tm_collection_name: str = "translation_memory",
                 glossary_collection_name: str = "glossaries"):
        self.settings = get_settings()
        self.tm_collection_name = tm_collection_name
        self.glossary_collection_name = glossary_collection_name
        self.logger = get_logger("chroma_client", "vector_operations")
        
        # Initialize client
        self.client: Optional[ClientAPI] = None
        self.tm_collection: Optional[Collection] = None
        self.glossary_collection: Optional[Collection] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Chroma client and ensure collections exist"""
        if self._initialized:
            return
        
        # Retry logic for connection
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Check if Chroma DB credentials are provided
                if not all([self.settings.chroma_cloud_api_key, self.settings.chroma_cloud_tenant, self.settings.chroma_cloud_database]):
                    raise DatabaseError(
                        "Chroma DB credentials not provided. Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE environment variables.",
                        severity=ErrorSeverity.CRITICAL
                    )
                
                # Create Chroma Cloud client
                self.client = chromadb.CloudClient(
                    api_key=self.settings.chroma_cloud_api_key,
                    tenant=self.settings.chroma_cloud_tenant,
                    database=self.settings.chroma_cloud_database
                )
                
                self.logger.info(f"Using Chroma Cloud: tenant={self.settings.chroma_cloud_tenant}, database={self.settings.chroma_cloud_database}")
                
                # Test connection
                await self._test_connection()
                
                # Ensure collections exist
                await self._ensure_collections_exist()
                
                self._initialized = True
                self.logger.info("Chroma client initialized successfully", 
                               extra={"context": {
                                   "tm_collection": self.tm_collection_name,
                                   "glossary_collection": self.glossary_collection_name
                               }})
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Chroma connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    error_msg = f"Failed to initialize Chroma client after {max_retries} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    raise DatabaseError(
                        error_msg,
                        severity=ErrorSeverity.CRITICAL,
                        original_error=e
                    )
    
    async def _test_connection(self) -> None:
        """Test Chroma connection"""
        try:
            # Test with a simple health check
            collections = await asyncio.to_thread(self.client.list_collections)
            self.logger.info(f"Chroma connection successful, found {len(collections)} collections")
        except Exception as e:
            raise DatabaseError(
                f"Chroma connection test failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
    
    async def _ensure_collections_exist(self) -> None:
        """Ensure both collections exist"""
        try:
            # Get or create Translation Memory collection
            self.tm_collection = await asyncio.to_thread(
                self.client.get_or_create_collection,
                name=self.tm_collection_name,
                metadata={"description": "Translation memory for source-target text pairs"}
            )
            self.logger.info(f"Translation Memory collection '{self.tm_collection_name}' ready")
            
            # Get or create Glossaries collection
            self.glossary_collection = await asyncio.to_thread(
                self.client.get_or_create_collection,
                name=self.glossary_collection_name,
                metadata={"description": "Glossaries and terminology definitions"}
            )
            self.logger.info(f"Glossaries collection '{self.glossary_collection_name}' ready")
                
        except Exception as e:
            error_msg = f"Failed to ensure collections exist: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                original_error=e
            )
    
    async def add_translation_memory(self, entries: List[TranslationMemoryEntry]) -> List[str]:
        """Add translation memory documents to the TM collection"""
        if not self._initialized:
            await self.initialize()
        
        if not entries:
            return []
        
        try:
            # Convert to Chroma format
            ids = [entry.id for entry in entries]
            documents_list = [entry.source_text for entry in entries]
            metadatas = [entry.to_chroma_format()["metadata"] for entry in entries]
            
            # Add to Translation Memory collection
            await asyncio.to_thread(
                self.tm_collection.add,
                ids=ids,
                documents=documents_list,
                metadatas=metadatas
            )
            
            self.logger.info(f"Successfully added {len(entries)} translation memory entries")
            return ids
            
        except Exception as e:
            error_msg = f"Failed to add translation memory: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                operation="add_translation_memory",
                context={"num_entries": len(entries)},
                original_error=e
            )
    
    async def add_glossaries(self, entries: List[GlossaryEntry]) -> List[str]:
        """Add glossary documents to the glossaries collection"""
        if not self._initialized:
            await self.initialize()
        
        if not entries:
            return []
        
        try:
            # Convert to Chroma format
            ids = [entry.id for entry in entries]
            documents_list = [f"{entry.term}: {entry.definition}" for entry in entries]
            metadatas = [entry.to_chroma_format()["metadata"] for entry in entries]
            
            # Add to Glossaries collection
            await asyncio.to_thread(
                self.glossary_collection.add,
                ids=ids,
                documents=documents_list,
                metadatas=metadatas
            )
            
            self.logger.info(f"Successfully added {len(entries)} glossary entries")
            return ids
            
        except Exception as e:
            error_msg = f"Failed to add glossaries: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                operation="add_glossaries",
                context={"num_entries": len(entries)},
                original_error=e
            )
    
    async def search_translation_memory(self, 
                                      query: str,
                                      top_k: int = 10,
                                      filters: Optional[Dict[str, Any]] = None,
                                      similarity_threshold: float = 0.0) -> List[RetrievedDocument]:
        """Search translation memory collection"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build where clause for filtering
            where_clause = self._build_where_clause(filters) if filters else None
            
            # Search Translation Memory collection
            results = await asyncio.to_thread(
                self.tm_collection.query,
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
            
            # Convert to RetrievedDocument objects
            documents = self._convert_query_results(results)
            
            self.logger.info(
                f"Translation memory search returned {len(documents)} results",
                extra={"context": {
                    "query": query,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "has_filters": filters is not None
                }}
            )
            return documents
            
        except Exception as e:
            error_msg = f"Translation memory search failed: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                operation="search_translation_memory",
                context={
                    "query": query,
                    "top_k": top_k,
                    "filters": filters
                },
                original_error=e
            )
    
    async def search_glossaries(self, 
                               query: str,
                               top_k: int = 10,
                               filters: Optional[Dict[str, Any]] = None,
                               similarity_threshold: float = 0.0) -> List[RetrievedDocument]:
        """Search glossaries collection"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build where clause for filtering
            where_clause = self._build_where_clause(filters) if filters else None
            
            # Search Glossaries collection
            results = await asyncio.to_thread(
                self.glossary_collection.query,
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )
            
            # Convert to RetrievedDocument objects
            documents = self._convert_query_results(results)
            
            self.logger.info(
                f"Glossaries search returned {len(documents)} results",
                extra={"context": {
                    "query": query,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "has_filters": filters is not None
                }}
            )
            return documents
            
        except Exception as e:
            error_msg = f"Glossaries search failed: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                operation="search_glossaries",
                context={
                    "query": query,
                    "top_k": top_k,
                    "filters": filters
                },
                original_error=e
            )
    
    async def multi_collection_search(self, 
                                    query: str,
                                    datasets: List[str] = None,
                                    top_k: int = 10,
                                    additional_filters: Optional[Dict[str, Any]] = None) -> Dict[str, List[RetrievedDocument]]:
        """
        Search across both translation memory and glossaries collections
        
        Args:
            query: Search query text
            datasets: List of datasets to search (default: ["translation_memory", "glossaries"])
            top_k: Maximum number of results per collection
            additional_filters: Additional filters to apply
            
        Returns:
            Dictionary with results from each collection
        """
        if not self._initialized:
            await self.initialize()
        
        # Default datasets if none provided
        if datasets is None:
            datasets = ["translation_memory", "glossaries"]
        
        results = {}
        
        try:
            # Search Translation Memory if requested
            if "translation_memory" in datasets:
                tm_results = await self.search_translation_memory(
                    query=query,
                    top_k=top_k,
                    filters=additional_filters
                )
                results["translation_memory"] = tm_results
            
            # Search Glossaries if requested
            if "glossaries" in datasets:
                glossary_results = await self.search_glossaries(
                    query=query,
                    top_k=top_k,
                    filters=additional_filters
                )
                results["glossaries"] = glossary_results
            
            self.logger.info(
                f"Multi-collection search completed",
                extra={"context": {
                    "query": query,
                    "datasets": datasets,
                    "top_k": top_k,
                    "tm_results": len(results.get("translation_memory", [])),
                    "glossary_results": len(results.get("glossaries", []))
                }}
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Multi-collection search failed: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                operation="multi_collection_search",
                context={
                    "query": query,
                    "datasets": datasets,
                    "top_k": top_k
                },
                original_error=e
            )
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Chroma where clause from filters"""
        if not filters:
            return None
        
        where_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                where_conditions.append({key: {"$eq": value}})
            elif isinstance(value, list):
                where_conditions.append({key: {"$in": value}})
            elif isinstance(value, dict) and "range" in value:
                range_val = value["range"]
                range_condition = {}
                if "gte" in range_val:
                    range_condition["$gte"] = range_val["gte"]
                if "lte" in range_val:
                    range_condition["$lte"] = range_val["lte"]
                if "gt" in range_val:
                    range_condition["$gt"] = range_val["gt"]
                if "lt" in range_val:
                    range_condition["$lt"] = range_val["lt"]
                where_conditions.append({key: range_condition})
        
        return {"$and": where_conditions} if where_conditions else None
    
    def _convert_query_results(self, results: QueryResult) -> List[RetrievedDocument]:
        """Convert Chroma query results to RetrievedDocument objects"""
        documents = []
        
        if not results or not results.get("documents"):
            return documents
        
        # Extract results from Chroma format
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        docs = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        
        for i, (doc_id, content, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances)):
            # Convert distance to similarity score (Chroma uses distance, we want similarity)
            # Ensure score is between 0 and 1
            if distance is not None:
                score = max(0.0, min(1.0, 1.0 - distance))
            else:
                score = 1.0
            
            documents.append(
                RetrievedDocument(
                    id=str(doc_id),
                    content=content,
                    metadata=metadata or {},
                    score=float(score)
                )
            )
        
        return documents
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and information"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get TM collection info
            tm_count = await asyncio.to_thread(self.tm_collection.count)
            
            # Get Glossaries collection info
            glossary_count = await asyncio.to_thread(self.glossary_collection.count)
            
            return {
                "translation_memory": {
                    "name": self.tm_collection_name,
                    "count": tm_count
                },
                "glossaries": {
                    "name": self.glossary_collection_name,
                    "count": glossary_count
                },
                "total_documents": tm_count + glossary_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            raise DatabaseError(
                f"Failed to get collection info: {str(e)}",
                original_error=e
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Lightweight health check that doesn't query data"""
        if not self._initialized:
            return {"status": "unhealthy", "message": "Client not initialized"}
        
        try:
            # Just check if collections exist without querying data
            # This is much cheaper than calling count()
            tm_name = self.tm_collection.name
            glossary_name = self.glossary_collection.name
            
            return {
                "status": "healthy",
                "message": "Chroma DB connection active",
                "collections": {
                    "translation_memory": tm_name,
                    "glossaries": glossary_name
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "message": f"Chroma DB health check failed: {str(e)}"
            }
    
    async def delete_documents(self, doc_ids: List[DocumentId], collection: str = "translation_memory") -> bool:
        """Delete documents by IDs from specified collection"""
        if not self._initialized:
            await self.initialize()
        
        try:
            target_collection = self.tm_collection if collection == "translation_memory" else self.glossary_collection
            
            await asyncio.to_thread(
                target_collection.delete,
                ids=[str(doc_id) for doc_id in doc_ids]
            )
            
            self.logger.info(f"Deleted {len(doc_ids)} documents from {collection}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete documents from {collection}: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                original_error=e
            )
    
    async def clear_collection(self, collection: str = "translation_memory") -> None:
        """Clear all documents from specified collection"""
        try:
            if not self._initialized:
                await self.initialize()
            
            target_collection = self.tm_collection if collection == "translation_memory" else self.glossary_collection
            
            # Get all document IDs and delete them
            all_docs = await asyncio.to_thread(target_collection.get)
            if all_docs.get("ids"):
                await asyncio.to_thread(
                    target_collection.delete,
                    ids=all_docs["ids"]
                )
            
            self.logger.info(f"✅ Cleared collection: {collection}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear collection {collection}: {e}")
            raise DatabaseError(f"Failed to clear collection {collection}: {e}")
    
    async def close(self) -> None:
        """Close the Chroma client connection"""
        try:
            if self.client:
                # Chroma client doesn't need explicit closing
                self.client = None
                self.tm_collection = None
                self.glossary_collection = None
                self._initialized = False
                self.logger.info("Chroma client connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Chroma client: {str(e)}")


# Factory function for easy client creation
async def create_chroma_client(tm_collection_name: str = "translation_memory",
                              glossary_collection_name: str = "glossaries") -> ChromaVectorStore:
    """Factory function to create and initialize Chroma client"""
    client = ChromaVectorStore(tm_collection_name, glossary_collection_name)
    await client.initialize()
    return client


if __name__ == "__main__":
    print("ChromaVectorStore - Use test_chroma_search.py to test the implementation")
