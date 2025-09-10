"""
Qdrant vector database client for Translation RAG Pipeline
Handles vector storage, similarity search, and collection management
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from uuid import uuid4
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.conversions import common_types
from qdrant_client.models import SparseVector
from qdrant_client.http import models as rest

from ..config import get_settings
from ..core.exceptions import QdrantError, DatabaseError, ErrorSeverity
from ..core.types import (
    DocumentId, Embedding, Score, RetrievedDocument, SearchType
)
from ..core.logger import get_logger


@dataclass
class VectorDocument:
    """Document with vector representation for Qdrant storage"""
    id: str
    content: str
    embedding: Embedding
    metadata: Dict[str, Any]
    sparse_vector: Optional[Dict[str, Any]] = None  # For hybrid search with proper format
    
    def to_qdrant_point(self) -> models.PointStruct:
        """Convert to Qdrant point structure with hybrid vectors"""
        from qdrant_client.models import SparseVector
        
        # Create the vector dictionary with both dense and sparse vectors
        vector_dict = {"m3_dense": self.embedding}
        
        # Add sparse vector if available - it goes in the same vector dict
        if self.sparse_vector and "indices" in self.sparse_vector and "values" in self.sparse_vector:
            sparse_vec = SparseVector(
                indices=self.sparse_vector["indices"],
                values=self.sparse_vector["values"]
            )
            vector_dict["sparse_vector"] = sparse_vec
        
        # Create the point structure with proper vector format
        point_data = {
            "id": self.id,
            "vector": vector_dict,
            "payload": {
                "content": self.content,
                **self.metadata
            }
        }
            
        return models.PointStruct(**point_data)


class QdrantVectorStore:
    """Qdrant vector database client with gaming translation optimizations"""
    
    def __init__(self, collection_name: str = None):
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.qdrant_collection_name
        self.logger = get_logger("qdrant_client", "vector_operations")
        
        # Initialize client
        self.client: Optional[QdrantClient] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Qdrant client and ensure collection exists"""
        if self._initialized:
            return
        
        # Retry logic for connection
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Create Qdrant client with cloud or local configuration
                if self.settings.use_qdrant_cloud and self.settings.qdrant_cloud_url:
                    # Use Qdrant Cloud
                    if not self.settings.qdrant_cloud_api_key:
                        raise QdrantError(
                            "Qdrant Cloud API key is required when using cloud mode",
                            ErrorSeverity.CRITICAL
                        )
                    self.client = QdrantClient(
                        url=self.settings.qdrant_cloud_url,
                        api_key=self.settings.qdrant_cloud_api_key,
                        timeout=self.settings.connection_timeout
                    )
                    self.logger.info(f"Using Qdrant Cloud: {self.settings.qdrant_cloud_url}")
                else:
                    # Use local Qdrant
                    qdrant_url = f"http://{self.settings.qdrant_host}:{self.settings.qdrant_port}"
                    
                    if self.settings.qdrant_api_key:
                        self.client = QdrantClient(
                            url=qdrant_url,
                            api_key=self.settings.qdrant_api_key,
                            timeout=self.settings.connection_timeout
                        )
                    else:
                        self.client = QdrantClient(
                            url=qdrant_url,
                            timeout=self.settings.connection_timeout
                        )
                    self.logger.info(f"Using local Qdrant: {qdrant_url}")
                
                # Test connection
                await self._test_connection()
                
                # Ensure collection exists
                await self._ensure_collection_exists()
                
                self._initialized = True
                self.logger.info("Qdrant client initialized successfully", 
                               extra={"context": {"collection": self.collection_name}})
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Qdrant connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    error_msg = f"Failed to initialize Qdrant client after {max_retries} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    raise QdrantError(
                        error_msg,
                        collection_name=self.collection_name,
                        severity=ErrorSeverity.CRITICAL,
                        original_error=e
                    )
    
    async def _test_connection(self) -> None:
        """Test Qdrant connection"""
        try:
            # Test with a simple health check
            collections = await asyncio.to_thread(self.client.get_collections)
            self.logger.info(f"Qdrant connection successful, found {len(collections.collections)} collections")
        except Exception as e:
            raise QdrantError(
                f"Qdrant connection test failed: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                original_error=e
            )
    
    async def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists with proper configuration"""
        try:
            # Check if collection exists
            try:
                collection_info = await asyncio.to_thread(
                    self.client.get_collection, self.collection_name
                )
                self.logger.info(f"Collection '{self.collection_name}' already exists")
                return
            except UnexpectedResponse as e:
                if e.status_code == 404:
                    # Collection doesn't exist, create it
                    pass
                else:
                    raise
            
            from qdrant_client.models import VectorParams, SparseVectorParams, SparseIndexParams
            
            # Create collection with proper hybrid vector support using the correct API
            await asyncio.to_thread(
                self.client.recreate_collection,
                collection_name=self.collection_name,
                vectors_config={
                    'm3_dense': VectorParams(
                        size=self.settings.embedding_dimension, 
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    'sparse_vector': SparseVectorParams(index=None)
                }
            )
            
            self.logger.info(f"Created collection '{self.collection_name}' successfully")
            

            
        except Exception as e:
            error_msg = f"Failed to ensure collection exists: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                original_error=e
            )
    

    
    async def add_vectors(self, documents: List[VectorDocument]) -> List[str]:
        """Add vector documents to the collection"""
        if not self._initialized:
            await self.initialize()
        
        if not documents:
            return []
        
        try:
            # Convert to Qdrant points
            points = [doc.to_qdrant_point() for doc in documents]
            
            # Add points in batches
            batch_size = self.settings.batch_size
            added_ids = []
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                
                batch_ids = [str(point.id) for point in batch]
                added_ids.extend(batch_ids)
                
                self.logger.info(f"Added batch of {len(batch)} vectors to collection")
            
            self.logger.info(f"Successfully added {len(added_ids)} vectors to collection")
            return added_ids
            
        except Exception as e:
            error_msg = f"Failed to add vectors: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                operation="add_vectors",
                context={"num_documents": len(documents)},
                original_error=e
            )
    
    async def search_vectors(self, query_vector: Embedding, top_k: int = 10,
                           filters: Optional[Dict[str, Any]] = None,
                           similarity_threshold: float = 0.0,
                           sparse_vector: Optional[Dict[str, Any]] = None) -> List[RetrievedDocument]:
        """
        DEPRECATED: Use multi_dataset_hybrid_search instead.
        Search for similar vectors using proper hybrid search with Query API and RRF fusion.
        This method is kept for backward compatibility but should not be used in new code.
        """
        # Deprecation warning
        import warnings
        warnings.warn(
            "search_vectors is deprecated. Use multi_dataset_hybrid_search instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build filter conditions
            q_filter = self._build_qdrant_filter(filters) if filters else None
            
            # Use proper hybrid search if sparse vector is provided
            if sparse_vector and "indices" in sparse_vector and "values" in sparse_vector:
                from qdrant_client.models import SparseVector
                
                # Create sparse vector object
                query_sparse = SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                )
                
                # NOTE: prefetch is TOP-LEVEL; FusionQuery only carries the fusion algorithm.
                search_results = await asyncio.to_thread(
                    self.client.query_points,
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=query_vector,           
                            using="m3_dense",         
                            limit=top_k,
                            filter=q_filter, 
                        ),
                        models.Prefetch(
                            query=query_sparse,           
                            using="sparse_vector",        
                            limit=top_k,
                            filter=q_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                    # with_vectors=False,  # optional; omit if you don't need vectors back
                    score_threshold=similarity_threshold,
                    # filter=q_filter, 
                )

                # Unwrap response
                hits = search_results.points
                
            else:
                # Dense vector only - use regular search with NamedVector
                from qdrant_client.models import NamedVector
                
                hits = await asyncio.to_thread(
                    self.client.search,
                    collection_name=self.collection_name,
                    query_vector=NamedVector(name="m3_dense", vector=query_vector),
                    limit=top_k,
                    query_filter=q_filter,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=similarity_threshold
                )
            
            # Convert to RetrievedDocument objects
            documents: List[RetrievedDocument] = []
            for r in hits:
                payload = r.payload or {}
                content = payload.pop("content", "")
                documents.append(
                    RetrievedDocument(
                        id=str(r.id),
                        content=content,
                        metadata=payload,
                        score=float(r.score),
                        retrieval_method=SearchType.VECTOR,
                    )
                )

            self.logger.info(
                f'{"hybrid" if sparse_vector else "dense"} search returned {len(documents)} results',
                extra={"context": {
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "has_filters": filters is not None,
                    "search_type": "hybrid" if sparse_vector else "dense",
                }},
            )
            return documents
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                operation="search_vectors",
                context={
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "filters": filters,
                    "has_sparse_vector": sparse_vector is not None
                },
                original_error=e
            )
    

    

    
    async def multi_dataset_hybrid_search(self, query_vector: Embedding, 
                                        sparse_vector: Optional[Dict[str, Any]] = None,
                                        datasets: List[str] = None,
                                        top_k: int = 10,
                                        similarity_threshold: float = 0.0,
                                        additional_filters: Optional[Dict[str, Any]] = None) -> List[RetrievedDocument]:
        """
        Perform hybrid search across translation memory and glossaries using prefetch filtering with RRF fusion.
        
        This implements the recommended single-collection approach with dataset filtering:
        - Uses prefetch branches with per-dataset filters
        - Guarantees coverage per dataset
        - Fuses results with RRF (Reciprocal Rank Fusion)
        - Supports filtering by source_language, target_language, and domain
        
        Args:
            query_vector: Dense vector for similarity search
            sparse_vector: Sparse vector for hybrid search (optional)
            datasets: List of dataset names to search (default: ["translation_memory", "glossaries"])
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            additional_filters: Dict of additional filters (source_language, target_language, domain)
            
        Returns:
            List of RetrievedDocument objects with fused results
        """
        if not self._initialized:
            await self.initialize()
        
        # Default datasets if none provided - only translation memory and glossaries
        if datasets is None:
            datasets = ["translation_memory", "glossaries"]
        
        try:
            from qdrant_client.models import SparseVector, Prefetch, FusionQuery
            
            # Build prefetch branches for each dataset
            prefetch_branches = []
            
            for dataset in datasets:
                # Create dataset filter
                filter_conditions = [models.FieldCondition(
                    key="dataset", 
                    match=models.MatchValue(value=dataset)
                )]
                
                # Add additional filters if provided
                if additional_filters:
                    for key, value in additional_filters.items():
                        if isinstance(value, (str, int, float, bool)):
                            filter_conditions.append(models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            ))
                        elif isinstance(value, list):
                            filter_conditions.append(models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            ))
                
                dataset_filter = models.Filter(must=filter_conditions)
                
                # Add dense vector prefetch for this dataset
                prefetch_branches.append(
                    models.Prefetch(
                        query=query_vector,
                        using="m3_dense",
                        limit=top_k,
                        filter=dataset_filter
                    )
                )
                
                # Add sparse vector prefetch for this dataset (if available)
                if sparse_vector and "indices" in sparse_vector and "values" in sparse_vector:
                    query_sparse = SparseVector(
                        indices=sparse_vector["indices"],
                        values=sparse_vector["values"]
                    )
                    
                    prefetch_branches.append(
                        models.Prefetch(
                            query=query_sparse,
                            using="sparse_vector",
                            limit=top_k, 
                            filter=dataset_filter
                        )
                    )
            
            # Perform multi-dataset hybrid search with RRF fusion
            search_results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                prefetch=prefetch_branches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                score_threshold=similarity_threshold
            )
            
            # Convert to RetrievedDocument objects
            documents: List[RetrievedDocument] = []
            for r in search_results.points:
                payload = r.payload or {}
                content = payload.pop("content", "")
                documents.append(
                    RetrievedDocument(
                        id=str(r.id),
                        content=content,
                        metadata=payload,
                        score=float(r.score),
                        retrieval_method=SearchType.VECTOR,
                    )
                )
            
            self.logger.info(
                f"Multi-dataset hybrid search returned {len(documents)} results",
                extra={"context": {
                    "datasets": datasets,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "has_sparse_vector": sparse_vector is not None,
                    "prefetch_branches": len(prefetch_branches)
                }},
            )
            
            return documents
            
        except Exception as e:
            error_msg = f"Multi-dataset hybrid search failed: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                operation="multi_dataset_hybrid_search",
                context={
                    "datasets": datasets,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "has_sparse_vector": sparse_vector is not None
                },
                original_error=e
            )
    
    def generate_sparse_vector(self, text: str) -> Dict[str, Any]:
        """
        Generate proper sparse vector for Qdrant using SPLADE via FastEmbed
        
        Returns a dictionary with 'indices' and 'values' for Qdrant's SparseVector format
        """
        try:
            from fastembed import SparseTextEmbedding
            
            # Initialize SPLADE model for sparse embeddings
            sparse_model = SparseTextEmbedding("prithivida/splade_pp_en")
            
            # Generate sparse embedding
            sparse_embeddings = list(sparse_model.embed([text]))
            sparse_embedding = sparse_embeddings[0]
            
            # Convert to Qdrant's SparseVector format
            # SPLADE returns indices and values
            sparse_vector = {
                "indices": sparse_embedding.indices.tolist(),
                "values": sparse_embedding.values.tolist()
            }
            
            return sparse_vector
            
        except ImportError:
            self.logger.error("FastEmbed SPLADE not available - this is required for hybrid search")
            return {"indices": [], "values": []}
        except Exception as e:
            self.logger.error(f"Failed to generate SPLADE sparse vector: {e}")
            return {"indices": [], "values": []}
    

    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from dictionary"""
        must_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                condition = models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
                must_conditions.append(condition)
            elif isinstance(value, list):
                condition = models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value)
                )
                must_conditions.append(condition)
            elif isinstance(value, dict) and "range" in value:
                # Handle range queries
                range_val = value["range"]
                condition = models.FieldCondition(
                    key=key,
                    range=models.Range(
                        gte=range_val.get("gte"),
                        lte=range_val.get("lte"),
                        gt=range_val.get("gt"),
                        lt=range_val.get("lt")
                    )
                )
                must_conditions.append(condition)
        
        return models.Filter(must=must_conditions) if must_conditions else None
    
    async def get_document(self, doc_id: DocumentId) -> Optional[RetrievedDocument]:
        """Retrieve a specific document by ID"""
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await asyncio.to_thread(
                self.client.retrieve,
                collection_name=self.collection_name,
                ids=[str(doc_id)],
                with_payload=True,
                with_vectors=False
            )
            
            if not result:
                return None
            
            point = result[0]
            payload = point.payload or {}
            content = payload.pop("content", "")
            
            return RetrievedDocument(
                id=str(point.id),
                content=content,
                metadata=payload,
                score=1.0,  # Direct retrieval, perfect match
                retrieval_method=SearchType.VECTOR
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id}: {str(e)}")
            raise QdrantError(
                f"Failed to retrieve document: {str(e)}",
                collection_name=self.collection_name,
                vector_id=str(doc_id),
                original_error=e
            )
    
    async def delete_documents(self, doc_ids: List[DocumentId]) -> bool:
        """Delete documents by IDs"""
        if not self._initialized:
            await self.initialize()
        
        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[str(doc_id) for doc_id in doc_ids]
                ),
                wait=True
            )
            
            self.logger.info(f"Deleted {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete documents: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                original_error=e
            )
    
    async def update_document(self, doc_id: DocumentId, 
                            new_metadata: Dict[str, Any]) -> bool:
        """Update document metadata"""
        if not self._initialized:
            await self.initialize()
        
        try:
            await asyncio.to_thread(
                self.client.set_payload,
                collection_name=self.collection_name,
                payload=new_metadata,
                points=[str(doc_id)],
                wait=True
            )
            
            self.logger.info(f"Updated document {doc_id} metadata")
            return True
            
        except Exception as e:
            error_msg = f"Failed to update document {doc_id}: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                vector_id=str(doc_id),
                original_error=e
            )
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and information"""
        if not self._initialized:
            await self.initialize()
        
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.collection_name
            )
            
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status.name,
                "optimizer_status": collection_info.optimizer_status,
                "vector_size": getattr(collection_info.config.params.vectors, 'size', 'multiple_vectors')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            raise QdrantError(
                f"Failed to get collection info: {str(e)}",
                collection_name=self.collection_name,
                original_error=e
            )
    
    async def delete_by_metadata_filter(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Delete documents by metadata filter using Qdrant client's native methods
        
        Args:
            metadata_filter: Dictionary containing filter conditions
                Example: {
                    "dataset": "translation_memory",
                    "domain": "Game - Music",
                    "source_language": "en"
                }
        
        Returns:
            Number of documents deleted
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build Qdrant filter from metadata conditions using client models
            filter_conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, list):
                    # Handle list values (e.g., multiple domains)
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    # Handle single values
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            
            # Create the filter using client models
            qdrant_filter = models.Filter(must=filter_conditions)
            
            # First, count how many documents match the filter
            count_result = await asyncio.to_thread(
                self.client.count,
                collection_name=self.collection_name,
                count_filter=qdrant_filter
            )
            
            matched_count = count_result.count if count_result else 0
            
            if matched_count == 0:
                self.logger.info(f"No documents found matching filter: {metadata_filter}")
                return 0
            
            # Delete documents matching the filter using client's delete method
            delete_result = await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=qdrant_filter),
                wait=True
            )
            
            self.logger.info(f"✅ Deleted {matched_count} documents matching filter: {metadata_filter}")
            return matched_count
            
        except Exception as e:
            error_msg = f"Failed to delete documents by metadata filter: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                operation="delete_by_metadata_filter",
                context={"metadata_filter": metadata_filter},
                original_error=e
            )
    
    async def delete_by_dataset(self, dataset: str) -> int:
        """
        Delete all documents from a specific dataset
        
        Args:
            dataset: Dataset name (e.g., "translation_memory", "glossaries")
        
        Returns:
            Number of documents deleted
        """
        return await self.delete_by_metadata_filter({"dataset": dataset})
    
    async def delete_by_domain(self, domain: str) -> int:
        """
        Delete all documents from a specific domain
        
        Args:
            domain: Domain name (e.g., "Game - Music", "Entertainment")
        
        Returns:
            Number of documents deleted
        """
        return await self.delete_by_metadata_filter({"domain": domain})
    
    async def delete_by_language(self, source_language: str = None, target_language: str = None) -> int:
        """
        Delete documents by language filters
        
        Args:
            source_language: Source language code (e.g., "en", "ja")
            target_language: Target language code (e.g., "en", "ja")
        
        Returns:
            Number of documents deleted
        """
        filter_dict = {}
        if source_language:
            filter_dict["source_language"] = source_language
        if target_language:
            filter_dict["target_language"] = target_language
        
        if not filter_dict:
            raise ValueError("At least one language filter must be provided")
        
        return await self.delete_by_metadata_filter(filter_dict)
    
    async def delete_by_file_source(self, file_source: str) -> int:
        """
        Delete documents from a specific file source
        
        Args:
            file_source: File name or path (e.g., "JPi - Localization Ref For AI - Translation Memory.csv")
        
        Returns:
            Number of documents deleted
        """
        return await self.delete_by_metadata_filter({"file_source": file_source})
    
    async def clear_collection(self) -> None:
        """Clear all documents from the collection (for data refresh)"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Method 1: Use Qdrant's native delete with empty filter (deletes all)
            # This is more efficient than the large range approach
            try:
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter()  # Empty filter matches all points
                    )
                )
            except Exception:
                # Fallback: Use the large range approach if empty filter doesn't work
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="id",
                                    match=models.MatchAny(any=[str(i) for i in range(1000000)])
                                )
                            ]
                        )
                    )
                )
            
            self.logger.info(f"✅ Cleared collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear collection: {e}")
            raise QdrantError(f"Failed to clear collection: {e}")
    
    async def batch_search(self, query_vectors: List[Embedding], 
                          top_k: int = 10,
                          filters: Optional[Dict[str, Any]] = None) -> List[List[RetrievedDocument]]:
        """Perform batch vector search for multiple queries"""
        if not self._initialized:
            await self.initialize()
        
        try:
            q_filter = self._build_qdrant_filter(filters) if filters else None
            
            # Prepare search requests
            search_requests = []
            for query_vector in query_vectors:
                search_requests.append(
                    models.SearchRequest(
                        vector=query_vector,
                        limit=top_k,
                        filter=q_filter,
                        with_payload=True,
                        with_vector=False
                    )
                )
            
            # Execute batch search
            batch_results = await asyncio.to_thread(
                self.client.search_batch,
                collection_name=self.collection_name,
                requests=search_requests
            )
            
            # Convert results
            all_results = []
            for search_results in batch_results:
                documents = []
                for result in search_results:
                    payload = result.payload or {}
                    content = payload.pop("content", "")
                    
                    doc = RetrievedDocument(
                        id=str(result.id),
                        content=content,
                        metadata=payload,
                        score=float(result.score),
                        retrieval_method=SearchType.VECTOR
                    )
                    documents.append(doc)
                all_results.append(documents)
            
            self.logger.info(f"Batch search completed for {len(query_vectors)} queries")
            return all_results
            
        except Exception as e:
            error_msg = f"Batch search failed: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                operation="batch_search",
                context={"num_queries": len(query_vectors)},
                original_error=e
            )
    
    async def create_payload_index(self, field_name: str, 
                                 field_type: str = "keyword") -> bool:
        """Create index on payload field for faster filtering"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Map field types
            qdrant_field_type = {
                "keyword": models.PayloadSchemaType.KEYWORD,
                "integer": models.PayloadSchemaType.INTEGER,
                "float": models.PayloadSchemaType.FLOAT,
                "geo": models.PayloadSchemaType.GEO,
                "text": models.PayloadSchemaType.TEXT
            }.get(field_type, models.PayloadSchemaType.KEYWORD)
            
            await asyncio.to_thread(
                self.client.create_payload_index,
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=qdrant_field_type,
                wait=True
            )
            
            self.logger.info(f"Created payload index on field '{field_name}' with type '{field_type}'")
            return True
            
        except Exception as e:
            error_msg = f"Failed to create payload index: {str(e)}"
            self.logger.error(error_msg)
            raise QdrantError(
                error_msg,
                collection_name=self.collection_name,
                context={"field_name": field_name, "field_type": field_type},
                original_error=e
            )
    
    async def setup_localization_indexes(self) -> None:
        """Setup indexes optimized for localization and content queries"""
        try:
            indexes_to_create = [
                ("dataset", "keyword"),        # For filtering by dataset type (translation_memory, glossaries)
                ("domain", "keyword"),         # For filtering by domain (Game - Music, Game - Casual, Entertainment)
                ("source_language", "keyword"), # For filtering by source language (en, fr, ja, etc.)
                ("target_language", "keyword"), # For filtering by target language (en, fr, ja, etc.)
            ]
            
            for field_name, field_type in indexes_to_create:
                try:
                    await self.create_payload_index(field_name, field_type)
                except QdrantError as e:
                    # Continue if index already exists
                    if "already exists" in str(e).lower():
                        self.logger.info(f"Index for '{field_name}' already exists")
                    else:
                        raise
            
            self.logger.info("Localization indexes setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup localization indexes: {str(e)}")
            raise QdrantError(
                f"Failed to setup localization indexes: {str(e)}",
                collection_name=self.collection_name,
                original_error=e
            )
    
    async def cleanup_old_vectors(self, older_than_days: int = 30) -> int:
        """Cleanup old vectors based on timestamp"""
        if not self._initialized:
            await self.initialize()
        
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
            
            # Create filter for old vectors
            old_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="created_at",
                        range=models.Range(lt=cutoff_date)
                    )
                ]
            )
            
            # Delete old vectors
            delete_result = await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=old_filter),
                wait=True
            )
            
            deleted_count = delete_result.operation_id if delete_result else 0
            self.logger.info(f"Cleaned up {deleted_count} old vectors")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old vectors: {str(e)}")
            raise QdrantError(
                f"Failed to cleanup old vectors: {str(e)}",
                collection_name=self.collection_name,
                original_error=e
            )
    
    async def close(self) -> None:
        """Close the Qdrant client connection"""
        try:
            if self.client:
                # Qdrant client doesn't need explicit closing
                self.client = None
                self._initialized = False
                self.logger.info("Qdrant client connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Qdrant client: {str(e)}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        if self._initialized and self.client:
            # Note: Can't use async in __del__, so we just set to None
            self.client = None
            self._initialized = False


# Factory function for easy client creation
async def create_qdrant_client(collection_name: str = None) -> QdrantVectorStore:
    """Factory function to create and initialize Qdrant client"""
    client = QdrantVectorStore(collection_name)
    await client.initialize()
    return client


# Utility functions for common operations
async def create_localization_collection(collection_name: str) -> QdrantVectorStore:
    """Create a Qdrant collection optimized for localization and content management"""
    client = QdrantVectorStore(collection_name)
    await client.initialize()
    await client.setup_localization_indexes()
    return client

if __name__ == "__main__":
    print("QdrantVectorStore - Use test_corrected_hybrid_search.py to test the implementation")