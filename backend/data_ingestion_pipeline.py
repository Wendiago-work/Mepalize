#!/usr/bin/env python3
"""
Data Ingestion Pipeline for Localized Translator
Processes documents from data folder and stores them in Qdrant with SPLADE hybrid search

This pipeline:
1. Processes documents using Docling
2. Generates dense and sparse vectors (SPLADE)
3. Stores in Qdrant with proper hybrid vector configuration
4. Integrates with LangChain RAG pipeline
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.services.data_preprocessing_service import DoclingDataPreprocessingService, ProcessingConfig
from src.core.model_manager import get_model_manager
from src.database.qdrant_client import QdrantVectorStore, VectorDocument
from src.config import get_preprocessing_config
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Complete data ingestion pipeline with SPLADE hybrid search"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.preprocessing_service = DoclingDataPreprocessingService()
        self.model_manager = None
        self.qdrant_client = None
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "vectors_stored": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def initialize(self):
        """Initialize the pipeline"""
        try:
            self.logger.info("🚀 Initializing Data Ingestion Pipeline")
            
            # Initialize model manager
            self.model_manager = get_model_manager()
            await self.model_manager.initialize()
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantVectorStore("translation_embeddings")
            await self.qdrant_client.initialize()
            
            self.logger.info("✅ Pipeline initialized successfully")
            self.logger.info(f"   Embedding model: {get_preprocessing_config().embedding_model}")
            self.logger.info(f"   Tokenizer model: {get_preprocessing_config().tokenizer_model}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    async def ingest_data_folder(self, data_folder: str = "data") -> Dict[str, Any]:
        """
        Ingest all documents from the data folder
        
        Args:
            data_folder: Path to the data folder containing documents
            
        Returns:
            Dictionary with ingestion statistics
        """
        self.stats["start_time"] = datetime.now()
        
        try:
            self.logger.info(f"📁 Starting data ingestion from: {data_folder}")
            
            # Initialize the pipeline
            await self.initialize()
            
            data_path = Path(data_folder)
            if not data_path.exists():
                raise FileNotFoundError(f"Data folder not found: {data_folder}")
            
            # Find all supported documents
            supported_files = self._find_supported_files(data_path)
            self.logger.info(f"📄 Found {len(supported_files)} supported files")
            
            if not supported_files:
                self.logger.warning("⚠️ No supported files found in data folder")
                return self.stats
            
            # Process documents in batches
            batch_size = 5  # Process 5 documents at a time
            for i in range(0, len(supported_files), batch_size):
                batch = supported_files[i:i + batch_size]
                await self._process_batch(batch)
            
            self.stats["end_time"] = datetime.now()
            self._log_final_stats()
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"❌ Data ingestion failed: {e}")
            self.stats["errors"] += 1
            raise
    
    async def ingest_specific_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Ingest specific files for translation memory and glossaries only"""
        try:
            self.stats["start_time"] = datetime.now()
            
            # Convert to Path objects and check existence
            file_paths = [Path(fp) for fp in file_paths]
            existing_files = [fp for fp in file_paths if fp.exists()]
            
            if not existing_files:
                self.logger.warning("No specified files found")
                return self.stats
            
            self.logger.info(f"Processing {len(existing_files)} specific files for Qdrant")
            
            # Process files in batches
            batch_size = 5
            for i in range(0, len(existing_files), batch_size):
                batch = existing_files[i:i + batch_size]
                await self._process_batch(batch)
            
            self.stats["end_time"] = datetime.now()
            self._log_final_stats()
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"❌ Specific file ingestion failed: {e}")
            self.stats["errors"] += 1
            raise
    
    def _find_supported_files(self, data_path: Path) -> List[Path]:
        """Find all supported files in the data folder"""
        supported_files = []
        config = get_preprocessing_config()
        
        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                file_extension = file_path.suffix.lower().lstrip('.')
                if file_extension in config.supported_formats:
                    supported_files.append(file_path)
        
        return supported_files
    
    def _get_metadata_for_file(self, file_path: Path) -> Dict[str, str]:
        """
        Get metadata for a file using the metadata configuration script.
        """
        from metadata_config import MetadataConfig
        
        # Initialize metadata config
        metadata_config = MetadataConfig()
        
        # Get metadata for this file
        return metadata_config.get_metadata_for_file(file_path)
    
    async def _process_batch(self, file_paths: List[Path]):
        """Process a batch of documents"""
        for file_path in file_paths:
            try:
                await self._process_single_document(file_path)
            except Exception as e:
                self.logger.error(f"❌ Failed to process {file_path.name}: {e}")
                self.stats["errors"] += 1
    
    async def _process_single_document(self, file_path: Path):
        """Process a single document"""
        try:
            self.logger.info(f"🔄 Processing: {file_path.name}")
            
            # Check if this file should go to Qdrant or MongoDB
            metadata = self._get_metadata_for_file(file_path)
            if metadata.get("dataset") == "mongodb_only":
                self.logger.info(f"⏭️  Skipping {file_path.name} - marked for MongoDB only")
                return
            
            # Step 1: Process document with Docling
            processed_doc = self.preprocessing_service.process_document(file_path)
            self.stats["documents_processed"] += 1
            
            # Step 2: Prepare data for Qdrant with hybrid vectors
            documents, metadatas, sparse_vectors = self.preprocessing_service.prepare_for_qdrant(processed_doc)
            self.stats["chunks_created"] += len(documents)
            
            # Step 3: Generate embeddings and create vector documents
            vector_documents = []
            for i, (doc, metadata, sparse_vector) in enumerate(zip(documents, metadatas, sparse_vectors)):
                # Generate dense embedding using model manager
                dense_embedding = self.model_manager.get_embedding(doc)
                
                # Add language and domain metadata based on file type
                enhanced_metadata = metadata.copy()
                enhanced_metadata.update(self._get_metadata_for_file(file_path))
                
                # Create vector document with hybrid vectors
                vector_doc = VectorDocument(
                    id=str(uuid4()),
                    content=doc,
                    embedding=dense_embedding,
                    metadata=enhanced_metadata,
                    sparse_vector=sparse_vector
                )
                vector_documents.append(vector_doc)
            
            # Step 4: Store in Qdrant
            if vector_documents:
                self.logger.info(f"Debug: vector_documents type: {type(vector_documents)}, length: {len(vector_documents)}")
                added_ids = await self.qdrant_client.add_vectors(vector_documents)
                self.stats["vectors_stored"] += len(added_ids)
                
                self.logger.info(f"✅ Stored {len(added_ids)} vectors for {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {file_path.name}: {e}")
            raise
    
    def _log_final_stats(self):
        """Log final ingestion statistics"""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        self.logger.info("🎉 Data Ingestion Pipeline Completed!")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Statistics:")
        self.logger.info(f"   Documents processed: {self.stats['documents_processed']}")
        self.logger.info(f"   Chunks created: {self.stats['chunks_created']}")
        self.logger.info(f"   Vectors stored: {self.stats['vectors_stored']}")
        self.logger.info(f"   Errors: {self.stats['errors']}")
        self.logger.info(f"   Duration: {duration:.2f} seconds")
        self.logger.info(f"   Rate: {self.stats['documents_processed']/duration:.2f} docs/sec")
    
    async def test_retrieval(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Test retrieval functionality after ingestion"""
        if test_queries is None:
            test_queries = [
                "dragon fire treasure",
                "inventory menu items",
                "boss level unlock",
                "save progress game",
                "princess tower rescue"
            ]
        
        self.logger.info("🔍 Testing retrieval functionality")
        
        results = {}
        for query in test_queries:
            try:
                # Generate query embedding using model manager
                query_embedding = self.model_manager.get_embedding(query)
                
                # Generate query sparse vector
                query_sparse_vector = self.preprocessing_service._generate_sparse_vector(query)
                
                # Perform hybrid search
                search_results = await self.qdrant_client.hybrid_search(
                    query_text=query,
                    query_vector=query_embedding,
                    top_k=3
                )
                
                results[query] = {
                    "results_count": len(search_results),
                    "top_result": search_results[0].content[:100] + "..." if search_results else "No results",
                    "sparse_vector_size": len(query_sparse_vector.get("indices", []))
                }
                
                self.logger.info(f"   Query: '{query}' → {len(search_results)} results")
                
            except Exception as e:
                self.logger.error(f"   Query '{query}' failed: {e}")
                results[query] = {"error": str(e)}
        
        return results
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics"""
        try:
            collection_info = await self.qdrant_client.get_collection_info()
            return {
                "collection_name": collection_info["name"],
                "points_count": collection_info["points_count"],
                "indexed_vectors_count": collection_info["indexed_vectors_count"],
                "segments_count": collection_info["segments_count"],
                "status": collection_info["status"]
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def close(self):
        """Close the pipeline"""
        if self.qdrant_client:
            await self.qdrant_client.close()
        if self.model_manager:
            await self.model_manager.close()
        self.logger.info("✅ Pipeline closed")

async def main():
    """Main function to run the data ingestion pipeline"""
    pipeline = DataIngestionPipeline()
    
    try:
        # Initialize pipeline
        await pipeline.initialize()
        
        # Ingest data from data folder
        stats = await pipeline.ingest_data_folder("data")
        
        # Test retrieval
        retrieval_results = await pipeline.test_retrieval()
        
        # Get collection stats
        collection_stats = await pipeline.get_collection_stats()
        
        print("\n🎯 Final Results:")
        print("=" * 50)
        print(f"📊 Ingestion Stats: {stats}")
        print(f"🔍 Retrieval Test: {len(retrieval_results)} queries tested")
        print(f"💾 Collection Stats: {collection_stats}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
