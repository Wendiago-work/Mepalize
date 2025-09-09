#!/usr/bin/env python3
"""
Qdrant-Only Data Ingestion Utility
Ingests specific files to Qdrant with custom metadata configuration

Usage:
    python ingest_qdrant_only.py --help
    python ingest_qdrant_only.py --file "Optimized_TM_Music-Game_JA_v3.csv" --dataset translation_memory --domain "Game - Music" --source-lang en --target-lang ja
    python ingest_qdrant_only.py --file "Optimized_Glossary_Music-Game_JA_v3.csv" --dataset glossaries --domain "Game - Music" --source-lang en --target-lang ja
    python ingest_qdrant_only.py --batch music-game-en-ja
"""

import asyncio
import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.qdrant_client import QdrantVectorStore, VectorDocument
from src.core.model_manager import get_model_manager
from src.services.data_preprocessing_service import DoclingDataPreprocessingService
from src.core.logger import get_logger

logger = get_logger("qdrant_ingestion", "data_ingestion")

class QdrantOnlyIngestion:
    """Utility for ingesting specific files to Qdrant with custom metadata"""
    
    def __init__(self):
        self.qdrant_client = None
        self.model_manager = None
        self.preprocessing_service = None
        self.logger = logger
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "vectors_stored": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize the ingestion pipeline"""
        try:
            # Initialize model manager
            self.model_manager = get_model_manager()
            await self.model_manager.initialize()
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantVectorStore("translation_embeddings")
            await self.qdrant_client.initialize()
            
            # Initialize preprocessing service
            self.preprocessing_service = DoclingDataPreprocessingService()
            
            self.logger.info("✅ Qdrant-only ingestion pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    async def ingest_file(self, 
                         file_path: str, 
                         dataset: str,
                         domain: str,
                         source_language: str,
                         target_language: str,
                         custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest any file to Qdrant using Docling for processing
        
        Args:
            file_path: Path to any supported file
            dataset: Dataset type (translation_memory, glossaries, etc.)
            domain: Domain (e.g., "Game - Music", "Entertainment")
            source_language: Source language code (e.g., "en", "ja")
            target_language: Target language code (e.g., "en", "ja")
            custom_metadata: Additional custom metadata
        
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            self.logger.info(f"🔄 Ingesting {file_path} to Qdrant using Docling...")
            
            # Use Docling to process the file (handles any supported format)
            processed_doc = self.preprocessing_service.process_document(Path(file_path))
            self.stats["documents_processed"] += 1
            
            # Prepare data for Qdrant with hybrid vectors
            documents, metadatas, sparse_vectors = self.preprocessing_service.prepare_for_qdrant(processed_doc)
            self.logger.info(f"📄 Docling processed {len(documents)} chunks from {file_path}")
            
            # Create vector documents
            vector_documents = []
            for i, (doc, metadata, sparse_vector) in enumerate(zip(documents, metadatas, sparse_vectors)):
                # Generate dense embedding
                embedding = self.model_manager.get_embedding(doc)
                
                # Create enhanced metadata
                enhanced_metadata = metadata.copy()
                enhanced_metadata.update({
                    "dataset": dataset,
                    "domain": domain,
                    "source_language": source_language,
                    "target_language": target_language,
                    "file_source": Path(file_path).name,
                    "chunk_index": i,
                    "content_type": dataset
                })
                
                # Add custom metadata if provided
                if custom_metadata:
                    enhanced_metadata.update(custom_metadata)
                
                # Create vector document
                vector_doc = VectorDocument(
                    id=str(uuid4()),
                    content=doc,
                    embedding=embedding,
                    metadata=enhanced_metadata,
                    sparse_vector=sparse_vector
                )
                
                vector_documents.append(vector_doc)
            
            # Store in Qdrant
            if vector_documents:
                added_ids = await self.qdrant_client.add_vectors(vector_documents)
                self.stats["vectors_stored"] += len(added_ids)
                
                self.logger.info(f"✅ Successfully ingested {len(added_ids)} vectors from {file_path}")
                
                return {
                    "file": file_path,
                    "dataset": dataset,
                    "domain": domain,
                    "language_pair": f"{source_language}→{target_language}",
                    "chunks_created": len(documents),
                    "vectors_stored": len(added_ids),
                    "errors": self.stats["errors"]
                }
            else:
                self.logger.warning(f"⚠️ No vectors created from {file_path}")
                return {"error": "No vectors created"}
                
        except Exception as e:
            self.logger.error(f"❌ Failed to ingest {file_path}: {e}")
            self.stats["errors"] += 1
            raise
    
    
    async def ingest_batch(self, batch_name: str) -> Dict[str, Any]:
        """Ingest predefined batches of files"""
        batches = {
            "music-game-en-ja": [
                {
                    "file_path": "data/Optimized_TM_Music-Game_JA_v3.csv",
                    "dataset": "translation_memory",
                    "domain": "Game - Music",
                    "source_language": "en",
                    "target_language": "ja"
                },
                {
                    "file_path": "data/Optimized_Glossary_Music-Game_JA_v3.csv",
                    "dataset": "glossaries",
                    "domain": "Game - Music",
                    "source_language": "en",
                    "target_language": "ja"
                }
            ]
        }
        
        if batch_name not in batches:
            raise ValueError(f"Unknown batch: {batch_name}. Available batches: {list(batches.keys())}")
        
        batch_config = batches[batch_name]
        results = []
        
        for config in batch_config:
            result = await self.ingest_file(**config)
            results.append(result)
        
        return {
            "batch_name": batch_name,
            "files_processed": len(results),
            "results": results,
            "total_vectors": sum(r.get("vectors_stored", 0) for r in results),
            "total_errors": sum(r.get("errors", 0) for r in results)
        }
    
    async def close(self):
        """Close the ingestion pipeline"""
        if self.qdrant_client:
            await self.qdrant_client.close()
        if self.model_manager:
            await self.model_manager.close()
        self.logger.info("✅ Ingestion pipeline closed")

async def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Qdrant-Only Data Ingestion Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest any file (CSV, PDF, DOCX, TXT, etc.) - Docling handles it automatically
  python ingest_qdrant_only.py --file "data/Optimized_TM_Music-Game_JA_v3.csv" --dataset translation_memory --domain "Game - Music" --source-lang en --target-lang ja
  
  # Ingest predefined batch
  python ingest_qdrant_only.py --batch music-game-en-ja
  
  # Ingest with custom metadata
  python ingest_qdrant_only.py --file "data/my_document.pdf" --dataset glossaries --domain "Game - Music" --source-lang en --target-lang ja --custom-metadata '{"version": "v3", "optimized": true}'
        """
    )
    
    # File ingestion arguments
    parser.add_argument('--file', help='Path to any supported file (CSV, PDF, DOCX, TXT, etc.)')
    parser.add_argument('--dataset', choices=['translation_memory', 'glossaries'], help='Dataset type')
    parser.add_argument('--domain', help='Domain (e.g., "Game - Music")')
    parser.add_argument('--source-lang', help='Source language code (e.g., en, ja)')
    parser.add_argument('--target-lang', help='Target language code (e.g., en, ja)')
    parser.add_argument('--custom-metadata', help='Custom metadata as JSON string')
    
    # Batch ingestion
    parser.add_argument('--batch', help='Predefined batch to ingest (e.g., music-game-en-ja)')
    
    args = parser.parse_args()
    
    if not args.batch and not args.file:
        parser.print_help()
        return
    
    # Initialize ingestion pipeline
    pipeline = QdrantOnlyIngestion()
    
    try:
        await pipeline.initialize()
        
        if args.batch:
            # Ingest predefined batch
            result = await pipeline.ingest_batch(args.batch)
            
            print(f"\n🎉 Batch Ingestion Completed: {result['batch_name']}")
            print("=" * 60)
            print(f"Files processed: {result['files_processed']}")
            print(f"Total vectors stored: {result['total_vectors']}")
            print(f"Total errors: {result['total_errors']}")
            
            for file_result in result['results']:
                print(f"\n📄 {file_result['file']}")
                print(f"   Dataset: {file_result['dataset']}")
                print(f"   Domain: {file_result['domain']}")
                print(f"   Language: {file_result['language_pair']}")
                print(f"   Vectors: {file_result['vectors_stored']}")
        
        else:
            # Ingest single file
            custom_metadata = None
            if args.custom_metadata:
                import json
                custom_metadata = json.loads(args.custom_metadata)
            
            result = await pipeline.ingest_file(
                file_path=args.file,
                dataset=args.dataset,
                domain=args.domain,
                source_language=args.source_lang,
                target_language=args.target_lang,
                custom_metadata=custom_metadata
            )
            
            print(f"\n🎉 File Ingestion Completed")
            print("=" * 40)
            print(f"File: {result['file']}")
            print(f"Dataset: {result['dataset']}")
            print(f"Domain: {result['domain']}")
            print(f"Language: {result['language_pair']}")
            print(f"Rows processed: {result['rows_processed']}")
            print(f"Vectors stored: {result['vectors_stored']}")
            if result.get('errors', 0) > 0:
                print(f"Errors: {result['errors']}")
    
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        sys.exit(1)
    
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
