#!/usr/bin/env python3
"""
Qdrant Data Management Utility
Provides easy-to-use commands for selective data removal from Qdrant

Usage:
    python qdrant_data_manager.py --help
    python qdrant_data_manager.py list
    python qdrant_data_manager.py delete --dataset translation_memory
    python qdrant_data_manager.py delete --domain "Game - Music"
    python qdrant_data_manager.py delete --source-language en --target-language ja
    python qdrant_data_manager.py delete --file-source "JPi - Localization Ref For AI - Translation Memory.csv"
    python qdrant_data_manager.py clear-all
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.qdrant_client import QdrantVectorStore
from src.core.logger import get_logger

logger = get_logger("qdrant_data_manager", "data_management")

class QdrantDataManager:
    """Utility class for managing Qdrant data with selective deletion capabilities"""
    
    def __init__(self):
        self.qdrant_client = None
        self.logger = logger
    
    async def initialize(self):
        """Initialize the Qdrant client"""
        try:
            self.qdrant_client = QdrantVectorStore("translation_embeddings")
            await self.qdrant_client.initialize()
            self.logger.info("✅ Connected to Qdrant")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
    
    async def list_data(self) -> Dict[str, Any]:
        """List current data in Qdrant with statistics"""
        try:
            # Get collection info
            collection_info = await self.qdrant_client.get_collection_info()
            
            # Get sample data to analyze metadata
            sample_results = await self.qdrant_client.search_vectors(
                query_vector=[0.0] * 1024,  # Dummy vector for sampling (correct dimension)
                top_k=100  # Get sample of 100 documents
            )
            
            # Analyze metadata distribution
            metadata_stats = self._analyze_metadata(sample_results)
            
            return {
                "collection_info": collection_info,
                "metadata_distribution": metadata_stats,
                "sample_size": len(sample_results)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list data: {e}")
            raise
    
    def _analyze_metadata(self, sample_results) -> Dict[str, Any]:
        """Analyze metadata distribution from sample results"""
        stats = {
            "datasets": {},
            "domains": {},
            "languages": {},
            "file_sources": {}
        }
        
        for result in sample_results:
            metadata = result.metadata
            
            # Count datasets
            dataset = metadata.get("dataset", "unknown")
            stats["datasets"][dataset] = stats["datasets"].get(dataset, 0) + 1
            
            # Count domains
            domain = metadata.get("domain", "unknown")
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
            
            # Count languages
            source_lang = metadata.get("source_language", "unknown")
            target_lang = metadata.get("target_language", "unknown")
            lang_pair = f"{source_lang}→{target_lang}"
            stats["languages"][lang_pair] = stats["languages"].get(lang_pair, 0) + 1
            
            # Count file sources
            file_source = metadata.get("file_source", "unknown")
            stats["file_sources"][file_source] = stats["file_sources"].get(file_source, 0) + 1
        
        return stats
    
    async def delete_by_dataset(self, dataset: str) -> int:
        """Delete all documents from a specific dataset"""
        self.logger.info(f"🗑️  Deleting all documents from dataset: {dataset}")
        return await self.qdrant_client.delete_by_dataset(dataset)
    
    async def delete_by_domain(self, domain: str) -> int:
        """Delete all documents from a specific domain"""
        self.logger.info(f"🗑️  Deleting all documents from domain: {domain}")
        return await self.qdrant_client.delete_by_domain(domain)
    
    async def delete_by_language(self, source_language: str = None, target_language: str = None) -> int:
        """Delete documents by language filters"""
        self.logger.info(f"🗑️  Deleting documents with language filter: {source_language}→{target_language}")
        return await self.qdrant_client.delete_by_language(source_language, target_language)
    
    async def delete_by_file_source(self, file_source: str) -> int:
        """Delete documents from a specific file source"""
        self.logger.info(f"🗑️  Deleting all documents from file: {file_source}")
        return await self.qdrant_client.delete_by_file_source(file_source)
    
    async def delete_by_custom_filter(self, filter_dict: Dict[str, Any]) -> int:
        """Delete documents by custom metadata filter"""
        self.logger.info(f"🗑️  Deleting documents with custom filter: {filter_dict}")
        return await self.qdrant_client.delete_by_metadata_filter(filter_dict)
    
    async def clear_all(self) -> None:
        """Clear all documents from the collection"""
        self.logger.warning("⚠️  CLEARING ALL DATA FROM QDRANT COLLECTION")
        await self.qdrant_client.clear_collection()
    
    async def close(self):
        """Close the Qdrant client"""
        if self.qdrant_client:
            await self.qdrant_client.close()

async def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Qdrant Data Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qdrant_data_manager.py list
  python qdrant_data_manager.py delete --dataset translation_memory
  python qdrant_data_manager.py delete --domain "Game - Music"
  python qdrant_data_manager.py delete --source-language en --target-language ja
  python qdrant_data_manager.py delete --file-source "JPi - Localization Ref For AI - Translation Memory.csv"
  python qdrant_data_manager.py clear-all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List current data in Qdrant with statistics')
    
    # Delete commands
    delete_parser = subparsers.add_parser('delete', help='Delete data by various criteria')
    delete_group = delete_parser.add_mutually_exclusive_group(required=True)
    
    delete_group.add_argument('--dataset', help='Delete by dataset (e.g., translation_memory, glossaries)')
    delete_group.add_argument('--domain', help='Delete by domain (e.g., "Game - Music", "Entertainment")')
    delete_group.add_argument('--file-source', help='Delete by file source (filename)')
    delete_group.add_argument('--source-language', help='Delete by source language (e.g., en, ja)')
    delete_group.add_argument('--target-language', help='Delete by target language (e.g., en, ja)')
    delete_group.add_argument('--custom-filter', help='Delete by custom JSON filter')
    
    # Clear all command
    subparsers.add_parser('clear-all', help='Clear all data from Qdrant collection')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize data manager
    manager = QdrantDataManager()
    
    try:
        await manager.initialize()
        
        if args.command == 'list':
            data_info = await manager.list_data()
            
            print("\n📊 Qdrant Collection Statistics")
            print("=" * 50)
            print(f"Collection: {data_info['collection_info']['name']}")
            print(f"Total Points: {data_info['collection_info']['points_count']}")
            print(f"Indexed Vectors: {data_info['collection_info']['indexed_vectors_count']}")
            print(f"Status: {data_info['collection_info']['status']}")
            
            print(f"\n📈 Metadata Distribution (from {data_info['sample_size']} sample documents)")
            print("-" * 50)
            
            print("\n🗂️  Datasets:")
            for dataset, count in data_info['metadata_distribution']['datasets'].items():
                print(f"  {dataset}: {count} documents")
            
            print("\n🏷️  Domains:")
            for domain, count in data_info['metadata_distribution']['domains'].items():
                print(f"  {domain}: {count} documents")
            
            print("\n🌍 Languages:")
            for lang_pair, count in data_info['metadata_distribution']['languages'].items():
                print(f"  {lang_pair}: {count} documents")
            
            print("\n📁 File Sources:")
            for file_source, count in data_info['metadata_distribution']['file_sources'].items():
                print(f"  {file_source}: {count} documents")
        
        elif args.command == 'delete':
            deleted_count = 0
            
            if args.dataset:
                deleted_count = await manager.delete_by_dataset(args.dataset)
            elif args.domain:
                deleted_count = await manager.delete_by_domain(args.domain)
            elif args.file_source:
                deleted_count = await manager.delete_by_file_source(args.file_source)
            elif args.source_language or args.target_language:
                deleted_count = await manager.delete_by_language(args.source_language, args.target_language)
            elif args.custom_filter:
                import json
                filter_dict = json.loads(args.custom_filter)
                deleted_count = await manager.delete_by_custom_filter(filter_dict)
            
            print(f"\n✅ Deletion completed: {deleted_count} documents removed")
        
        elif args.command == 'clear-all':
            confirm = input("⚠️  Are you sure you want to clear ALL data from Qdrant? (yes/no): ")
            if confirm.lower() == 'yes':
                await manager.clear_all()
                print("\n✅ All data cleared from Qdrant collection")
            else:
                print("❌ Operation cancelled")
    
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        sys.exit(1)
    
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(main())
