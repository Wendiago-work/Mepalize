#!/usr/bin/env python3
"""
Data Ingestion Script with Metadata Configuration

This script runs the data ingestion pipeline using the metadata configuration
to properly categorize files for Qdrant vs MongoDB.

Usage:
    python run_data_ingestion.py
"""

import asyncio
import logging
from pathlib import Path
from data_ingestion_pipeline import DataIngestionPipeline
from metadata_config import MetadataConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run data ingestion"""
    try:
        logger.info("🚀 Starting Data Ingestion with Metadata Configuration")
        logger.info("=" * 60)
        
        # Initialize metadata config and show what will be processed
        metadata_config = MetadataConfig()
        data_dir = Path("data")
        
        if not data_dir.exists():
            logger.error("❌ Data directory not found. Please run from backend directory.")
            return
        
        # Show configuration summary
        logger.info("📋 Configuration Summary:")
        file_metadata = metadata_config.list_configured_files(data_dir)
        
        qdrant_files = []
        mongodb_files = []
        
        for file_path, metadata in file_metadata.items():
            if metadata["dataset"] == "mongodb_only":
                mongodb_files.append(Path(file_path).name)
            else:
                qdrant_files.append(Path(file_path).name)
        
        logger.info(f"📊 Files for Qdrant: {len(qdrant_files)}")
        for file_name in qdrant_files:
            logger.info(f"   📄 {file_name}")
        
        logger.info(f"📊 Files for MongoDB: {len(mongodb_files)}")
        for file_name in mongodb_files:
            logger.info(f"   📄 {file_name}")
        
        logger.info("=" * 60)
        
        # Initialize and run ingestion pipeline
        pipeline = DataIngestionPipeline()
        
        # Run ingestion (this will skip MongoDB-only files automatically)
        await pipeline.ingest_data_folder("data")
        
        logger.info("🎉 Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
