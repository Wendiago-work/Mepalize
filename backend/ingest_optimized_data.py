#!/usr/bin/env python3
"""
Data ingestion script for Optimized CSV files
Handles Optimized_Glossary_Music-Game_JA_v3.csv and Optimized_TM_Music-Game_JA_v3.csv
"""

import asyncio
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from src.database.chroma_client import ChromaVectorStore, TranslationMemoryEntry, GlossaryEntry
from src.config.config import get_settings


def create_translation_memory_from_optimized_tm(
    row: Dict[str, str],
    source_language: str = "en",
    target_language: str = "ja"
) -> TranslationMemoryEntry:
    """Create TranslationMemoryEntry from Optimized TM CSV row"""
    
    # Extract main fields
    source_text = row.get("Original English", "").strip()
    target_text = row.get("Optimized Japanese Translation", "").strip()
    category = row.get("Category", "").strip()
    
    if not source_text or not target_text:
        return None
    
    # Create metadata with all available fields
    metadata = {
        "source_language": source_language,
        "target_language": target_language,
        "domain": category,  # Use category as domain
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        # Additional fields from CSV
        "use_case": row.get("Use Case", "").strip(),
    }
    
    # Remove empty fields
    metadata = {k: v for k, v in metadata.items() if v}
    
    return TranslationMemoryEntry(
        id=str(uuid.uuid4()),
        source_text=source_text,
        target_text=target_text,
        metadata=metadata
    )


def create_glossary_from_optimized_glossary(
    row: Dict[str, str],
    source_language: str = "en",
    target_language: str = "ja",
    domain: str = "Game - Music"
) -> GlossaryEntry:
    """Create GlossaryEntry from Optimized Glossary CSV row"""
    
    # Extract main fields
    term = row.get("Term (English)", "").strip()
    definition = row.get("Definition / Context", "").strip()
    
    if not term or not definition:
        return None
    
    # Create metadata with all available fields
    metadata = {
        "source_language": source_language,
        "target_language": target_language,
        "domain": domain,  # Domain determined from filename
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        # Additional fields from CSV
        "full_name_japanese": row.get("Full Name (Japanese)", "").strip(),
        "common_display_japanese": row.get("Common Display (Japanese)", "").strip(),
    }
    
    # Remove empty fields
    metadata = {k: v for k, v in metadata.items() if v}
    
    return GlossaryEntry(
        id=str(uuid.uuid4()),
        term=term,
        definition=definition,
        metadata=metadata
    )


async def ingest_optimized_data():
    """Main function to ingest optimized CSV data to Chroma DB"""
    settings = get_settings()
    
    # Check if Chroma DB credentials are available
    if not all([settings.chroma_cloud_api_key, settings.chroma_cloud_tenant, settings.chroma_cloud_database]):
        print("❌ Chroma DB credentials not provided. Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE")
        return
    
    # Initialize Chroma client
    chroma_client = ChromaVectorStore(
        tm_collection_name=settings.tm_collection_name,
        glossary_collection_name=settings.glossary_collection_name
    )
    
    try:
        await chroma_client.initialize()
        print("✅ Chroma DB initialized successfully")
        
        # Data directory
        data_dir = Path("data")
        
        # Load Translation Memory from Optimized_TM_Music-Game_JA_v3.csv
        tm_file = data_dir / "Optimized_TM_Music-Game_JA_v3.csv"
        tm_entries = []
        
        if tm_file.exists():
            print(f"📄 Loading translation memory from: {tm_file.name}")
            
            with open(tm_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    entry = create_translation_memory_from_optimized_tm(row)
                    if entry:
                        tm_entries.append(entry)
            
            print(f"   Loaded {len(tm_entries)} translation memory entries")
        else:
            print(f"⚠️ Translation memory file not found: {tm_file}")
        
        # Load Glossary from Optimized_Glossary_Music-Game_JA_v3.csv
        glossary_file = data_dir / "Optimized_Glossary_Music-Game_JA_v3.csv"
        glossary_entries = []
        
        if glossary_file.exists():
            print(f"📚 Loading glossary from: {glossary_file.name}")
            
            # Domain is always "Game - Music" for this dataset
            domain = "Game - Music"
            
            with open(glossary_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    entry = create_glossary_from_optimized_glossary(row, domain=domain)
                    if entry:
                        glossary_entries.append(entry)
            
            print(f"   Loaded {len(glossary_entries)} glossary entries")
        else:
            print(f"⚠️ Glossary file not found: {glossary_file}")
        
        # Add to Chroma DB
        if tm_entries:
            print(f"\n🔄 Adding {len(tm_entries)} translation memory entries to Chroma DB...")
            await chroma_client.add_translation_memory(tm_entries)
            print("✅ Translation memory added successfully")
        
        if glossary_entries:
            print(f"\n🔄 Adding {len(glossary_entries)} glossary entries to Chroma DB...")
            await chroma_client.add_glossaries(glossary_entries)
            print("✅ Glossaries added successfully")
        
        # Get collection info
        info = await chroma_client.get_collection_info()
        print(f"\n📊 Collection Statistics:")
        print(f"   Translation Memory: {info.get('translation_memory', {}).get('count', 0)} documents")
        print(f"   Glossaries: {info.get('glossaries', {}).get('count', 0)} documents")
        
        # Show sample entries
        if tm_entries:
            print(f"\n📝 Sample Translation Memory Entry:")
            sample_tm = tm_entries[0]
            print(f"   Source: {sample_tm.source_text}")
            print(f"   Target: {sample_tm.target_text}")
            print(f"   Metadata: {sample_tm.metadata}")
        
        if glossary_entries:
            print(f"\n📝 Sample Glossary Entry:")
            sample_glossary = glossary_entries[0]
            print(f"   Term: {sample_glossary.term}")
            print(f"   Definition: {sample_glossary.definition}")
            print(f"   Metadata: {sample_glossary.metadata}")
        
    except Exception as e:
        print(f"❌ Error during data ingestion: {e}")
        raise
    finally:
        await chroma_client.close()


if __name__ == "__main__":
    asyncio.run(ingest_optimized_data())
