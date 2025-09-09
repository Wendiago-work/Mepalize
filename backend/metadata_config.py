"""
Metadata Configuration Script for Data Ingestion

This script allows you to define metadata (dataset, domain, language pairs) 
for each file before ingesting data into Qdrant.

Usage:
1. Define your file patterns and metadata in the FILE_METADATA_CONFIG below
2. Run the ingestion pipeline - it will use this configuration
3. Modify the config as needed for different datasets
"""

from typing import Dict, List, Optional
from pathlib import Path
import re

# Configuration for file metadata
FILE_METADATA_CONFIG = {
    # Translation Memory files
    "translation_memory": {
        "patterns": [
            r".*translation.*memory.*\.csv$",
            r".*jpi.*localization.*ref.*ai.*translation.*memory.*\.csv$"
        ],
        "metadata": {
            "dataset": "translation_memory",
            "domain": "Game - Music",
            "source_language": "en",
            "target_language": "ja"
        }
    },
    
    # Glossary files by domain and language
    "glossaries_game_music": {
        "patterns": [
            r".*glossary.*game.*music.*en.*\.csv$",
            r".*glossary.*game.*music.*english.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Game - Music", 
            "source_language": "en",
            "target_language": "en"
        }
    },
    
    "glossaries_game_music_fr": {
        "patterns": [
            r".*glossary.*game.*music.*fr.*\.csv$",
            r".*glossary.*game.*music.*french.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Game - Music",
            "source_language": "en", 
            "target_language": "fr"
        }
    },
    
    "glossaries_game_music_ja": {
        "patterns": [
            r".*glossary.*game.*music.*ja.*\.csv$",
            r".*glossary.*game.*music.*japanese.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Game - Music",
            "source_language": "en",
            "target_language": "ja"
        }
    },
    
    "glossaries_game_casual": {
        "patterns": [
            r".*glossary.*game.*casual.*en.*\.csv$",
            r".*glossary.*game.*casual.*english.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Game - Casual",
            "source_language": "en",
            "target_language": "en"
        }
    },
    
    "glossaries_game_casual_fr": {
        "patterns": [
            r".*glossary.*game.*casual.*fr.*\.csv$",
            r".*glossary.*game.*casual.*french.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Game - Casual",
            "source_language": "en",
            "target_language": "fr"
        }
    },
    
    "glossaries_game_casual_ja": {
        "patterns": [
            r".*glossary.*game.*casual.*ja.*\.csv$",
            r".*glossary.*game.*casual.*japanese.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Game - Casual",
            "source_language": "en",
            "target_language": "ja"
        }
    },
    
    "glossaries_entertainment": {
        "patterns": [
            r".*glossary.*entertainment.*en.*\.csv$",
            r".*glossary.*entertainment.*english.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Entertainment",
            "source_language": "en",
            "target_language": "en"
        }
    },
    
    "glossaries_entertainment_fr": {
        "patterns": [
            r".*glossary.*entertainment.*fr.*\.csv$",
            r".*glossary.*entertainment.*french.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Entertainment",
            "source_language": "en",
            "target_language": "fr"
        }
    },
    
    "glossaries_entertainment_ja": {
        "patterns": [
            r".*glossary.*entertainment.*ja.*\.csv$",
            r".*glossary.*entertainment.*japanese.*\.csv$"
        ],
        "metadata": {
            "dataset": "glossaries",
            "domain": "Entertainment",
            "source_language": "en",
            "target_language": "ja"
        }
    },
    
    # French reference files (these should NOT go to Qdrant - they go to MongoDB)
    "french_reference_files": {
        "patterns": [
            r".*fri.*localization.*ref.*ai.*\.csv$",
            r".*french.*reference.*\.csv$",
            r".*fri.*localization.*ref.*ai.*all.*ref.*fr.*\.csv$",
            r".*fri.*localization.*ref.*ai.*style.*guide.*\.csv$",
            r".*fri.*localization.*ref.*ai.*writing.*traits.*\.csv$",
            r".*fri.*localization.*ref.*ai.*mcr.*\.csv$"
        ],
        "metadata": {
            "dataset": "mongodb_only",  # Special marker to exclude from Qdrant
            "domain": "general",
            "source_language": "en",
            "target_language": "fr"
        }
    },
    
    # Cultural notes (these should NOT go to Qdrant - they go to MongoDB)
    "cultural_notes": {
        "patterns": [
            r".*cultural.*notes.*\.csv$",
            r".*cultural.*note.*\.csv$"
        ],
        "metadata": {
            "dataset": "mongodb_only",  # Special marker to exclude from Qdrant
            "domain": "general",
            "source_language": "en",
            "target_language": "en"
        }
    }
}

# Default metadata for unmatched files
DEFAULT_METADATA = {
    "dataset": "general",
    "domain": "general", 
    "source_language": "en",
    "target_language": "en"
}

class MetadataConfig:
    """Manages metadata configuration for data ingestion"""
    
    def __init__(self, config: Dict = None):
        self.config = config or FILE_METADATA_CONFIG
        self.default_metadata = DEFAULT_METADATA
    
    def get_metadata_for_file(self, file_path: Path) -> Dict[str, str]:
        """
        Get metadata for a file based on filename patterns.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with metadata fields
        """
        file_name = file_path.name.lower()
        
        # Check each configuration entry
        for config_name, config_data in self.config.items():
            patterns = config_data["patterns"]
            metadata = config_data["metadata"]
            
            # Check if filename matches any pattern
            for pattern in patterns:
                if re.match(pattern, file_name, re.IGNORECASE):
                    return metadata.copy()
        
        # Return default metadata if no match found
        return self.default_metadata.copy()
    
    def add_config(self, name: str, patterns: List[str], metadata: Dict[str, str]):
        """Add a new configuration entry"""
        self.config[name] = {
            "patterns": patterns,
            "metadata": metadata
        }
    
    def list_configured_files(self, data_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        List all files in data directory with their configured metadata.
        Useful for verification before ingestion.
        """
        results = {}
        
        for file_path in data_dir.rglob("*.csv"):
            metadata = self.get_metadata_for_file(file_path)
            results[str(file_path)] = metadata
            
        return results
    
    def validate_config(self, data_dir: Path) -> Dict[str, List[str]]:
        """
        Validate configuration against actual files.
        Returns files that don't match any pattern.
        """
        unmatched_files = []
        matched_files = []
        
        for file_path in data_dir.rglob("*.csv"):
            file_name = file_path.name.lower()
            matched = False
            
            for config_name, config_data in self.config.items():
                patterns = config_data["patterns"]
                for pattern in patterns:
                    if re.match(pattern, file_name, re.IGNORECASE):
                        matched = True
                        matched_files.append(str(file_path))
                        break
                if matched:
                    break
            
            if not matched:
                unmatched_files.append(str(file_path))
        
        return {
            "matched": matched_files,
            "unmatched": unmatched_files
        }

# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize metadata config
    metadata_config = MetadataConfig()
    
    # Test with actual data directory
    data_dir = Path("data")
    
    if data_dir.exists():
        print("🔍 File Metadata Configuration:")
        print("=" * 50)
        
        # List all files with their metadata
        file_metadata = metadata_config.list_configured_files(data_dir)
        for file_path, metadata in file_metadata.items():
            print(f"📄 {Path(file_path).name}")
            print(f"   Dataset: {metadata['dataset']}")
            print(f"   Domain: {metadata['domain']}")
            print(f"   Language Pair: {metadata['source_language']} → {metadata['target_language']}")
            print()
        
        # Validate configuration
        validation = metadata_config.validate_config(data_dir)
        if validation["unmatched"]:
            print("⚠️  Unmatched files (will use default metadata):")
            for file_path in validation["unmatched"]:
                print(f"   {Path(file_path).name}")
        else:
            print("✅ All files matched with configuration!")
    else:
        print("❌ Data directory not found. Please run from backend directory.")
