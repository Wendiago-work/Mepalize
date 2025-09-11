"""
Core configuration management for Translation RAG Pipeline
Handles environment-specific settings with simple environment variable loading
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
load_dotenv(env_path)


class Settings:
    """Main application settings with environment variable support"""
    
    def __init__(self):
        # Environment
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        
        # API Settings
        self.api_host: str = os.getenv("API_HOST", "localhost")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.api_version: str = os.getenv("API_VERSION", "v1")
        
        # Chroma DB Cloud Settings (Required)
        self.chroma_cloud_api_key: str = os.getenv("CHROMA_API_KEY", "")
        self.chroma_cloud_tenant: str = os.getenv("CHROMA_TENANT", "")
        self.chroma_cloud_database: str = os.getenv("CHROMA_DATABASE", "")
        self.tm_collection_name: str = os.getenv("TM_COLLECTION_NAME", "translation_memory")
        self.glossary_collection_name: str = os.getenv("GLOSSARY_COLLECTION_NAME", "glossaries")
            
        # MongoDB Atlas Settings (Required)
        self.mongo_connection_string: str = os.getenv("MONGO_CONNECTION_STRING", "")
        self.mongo_database: str = os.getenv("MONGO_DATABASE", "LocalizationDB")
                
        
        # Safety Settings - CRITICAL: Keep these enabled for user protection
        self.enable_safety_filters: bool = os.getenv("ENABLE_SAFETY_FILTERS", "true").lower() == "true"
        self.safety_threshold: str = os.getenv("SAFETY_THRESHOLD", "MEDIUM_AND_ABOVE")

        
        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_file: Optional[str] = os.getenv("LOG_FILE")
        
        # Note: No model cache directories needed - Chroma DB handles embeddings natively
        self.translation_model_cache_dir: str = os.getenv("TRANSLATION_MODEL_CACHE_DIR", "./models/translation")
        
        # Performance Settings
        self.use_gpu: bool = os.getenv("USE_GPU", "true").lower() == "true"
        self.fp16: bool = os.getenv("FP16", "true").lower() == "true"
        self.max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
        
        # Database Connection Pooling
        self.max_connections: int = int(os.getenv("MAX_CONNECTIONS", "20"))
        self.min_connections: int = int(os.getenv("MIN_CONNECTIONS", "5"))
        self.connection_timeout: int = int(os.getenv("CONNECTION_TIMEOUT", "30"))
        self.query_timeout: int = int(os.getenv("QUERY_TIMEOUT", "60"))
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay: float = float(os.getenv("RETRY_DELAY", "1.0"))
        
        # Vector Database Settings
        self.vector_index_type: str = os.getenv("VECTOR_INDEX_TYPE", "HNSW")
        self.vector_distance_metric: str = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")
        self.hnsw_m: int = int(os.getenv("HNSW_M", "16"))
        self.hnsw_ef_construct: int = int(os.getenv("HNSW_EF_CONSTRUCT", "200"))
        
        # Batch Processing
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
        self.parallel_workers: int = int(os.getenv("PARALLEL_WORKERS", "4"))
        
        # Local Models Flag
        self.use_local_models: bool = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"
    
@dataclass
class RetrievalConfig:
    """Configuration for Chroma DB collection search"""
    # Search parameters
    top_k: int = 10
    
    # Thresholds
    similarity_threshold: float = 0.7
    
    # Context parameters
    include_metadata: bool = True
    max_context_length: int = 8000
    
    def __post_init__(self):
        """Validate retrieval configuration"""
        if not 0 < self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0 and 1")

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config")
        self._settings: Optional[Settings] = None
    
    @property
    def settings(self) -> Settings:
        """Get main application settings"""
        if self._settings is None:
            self._settings = Settings()
            print(f"🔧 Loaded settings from environment")
        return self._settings
    
    
    def get_retrieval_config(self, **overrides) -> RetrievalConfig:
        """Create retrieval configuration with optional overrides"""
        config = RetrievalConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all configurations and return status"""
        validation_results = {}
        
        try:
            # Validate main settings
            settings = self.settings
            validation_results["settings"] = {"status": "valid", "message": "OK"}
            
            # Check critical environment variables
            env_checks = {}
            
            # Check Chroma DB credentials
            chroma_creds = [settings.chroma_cloud_api_key, settings.chroma_cloud_tenant, settings.chroma_cloud_database]
            if any(cred for cred in chroma_creds):
                if all(cred for cred in chroma_creds):
                    env_checks["chroma_db"] = "✅ Set"
                else:
                    env_checks["chroma_db"] = "❌ Incomplete - please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE"
            else:
                env_checks["chroma_db"] = "⚠️ Not set - using default (will fail if Chroma DB is needed)"
            
            # Note: No embedding models needed - Chroma DB handles embeddings natively
            
            validation_results["environment"] = {"status": "checked", "message": "Environment variables checked", "details": env_checks}
            
        except Exception as e:
            validation_results["settings"] = {"status": "invalid", "message": str(e)}
        
        # Note: No additional validation needed - Chroma DB handles everything natively
        
        return validation_results


# Convenience functions for common access patterns
def get_settings() -> Settings:
    """Get application settings"""
    # Ensure .env file is loaded (relative to this file)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_path)
    return Settings()


# Example usage and validation
if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager()
    
    # Validate all configurations
    validation_results = config.validate_configuration()
    print("Configuration validation results:")
    for component, result in validation_results.items():
        if isinstance(result, dict) and "status" in result:
            print(f"  {component}: {result['status']} - {result['message']}")
            if "details" in result:
                for key, value in result["details"].items():
                    print(f"    {key}: {value}")
        else:
            print(f"  {component}: {result}")
    
    # Example retrieval config
    retrieval_config = config.get_retrieval_config(top_k=10)
    print(f"Retrieval config: {retrieval_config}")
