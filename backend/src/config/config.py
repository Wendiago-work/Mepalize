"""
Core configuration management for Translation RAG Pipeline
Handles environment-specific settings with type safety
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml


class Settings(BaseSettings):
    """Main application settings with environment variable support"""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # API Settings
    api_host: str = "localhost"
    api_port: int = 8000
    api_version: str = "v1"
    
    # Database Settings - Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "translation_embeddings"
    qdrant_api_key: Optional[str] = None
    
    # Database Settings - MongoDB
    mongo_connection_string: str = Field(default="mongodb://localhost:27017", env="TRANSLATION_MONGO_CONNECTION_STRING")
    mongo_database: str = Field(default="LocalizationDB", env="TRANSLATION_MONGO_DATABASE")
            
    # Gemini Settings
    gemini_api_key: str = Field(default="your_gemini_api_key_here", env="TRANSLATION_GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-pro", env="TRANSLATION_GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.1, env="TRANSLATION_GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=4000, env="TRANSLATION_GEMINI_MAX_TOKENS")
    
    # Safety Settings - CRITICAL: Keep these enabled for user protection
    enable_safety_filters: bool = Field(default=True, env="TRANSLATION_ENABLE_SAFETY_FILTERS")
    safety_threshold: str = Field(default="MEDIUM_AND_ABOVE", env="TRANSLATION_SAFETY_THRESHOLD")
    
    # Embedding Settings
    embedding_model_name: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024
    max_embedding_batch_size: int = 32
    
    # Sparse Vector Settings
    sparse_model_name: str = "prithvida/Splade_PP_en_v1"
    
    # Cross-encoder Settings
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_batch_size: int = 16
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Model Cache Directories
    embedding_model_cache_dir: str = "./models/embeddings"
    cross_encoder_cache_dir: str = "./models/cross_encoder"
    translation_model_cache_dir: str = "./models/translation"
    
    # Performance Settings
    use_gpu: bool = True
    fp16: bool = True
    max_sequence_length: int = 512
    
    # Database Connection Pooling
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: int = 30
    query_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Vector Database Settings
    vector_index_type: str = "HNSW"
    vector_distance_metric: str = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    
    # Batch Processing
    batch_size: int = 100
    parallel_workers: int = 4
    
    # Local Models Flag
    use_local_models: bool = False
    
    class Config:
        env_file = ".env"
        env_prefix = "TRANSLATION_"
        extra = "ignore"  # Allow extra fields but ignore them
        case_sensitive = False

    @property
    # PostgreSQL URL method removed - using Qdrant only
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@dataclass
class TranslationConfig:
    """Configuration for translation tasks"""
    source_language: str
    target_language: str
    content_type: str = "general"
    context_window: int = 3
    max_tokens: int = 2000
    temperature: float = 0.1
    enable_safety_filter: bool = True
    preserve_formatting: bool = True
    cultural_adaptation: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.source_language or not self.target_language:
            raise ValueError("Source and target languages are required")
        if self.source_language == self.target_language:
            raise ValueError("Source and target languages must be different")


@dataclass
class RetrievalConfig:
    """Configuration for retrieval components"""
    # Hybrid search parameters
    vector_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 10
    
    # Thresholds
    similarity_threshold: float = 0.7
    rerank_threshold: float = 0.8
    
    # Fusion parameters
    rrf_k: int = 60  # Reciprocal Rank Fusion parameter
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    
    # Context parameters
    include_metadata: bool = True
    max_context_length: int = 8000
    
    def __post_init__(self):
        """Validate retrieval configuration"""
        if self.vector_weight + self.bm25_weight != 1.0:
            raise ValueError("Vector and BM25 weights must sum to 1.0")
        if not 0 < self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0 and 1")


@dataclass
class LanguageConfig:
    """Language-specific configuration"""
    # Character handling
    normalize_unicode: bool = True
    
    # Cultural adaptation
    localize_names: bool = True
    adapt_cultural_references: bool = True
    
    # Text processing
    segmentation_model: Optional[str] = None
    tokenizer: Optional[str] = None

@dataclass
class DatabaseConfig:
    """Database-specific configurations"""
    # Connection pooling
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: int = 30
    
    # Query settings
    query_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Vector settings
    vector_index_type: str = "HNSW"  # Hierarchical Navigable Small World
    vector_distance_metric: str = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    
    # Batch processing
    batch_size: int = 100
    parallel_workers: int = 4


@dataclass
class ModelConfig:
    """Model-specific configurations"""
    # Model paths and versions
    embedding_model_cache_dir: str = "./models/embeddings"
    cross_encoder_cache_dir: str = "./models/cross_encoder"
    translation_model_cache_dir: str = "./models/translation"
    
    # Performance settings
    use_gpu: bool = True
    fp16: bool = True
    max_sequence_length: int = 512
    
    # Caching
    enable_model_caching: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class PreprocessingConfig:
    """Configuration for document preprocessing and chunking"""
    # Model configurations
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_provider: str = "fastembed"  # fastembed, sentence_transformers, cloud
    sparse_model: str = "prithvida/Splade_PP_en_v1"  # SPLADE model for sparse vectors
    
    def __post_init__(self):
        """Initialize with environment variables if available"""
        import os
        # Allow environment variable override
        if os.getenv("TRANSLATION_SPARSE_MODEL_NAME"):
            self.sparse_model = os.getenv("TRANSLATION_SPARSE_MODEL_NAME")
    
    @property
    def tokenizer_model(self) -> str:
        """Tokenizer model is always inferred from embedding model for perfect alignment"""
        return self.embedding_model
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    use_docling_chunking: bool = True
    preserve_tables: bool = True
    min_text_length: int = 3
    quality_threshold: float = 0.3
    
    # Text cleaning settings
    collapse_whitespace: bool = True
    normalize_quotes_dashes: bool = True
    strip_control_chars: bool = True
    preserve_diacritics: bool = True
    force_lowercase: bool = False
    
    # Supported formats (only formats supported by Docling)
    supported_formats: List[str] = field(default_factory=lambda: [
        "pdf", "docx", "md", "html", "csv", "xlsx", "pptx", "asciidoc", "xml_uspto", "xml_jats", "mets_gbs", "json_docling", "audio", "image"
    ])
    
    # Docling pipeline options
    enable_ocr: bool = True
    enable_table_extraction: bool = True
    enable_figure_extraction: bool = True
    enable_formula_extraction: bool = True
    
    def __post_init__(self):
        """Validate preprocessing configuration"""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError("Quality threshold must be between 0 and 1")


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config")
        self._settings: Optional[Settings] = None
        self._language_config: Optional[LanguageConfig] = None
        self._database_config: Optional[DatabaseConfig] = None
        self._model_config: Optional[ModelConfig] = None
        self._preprocessing_config: Optional[PreprocessingConfig] = None
    
    @property
    def settings(self) -> Settings:
        """Get main application settings"""
        if self._settings is None:
            self._settings = Settings()
            print(f"🔧 Loaded settings from environment")
            print(f"   Gemini Model: {self._settings.gemini_model}")
            print(f"   Gemini API Key: {'✅ Set' if self._settings.gemini_api_key != 'your_gemini_api_key_here' else '❌ Not set'}")
        return self._settings
    
    @property
    def language_config(self) -> LanguageConfig:
        """Get language-specific configuration"""
        if self._language_config is None:
            config_file = self.config_path / "language_config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                self._language_config = LanguageConfig(**config_data)
            else:
                self._language_config = LanguageConfig()
        return self._language_config
        
    @property
    def database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        if self._database_config is None:
            self._database_config = DatabaseConfig()
        return self._database_config
    
    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration"""
        if self._model_config is None:
            self._model_config = ModelConfig()
        return self._model_config
    
    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        """Get preprocessing configuration"""
        if self._preprocessing_config is None:
            self._preprocessing_config = PreprocessingConfig()
        return self._preprocessing_config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.database_config
    
    def get_translation_config(self, source_lang: str, target_lang: str, 
                             content_type: str = "general") -> TranslationConfig:
        """Create translation configuration"""
        return TranslationConfig(
            source_language=source_lang,
            target_language=target_lang,
            content_type=content_type,
            max_tokens=self.settings.gemini_max_tokens,
            temperature=self.settings.gemini_temperature
        )
    
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
            if settings.gemini_api_key == "your_gemini_api_key_here":
                env_checks["gemini_api_key"] = "❌ Not set - please set TRANSLATION_GEMINI_API_KEY"
            else:
                env_checks["gemini_api_key"] = "✅ Set"
            
            # Check model configurations
            preprocessing_config = self.preprocessing_config
            env_checks["embedding_model"] = f"✅ {preprocessing_config.embedding_model}"
            env_checks["sparse_model"] = f"✅ {preprocessing_config.sparse_model}"
            
            validation_results["environment"] = env_checks
            
        except Exception as e:
            validation_results["settings"] = {"status": "invalid", "message": str(e)}
        
        try:
            # Validate Language config
            lang_config = self.language_config
            validation_results["language_config"] = {"status": "valid", "message": "OK"}
        except Exception as e:
            validation_results["language_config"] = {"status": "invalid", "message": str(e)}
        
        # Check required directories
        required_dirs = [
            self.model_config.embedding_model_cache_dir,
            self.model_config.cross_encoder_cache_dir,
            self.model_config.translation_model_cache_dir
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        
        validation_results["directories"] = {"status": "valid", "message": "All required directories created"}
        
        return validation_results


# Convenience functions for common access patterns
def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


def get_language_config() -> LanguageConfig:
    """Get language configuration"""
    return LanguageConfig()

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return DatabaseConfig()


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return ModelConfig()


def get_preprocessing_config() -> PreprocessingConfig:
    """Get preprocessing configuration"""
    return PreprocessingConfig()


# Example usage and validation
if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager()
    
    # Validate all configurations
    validation_results = config.validate_configuration()
    print("Configuration validation results:")
    for component, result in validation_results.items():
        print(f"  {component}: {result['status']} - {result['message']}")
    
    # Example translation config for English to Japanese
    translation_config = config.get_translation_config("en", "ja", "ui_text")
    print(f"\nTranslation config: {translation_config}")
    
    # Example retrieval config
    retrieval_config = config.get_retrieval_config(vector_top_k=15, final_top_k=8)
    print(f"Retrieval config: {retrieval_config}")
