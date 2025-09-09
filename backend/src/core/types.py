"""
Core type definitions for Translation RAG Pipeline
Minimal, focused types used by the current architecture
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from uuid import UUID
from pydantic import BaseModel
import numpy as np


# Base Types
DocumentId = Union[str, int, UUID]
Score = float
Embedding = List[float]
Vector = np.ndarray

# Search Types (used by Qdrant client)
class SearchType(str):
    """Types of search methods"""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


# Core Data Structure (used by Qdrant client)
@dataclass
class RetrievedDocument:
    """Document retrieved from the knowledge base"""
    id: DocumentId
    content: str
    metadata: Dict[str, Any]
    score: Score
    embedding: Optional[Embedding] = None
    retrieval_method: Optional[SearchType] = None
    
    def __post_init__(self):
        """Validate document after initialization"""
        if self.score < 0 or self.score > 1:
            raise ValueError("Score must be between 0 and 1")


# API Models (used by FastAPI endpoints)
class Attachment(BaseModel):
    """Attachment model for images, files, etc."""
    type: str  # "image", "file", "audio", etc.
    url: Optional[str] = None
    base64_data: Optional[str] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    description: Optional[str] = None


class TranslateReq(BaseModel):
    """Translation request with domain support"""
    text: str
    source_language: str = "en"
    target_language: str = "ja"
    domain: str = "general"  # e.g., "music_game", "casual_game", "entertainment"
    
    # User-provided context (takes priority over RAG content)
    context_notes: Optional[str] = None  # Additional context from user
    attachments: List[Attachment] = []  # Images, files, etc.


class RunStatus(BaseModel):
    """Run status model for background tasks"""
    runId: str
    status: str
    history: Optional[list] = None
    progress: Optional[float] = None


class RunPass(BaseModel):
    """Successful run result model"""
    status: str
    runId: str
    finalText: str
    execution_time: float


class RunFail(BaseModel):
    """Failed run result model"""
    status: str
    runId: str
    error: str
