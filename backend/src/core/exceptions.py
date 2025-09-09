"""
Core exception classes for Translation RAG Pipeline
Minimal, focused exceptions used by the current architecture
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    DATABASE = "database"
    SYSTEM = "system"


class BaseTranslationError(Exception):
    """Base exception class for all translation pipeline errors"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.original_error = original_error
        
        super().__init__(self.message)
    
    def _generate_error_code(self) -> str:
        """Generate error code based on category"""
        return f"{self.category.value.upper()}_{id(self) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.category.value}: {self.message}"


# Database Errors (actually used)
class DatabaseError(BaseTranslationError):
    """Database-related errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 table_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        context.update({
            "operation": operation,
            "table_name": table_name
        })
        kwargs['context'] = context
        super().__init__(message, ErrorCategory.DATABASE, **kwargs)


class QdrantError(DatabaseError):
    """Qdrant-specific errors"""
    
    def __init__(self, message: str, collection_name: Optional[str] = None,
                 vector_id: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        context.update({
            "collection_name": collection_name,
            "vector_id": vector_id
        })
        kwargs['context'] = context
        super().__init__(message, **kwargs)