"""
Centralized logging configuration for Translation RAG Pipeline
Provides structured logging with context and performance metrics
"""

import logging
import logging.config
import sys
import json
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from contextvars import ContextVar
from functools import wraps
import asyncio
from dataclasses import dataclass, asdict
from ..config import get_settings


# Context variables for request tracing
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_context: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


@dataclass
class LogContext:
    """Structured logging context"""
    component: str
    operation: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    language_pair: Optional[str] = None
    content_type: Optional[str] = None
    domain: Optional[str] = None
    model_name: Optional[str] = None
    processing_time: Optional[float] = None
    token_count: Optional[int] = None
    error_code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class TranslationFormatter(logging.Formatter):
    """Custom formatter for translation pipeline logs"""
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured context"""
        # Base log data
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context variables
        request_id = request_id_context.get()
        user_id = user_id_context.get()
        session_id = session_id_context.get()
        
        if request_id:
            log_data["request_id"] = request_id
        if user_id:
            log_data["user_id"] = user_id
        if session_id:
            log_data["session_id"] = session_id
        
        # Add custom context if present
        if hasattr(record, 'context') and record.context:
            if isinstance(record.context, LogContext):
                log_data["context"] = record.context.to_dict()
            else:
                log_data["context"] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add performance metrics if present
        if hasattr(record, 'duration'):
            log_data["duration_ms"] = record.duration
        
        if hasattr(record, 'memory_usage'):
            log_data["memory_mb"] = record.memory_usage
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance context to log records"""
        # Add thread/process info
        record.thread_name = record.thread if hasattr(record, 'thread') else None
        record.process_name = record.process if hasattr(record, 'process') else None
        
        return True


class TranslationLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with translation-specific context"""
    
    def __init__(self, logger: logging.Logger, context: LogContext):
        self.context = context
        super().__init__(logger, {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process log message with context"""
        # Add context to extra
        kwargs.setdefault('extra', {})
        kwargs['extra']['context'] = self.context
        return msg, kwargs
    
    def with_context(self, **context_updates) -> 'TranslationLoggerAdapter':
        """Create new adapter with updated context"""
        new_context = LogContext(
            **{**asdict(self.context), **context_updates}
        )
        return TranslationLoggerAdapter(self.logger, new_context)


class LogManager:
    """Centralized log management"""
    
    def __init__(self):
        self._configured = False
        self._loggers: Dict[str, logging.Logger] = {}
    
    def configure_logging(self, log_level: str = "INFO", log_file: Optional[str] = None,
                         enable_console: bool = True, enable_json: bool = True) -> None:
        """Configure logging for the entire application"""
        
        if self._configured:
            return
        
        # Create logs directory if needed
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if enable_json:
                console_handler.setFormatter(TranslationFormatter())
            else:
                console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                console_handler.setFormatter(logging.Formatter(console_format))
            console_handler.addFilter(PerformanceFilter())
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setFormatter(TranslationFormatter())
            file_handler.addFilter(PerformanceFilter())
            root_logger.addHandler(file_handler)
        
        # Error file handler (separate file for errors)
        if log_file:
            error_file = str(log_path).replace('.log', '_error.log')
            error_handler = logging.handlers.RotatingFileHandler(
                error_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(TranslationFormatter())
            root_logger.addHandler(error_handler)
        
        # Set specific logger levels
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("qdrant_client").setLevel(logging.INFO)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        
        self._configured = True
    
    def get_logger(self, component: str, operation: str = "general") -> TranslationLoggerAdapter:
        """Get logger with translation context"""
        logger_name = f"translation.{component}"
        
        if logger_name not in self._loggers:
            self._loggers[logger_name] = logging.getLogger(logger_name)
        
        context = LogContext(
            component=component,
            operation=operation,
            request_id=request_id_context.get(),
            user_id=user_id_context.get(),
            session_id=session_id_context.get()
        )
        
        return TranslationLoggerAdapter(self._loggers[logger_name], context)
    
    def get_performance_logger(self) -> logging.Logger:
        """Get logger specifically for performance metrics"""
        return logging.getLogger("translation.performance")
    
    def get_audit_logger(self) -> logging.Logger:
        """Get logger for audit trails"""
        return logging.getLogger("translation.audit")


# Global log manager instance
log_manager = LogManager()


# Convenience functions
def configure_logging(log_level: str = None, log_file: str = None) -> None:
    """Configure logging with settings from config"""
    settings = get_settings()
    
    log_manager.configure_logging(
        log_level=log_level or settings.log_level,
        log_file=log_file or settings.log_file,
        enable_console=True,
        enable_json=True
    )


def get_logger(component: str, operation: str = "general") -> TranslationLoggerAdapter:
    """Get logger for a component"""
    return log_manager.get_logger(component, operation)


# Context managers for request tracking
class RequestContext:
    """Context manager for request-level logging context"""
    
    def __init__(self, request_id: str, user_id: str = None, session_id: str = None):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self.tokens = {}
    
    def __enter__(self):
        self.tokens['request_id'] = request_id_context.set(self.request_id)
        if self.user_id:
            self.tokens['user_id'] = user_id_context.set(self.user_id)
        if self.session_id:
            self.tokens['session_id'] = session_id_context.set(self.session_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in self.tokens.values():
            token.var.reset(token)


# Decorators for logging
def log_execution(component: str, operation: str = None, 
                  log_args: bool = False, log_result: bool = False):
    """Decorator to log function execution"""
    def decorator(func):
        func_operation = operation or func.__name__
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(component, func_operation)
            start_time = time.time()
            
            try:
                # Log function start
                log_data = {"function": func.__name__}
                if log_args and args:
                    log_data["args"] = str(args)[:200]  # Truncate long args
                if log_args and kwargs:
                    log_data["kwargs"] = {k: str(v)[:200] for k, v in kwargs.items()}
                
                logger.info(f"Starting {func.__name__}", extra={"context": log_data})
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration = (time.time() - start_time) * 1000  # Convert to ms
                success_data = {
                    "function": func.__name__,
                    "duration_ms": round(duration, 2),
                    "success": True
                }
                
                if log_result and result is not None:
                    success_data["result_type"] = type(result).__name__
                    if hasattr(result, '__len__'):
                        success_data["result_size"] = len(result)
                
                logger.info(f"Completed {func.__name__}", extra={"context": success_data})
                return result
                
            except Exception as e:
                # Log error
                duration = (time.time() - start_time) * 1000
                error_data = {
                    "function": func.__name__,
                    "duration_ms": round(duration, 2),
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                
                logger.error(f"Failed {func.__name__}", extra={"context": error_data}, exc_info=True)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(component, func_operation)
            start_time = time.time()
            
            try:
                # Log function start
                log_data = {"function": func.__name__, "async": True}
                if log_args and args:
                    log_data["args"] = str(args)[:200]
                if log_args and kwargs:
                    log_data["kwargs"] = {k: str(v)[:200] for k, v in kwargs.items()}
                
                logger.info(f"Starting {func.__name__}", extra={"context": log_data})
                
                # Execute async function
                result = await func(*args, **kwargs)
                
                # Log success
                duration = (time.time() - start_time) * 1000
                success_data = {
                    "function": func.__name__,
                    "duration_ms": round(duration, 2),
                    "success": True,
                    "async": True
                }
                
                if log_result and result is not None:
                    success_data["result_type"] = type(result).__name__
                    if hasattr(result, '__len__'):
                        success_data["result_size"] = len(result)
                
                logger.info(f"Completed {func.__name__}", extra={"context": success_data})
                return result
                
            except Exception as e:
                # Log error
                duration = (time.time() - start_time) * 1000
                error_data = {
                    "function": func.__name__,
                    "duration_ms": round(duration, 2),
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "async": True
                }
                
                logger.error(f"Failed {func.__name__}", extra={"context": error_data}, exc_info=True)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def log_performance(component: str, operation: str = None):
    """Decorator to log performance metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            perf_logger = log_manager.get_performance_logger()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Log performance metrics
                duration = time.time() - start_time
                metrics = {
                    "component": component,
                    "operation": operation or func.__name__,
                    "function": func.__name__,
                    "duration_seconds": round(duration, 4),
                    "success": True
                }
                
                # Add result size if applicable
                if hasattr(result, '__len__'):
                    metrics["result_size"] = len(result)
                
                perf_logger.info("Performance metric", extra={"context": metrics})
                return result
                
            except Exception as e:
                # Log failed performance
                duration = time.time() - start_time
                metrics = {
                    "component": component,
                    "operation": operation or func.__name__,
                    "function": func.__name__,
                    "duration_seconds": round(duration, 4),
                    "success": False,
                    "error": str(e)
                }
                
                perf_logger.warning("Performance metric (failed)", extra={"context": metrics})
                raise
        
        return wrapper
    return decorator


# Audit logging functions
def log_translation_request(source_text: str, source_lang: str, 
                          target_lang: str, content_type: str,
                          user_id: str = None) -> None:
    """Log translation request for audit purposes"""
    audit_logger = log_manager.get_audit_logger()
    
    audit_data = {
        "event": "translation_request",
        "source_language": source_lang.value,
        "target_language": target_lang.value,
        "content_type": content_type.value,
        "text_length": len(source_text),
        "user_id": user_id or user_id_context.get(),
        "request_id": request_id_context.get(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    audit_logger.info("Translation request", extra={"context": audit_data})


def log_translation_result(translated_text: str, confidence_score: float = None,
                         processing_time: float = None, model_used: str = None) -> None:
    """Log translation result for audit purposes"""
    audit_logger = log_manager.get_audit_logger()
    
    audit_data = {
        "event": "translation_result",
        "result_length": len(translated_text),
        "confidence_score": confidence_score,
        "processing_time_ms": processing_time,
        "model_used": model_used,
        "request_id": request_id_context.get(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    audit_logger.info("Translation result", extra={"context": audit_data})


def log_user_feedback(feedback_score: int, feedback_text: str = None,
                     translation_id: str = None, user_id: str = None) -> None:
    """Log user feedback for quality monitoring"""
    audit_logger = log_manager.get_audit_logger()
    
    audit_data = {
        "event": "user_feedback",
        "feedback_score": feedback_score,
        "feedback_text": feedback_text,
        "translation_id": translation_id,
        "user_id": user_id or user_id_context.get(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    audit_logger.info("User feedback", extra={"context": audit_data})


# Testing and example usage
if __name__ == "__main__":
    import uuid
    import asyncio
    
    # Configure logging
    configure_logging(log_level="INFO")
    
    # Test basic logging
    logger = get_logger("test_component", "initialization")
    logger.info("Testing basic logging functionality")
    
    # Test context manager
    with RequestContext(str(uuid.uuid4()), "test_user", "test_session"):
        context_logger = get_logger("test_component", "context_test")
        context_logger.info("Testing with request context")
        
        # Test audit logging
        log_translation_request(
            "Hello world", 
            "en", 
            "ja", 
            "ui_text"
        )
    
    # Test decorators
    @log_execution("test_component", "sync_operation", log_args=True, log_result=True)
    def test_sync_function(text: str, multiplier: int = 1) -> str:
        time.sleep(0.1)  # Simulate processing
        return text * multiplier
    
    @log_execution("test_component", "async_operation", log_args=True)
    async def test_async_function(text: str) -> str:
        await asyncio.sleep(0.1)  # Simulate async processing
        return f"Processed: {text}"
    
    @log_performance("test_component", "performance_test")
    def performance_test_function() -> list:
        return list(range(1000))
    
    # Run tests
    async def run_tests():
        # Test sync function
        result1 = test_sync_function("test", 3)
        print(f"Sync result: {result1}")
        
        # Test async function
        result2 = await test_async_function("async test")
        print(f"Async result: {result2}")
        
        # Test performance logging
        result3 = performance_test_function()
        print(f"Performance test completed: {len(result3)} items")
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except ValueError as e:
            error_logger = get_logger("test_component", "error_test")
            error_logger.error("Test error occurred", exc_info=True)
    
    # Run the test
    asyncio.run(run_tests())
    
    print("Logging system test completed successfully!")