#!/usr/bin/env python3
"""
LangChain Orchestration Service for Localized Translator MVP
Coordinates retrieval, translation, and context management using LangChain patterns
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

# LangChain imports
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

# Local services
from .hybrid_retrieval_service import HybridRetrievalService
from .gemini_service import GeminiService
from .prompt_service import PromptService
from .context_service import ContextService, UserConfiguration

@dataclass
class TranslationRequest:
    """Request for translation with natural user context"""
    text: str
    source_language: str = "en"
    target_language: str = "ja"
    domain: str = "general"  # e.g., "music_game", "casual_game", "entertainment"
    # Natural user context (takes priority over RAG content)
    context_notes: Optional[str] = None  # User-provided context
    attachments: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []

@dataclass
class TranslationResult:
    """Result of translation pipeline"""
    request_id: str
    source_text: str
    translated_text: str
    target_language: str
    reasoning: str
    cultural_notes: str
    style_applied: str
    domain_considerations: str
    full_prompt: str = ""  # Add full_prompt field for frontend display
    rag_context: Dict[str, Any] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if self.rag_context is None:
            self.rag_context = {}

class LangChainOrchestrator:
    """LangChain-based pipeline orchestrator for translation"""
    
    def __init__(self, 
                 qdrant_service: HybridRetrievalService,
                 gemini_service: GeminiService,
                 prompt_service: PromptService,
                 context_service: ContextService):
        self.qdrant_service = qdrant_service
        self.gemini_service = gemini_service
        self.prompt_service = prompt_service
        self.context_service = context_service
        self.logger = self._setup_logger()
        
        # Initialize LangChain pipeline
        self.pipeline = self._build_pipeline()
        self._initialized = False
    
    def _setup_logger(self):
        """Setup logging for the orchestrator"""
        import logging
        from ..core.logger import get_logger
        logger = get_logger("langchain_orchestrator", "pipeline")
        logger.setLevel(logging.INFO)
        return logger
    
    def _build_pipeline(self) -> RunnableSequence:
        """Build the LangChain pipeline for translation"""
        
        # Convert dataclass to dict for LangChain
        def convert_to_dict(request: TranslationRequest) -> Dict[str, Any]:
            return {
                "text": request.text,
                "source_language": request.source_language,
                "target_language": request.target_language,
                "domain": request.domain,
                "attachments": request.attachments,
                "context_notes": request.context_notes
            }
        
        # Create wrapper functions for async methods
        async def retrieve_context_wrapper(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            return await self._retrieve_context(input_dict)
        
        async def generate_translation_wrapper(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            rag_context = input_dict.get("rag_context", {})
            return await self._generate_translation(input_dict, rag_context)
        
        def format_response_wrapper(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            rag_context = input_dict.get("rag_context", {})
            translation_result = input_dict.get("translation_result", {})
            return self._format_response(input_dict, rag_context, translation_result)
        
        # Step 1: Context Retrieval Chain
        retrieval_chain = RunnablePassthrough.assign(
            rag_context=retrieve_context_wrapper,
            search_query=self._build_search_query
        )
        
        # Step 2: Translation Generation Chain
        translation_chain = RunnablePassthrough.assign(
            translation_result=generate_translation_wrapper
        )
        
        # Step 3: Response Formatting Chain
        formatting_chain = RunnablePassthrough.assign(
            final_response=format_response_wrapper
        )
        
        # Combine all chains with conversion
        pipeline = (
            convert_to_dict
            | retrieval_chain
            | translation_chain
            | formatting_chain
        )
        
        return pipeline
    
    async def _retrieve_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context using both MongoDB and Qdrant RAG"""
        try:
            # Use the processed text (OCR already handled in main pipeline)
            processed_text = request["text"]
            
            from ..core.model_manager import get_model_manager
            model_manager = get_model_manager()
            real_embedding = model_manager.get_embedding(processed_text)
            
            # 1. Get MongoDB context (writing traits, guidelines, cultural notes)
            user_config = UserConfiguration(
                target_language=request["target_language"],
                domain=request["domain"],
                source_language=request["source_language"]
            )
            mongo_context = await self.context_service.get_context(user_config)
            
            # 2. Get Qdrant RAG context (translation memory and glossaries)
            rag_results = await self.qdrant_service.hybrid_search(
                query_embedding=real_embedding,
                query_text=processed_text,  # Use processed text that includes OCR results
                limit=10,
                source_language=request["source_language"],
                target_language=request["target_language"],
                domain=request["domain"]
            )
            
            # Filter results by dataset type
            translation_memory = [r for r in rag_results if r.get("payload", {}).get("dataset") == "translation_memory"]
            glossaries = [r for r in rag_results if r.get("payload", {}).get("dataset") == "glossaries"]
            
            # Limit results per type
            translation_memory = translation_memory[:3]
            glossaries = glossaries[:3]
            
            self.logger.info(f"🔍 Context Retrieved: {len(translation_memory)} TM, {len(glossaries)} glossaries, style guide for {mongo_context.domain}, {len(mongo_context.cultural_notes)} cultural notes")
            
            return {
                "translation_memory": translation_memory,
                "glossaries": glossaries,
                "mongo_context": mongo_context.to_dict(),
                "processed_text": processed_text  # Store processed text for translation
            }
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return {
                "translation_memory": [],
                "glossaries": [],
                "mongo_context": {
                    "style_guide": {},
                    "cultural_notes": [],
                    "domain": request.get("domain", "general"),
                    "target_language": request.get("target_language", "ja"),
                    "source_language": request.get("source_language", "en")
                }
            }
    
    def _build_search_query(self, request: Dict[str, Any]) -> str:
        """Build search query from request - let RAG do the work naturally"""
        # Use the user's natural query text - RAG will find semantically similar content
        # Only add explicit context if it's naturally part of the user's request
        query_text = request["text"]
        
        # Add context notes if provided (these are usually natural user context)
        if request.get("context_notes"):
            query_text += f" {request['context_notes']}"
        
        return query_text
    
    async def _generate_translation(self, 
                                  request: Dict[str, Any],
                                  rag_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate translation using Gemini service with combined context"""
        try:
            # Extract context components
            translation_memory = rag_context.get("translation_memory", [])
            glossaries = rag_context.get("glossaries", [])
            mongo_context = rag_context.get("mongo_context", {})
            
            # Use processed text that includes OCR results for translation
            source_text = rag_context.get("processed_text", request["text"])
            
            response = self.gemini_service.generate_translation_with_context(
                source_text=source_text,
                source_language=request["source_language"],
                target_language=request["target_language"],
                retrieved_translations=translation_memory,
                glossaries=glossaries,  # Glossaries from Qdrant
                domain_style_guide=mongo_context.get("style_guide", {}),  # Complete style guide from MongoDB
                cultural_notes=mongo_context.get("cultural_notes", []),
                domain=request.get("domain", "general"),
                attachments=request.get("attachments", []),
                context_notes=request.get("context_notes")
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Translation generation failed: {e}")
            return {
                "translated_text": f"Translation failed: {str(e)}",
                "reasoning": f"Error: {str(e)}",
                "cultural_notes": "",
                "style_applied": "",
                "domain_considerations": ""
            }
    
    def _format_response(self, 
                        request: Dict[str, Any],
                        rag_context: Dict[str, Any],
                        translation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response"""
        # Handle error cases where translation_result might not have expected fields
        translated_text = translation_result.get("translated_text", "[ERROR: Translation failed]")
        
        # Debug: Check if full_prompt is in translation_result
        print(f"🔍 DEBUG: translation_result keys: {list(translation_result.keys())}")
        print(f"🔍 DEBUG: full_prompt in translation_result: {'full_prompt' in translation_result}")
        if 'full_prompt' in translation_result:
            print(f"🔍 DEBUG: full_prompt length: {len(translation_result.get('full_prompt', ''))}")
        
        return {
            "request_id": str(uuid.uuid4()),
            "source_text": request["text"],
            "translated_text": translated_text,
            "target_language": request["target_language"],
            "reasoning": translation_result.get("reasoning", ""),
            "cultural_notes": translation_result.get("cultural_notes", ""),
            "style_applied": translation_result.get("style_applied", ""),
            "domain_considerations": translation_result.get("domain_considerations", ""),
            "full_prompt": translation_result.get("full_prompt", ""),  # Add full_prompt field for frontend display
            "rag_context": {
                "translation_memory": len(rag_context.get("translation_memory", [])),
                "glossaries": len(rag_context.get("glossaries", [])),
                "mongo_context": {
                    "style_guide": "present" if rag_context.get("mongo_context", {}).get("style_guide") else "empty",
                    "cultural_notes": len(rag_context.get("mongo_context", {}).get("cultural_notes", []))
                }
            },
            "execution_time": 0.0,  # Will be set by caller
            "metadata": {
                "pipeline": "langchain_orchestrator",
                "source_language": request["source_language"],
                "target_language": request["target_language"],
                "domain": request.get("domain", "general"),
                "context_notes": request.get("context_notes")
            }
        }
    

    
    async def execute_pipeline(self, request: TranslationRequest) -> TranslationResult:
        """Execute the complete translation pipeline using LangChain"""
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting LangChain pipeline for: {request.text[:50]}...")
            
            # Execute the pipeline
            result = await self.pipeline.ainvoke(request)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Debug: print the result structure
            print(f"🔍 Pipeline result structure: {result.keys()}")
            
            # Extract the final response from the pipeline
            final_response = result.get("final_response", {})
            final_response["execution_time"] = execution_time
            
            # Debug: Check if full_prompt is in final_response
            print(f"🔍 DEBUG: final_response keys: {list(final_response.keys())}")
            print(f"🔍 DEBUG: full_prompt in final_response: {'full_prompt' in final_response}")
            if 'full_prompt' in final_response:
                print(f"🔍 DEBUG: full_prompt length in final_response: {len(final_response.get('full_prompt', ''))}")
            
            # Keep as dictionary for logging, convert to TranslationResult at the end
            
            # 🆕 DETAILED LOGGING: Final prompt and translation result
            print(f"\n🎯 ===== DETAILED PIPELINE LOGGING =====")
            print(f"📝 Original Request:")
            print(f"   Text: '{request.text}'")
            print(f"   Source: {request.source_language} → Target: {request.target_language}")
            print(f"   Context Notes: {request.context_notes or 'None'}")
            
            print(f"\n🔍 RAG Context Retrieved:")
            rag_context = result.get("rag_context", {})
            print(f"   Translation Memory: {len(rag_context.get('translation_memory', []))}")
            print(f"   Glossaries: {len(rag_context.get('glossaries', []))}")

            
            print(f"\n🤖 Translation Result:")
            print(f"   Translated Text: '{final_response.get('translated_text', 'N/A')}'")
            print(f"   Reasoning: {final_response.get('reasoning', 'N/A')[:200]}...")
            print(f"   Cultural Notes: {final_response.get('cultural_notes', 'N/A')[:200]}...")
            print(f"   Style Applied: {final_response.get('style_applied', 'N/A')[:200]}...")
            
            print(f"\n⏱️  Performance:")
            print(f"   Execution Time: {execution_time:.3f}s")
            print(f"   Pipeline: {result.get('pipeline', 'unknown')}")
            print(f"==========================================\n")
            
            self.logger.info(f"LangChain pipeline completed in {execution_time:.3f}s")
            
            # Convert to TranslationResult at the end
            translation_result = TranslationResult(**final_response)
            return translation_result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"LangChain pipeline failed after {execution_time:.3f}s: {e}")
            raise
    
    async def batch_translate(self, requests: List[TranslationRequest]) -> List[TranslationResult]:
        """Execute batch translation using LangChain"""
        try:
            # Execute all requests in parallel
            tasks = [self.execute_pipeline(request) for request in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch translation failed for request {i}: {result}")
                    # Create error result
                    error_result = TranslationResult(
                        request_id=str(uuid.uuid4()),
                        source_text=requests[i].text,
                        translated_text=f"Translation failed: {str(result)}",
                        target_language=requests[i].target_language,
                        reasoning=f"Error: {str(result)}",
                        cultural_notes="",
                        style_applied="",
                        domain_considerations="",
                        rag_context={"translation_memory": 0, "glossaries": 0, "mongo_context": {"style_guide": "empty", "cultural_notes": 0}},
                        execution_time=0.0,
                        metadata={"error": True, "error_message": str(result)}
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            return []
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and health"""
        return {
            "orchestrator": "langchain_orchestrator",
            "initialized": self._initialized,
            "pipeline_components": [
                "retrieval_chain",
                "translation_chain", 
                "formatting_chain"
            ],
            "services": {
                "qdrant_service": "available" if self.qdrant_service else "missing",
                "gemini_service": "available" if self.gemini_service else "missing",
                "prompt_service": "available" if self.prompt_service else "missing"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

def create_langchain_orchestrator(qdrant_service: HybridRetrievalService,
                                gemini_service: GeminiService,
                                prompt_service: PromptService,
                                context_service: ContextService) -> LangChainOrchestrator:
    """Factory function to create LangChain orchestrator"""
    return LangChainOrchestrator(qdrant_service, gemini_service, prompt_service, context_service)
