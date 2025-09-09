#!/usr/bin/env python3
"""
Context Service for Localized Translator
Retrieves style guides and cultural notes from MongoDB
based on user configuration (domain and target language)
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..database.mongodb_client import MongoDBClient, StyleGuide, CulturalNote
from ..core.logger import get_logger

logger = get_logger("context_service", "retrieval")

@dataclass
class UserConfiguration:
    """User configuration for context retrieval"""
    target_language: str
    domain: str
    source_language: str = "en"
    
    def __post_init__(self):
        if not self.target_language or not self.domain:
            raise ValueError("target_language and domain are required")

@dataclass
class ContextResult:
    """Result of context retrieval"""
    style_guide: Dict[str, Any]
    cultural_notes: List[CulturalNote]
    domain: str
    target_language: str
    source_language: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt injection"""
        return {
            "style_guide": self.style_guide,
            "cultural_notes": [
                {
                    "language": note.language,
                    "domain": note.domain,
                    "cultural_note": note.cultural_note
                }
                for note in self.cultural_notes
            ],
            "domain": self.domain,
            "target_language": self.target_language,
            "source_language": self.source_language
        }

class ContextService:
    """Service for retrieving context from MongoDB based on user configuration"""
    
    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
        self.logger = get_logger("context_service", "mongodb_operations")
    
    async def get_context(self, config: UserConfiguration) -> ContextResult:
        """Get all context for a user configuration"""
        try:
            self.logger.info(f"🔍 Retrieving context for domain: {config.domain}, target_language: {config.target_language}")
            
            # Retrieve all context types in parallel
            style_guide_task = self.get_style_guide(config.domain)
            cultural_notes_task = self.mongodb_client.get_cultural_notes(config.target_language, config.domain)
            
            # Wait for all tasks to complete
            style_guide, cultural_notes = await asyncio.gather(
                style_guide_task,
                cultural_notes_task
            )
            
            result = ContextResult(
                style_guide=style_guide,
                cultural_notes=cultural_notes,
                domain=config.domain,
                target_language=config.target_language,
                source_language=config.source_language
            )
            
            self.logger.info(f"✅ Context retrieved: style guide for {config.domain}, {len(cultural_notes)} cultural notes")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get context: {e}")
            # Return empty context on error
            return ContextResult(
                style_guide={},
                cultural_notes=[],
                domain=config.domain,
                target_language=config.target_language,
                source_language=config.source_language
            )
    
    async def get_style_guide(self, domain: str) -> Dict[str, Any]:
        """Get complete style guide for a specific domain"""
        try:
            return await self.mongodb_client.get_style_guide(domain) or {}
        except Exception as e:
            self.logger.error(f"❌ Failed to get style guide for domain {domain}: {e}")
            return {}
    
    async def get_cultural_notes(self, language: str, domain: str = None) -> List[CulturalNote]:
        """Get cultural notes for a specific language and optionally domain"""
        try:
            return await self.mongodb_client.get_cultural_notes(language, domain)
        except Exception as e:
            self.logger.error(f"❌ Failed to get cultural notes for language {language}, domain {domain}: {e}")
            return []
    
    async def get_available_domains(self) -> List[str]:
        """Get all available domains"""
        try:
            return await self.mongodb_client.get_domains()
        except Exception as e:
            self.logger.error(f"❌ Failed to get available domains: {e}")
            return []
    
    async def get_available_languages(self) -> List[str]:
        """Get all available languages"""
        try:
            return await self.mongodb_client.get_languages()
        except Exception as e:
            self.logger.error(f"❌ Failed to get available languages: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get context service statistics"""
        try:
            return await self.mongodb_client.get_stats()
        except Exception as e:
            self.logger.error(f"❌ Failed to get context service stats: {e}")
            return {"status": "error", "message": str(e)}

# Factory function
def create_context_service(mongodb_client: MongoDBClient) -> ContextService:
    """Create context service instance"""
    return ContextService(mongodb_client)
