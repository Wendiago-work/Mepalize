#!/usr/bin/env python3
"""
MongoDB Client for Localized Translator
Handles style guides and cultural notes storage and retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class StyleGuide(BaseModel):
    """Style guide for a specific domain"""
    id: Optional[str] = Field(alias="_id", default=None)
    domain: str
    style_guide: Dict[str, Any]  # Complete style guide JSON
    created_at: datetime
    updated_at: datetime

class CulturalNote(BaseModel):
    """Cultural note for a specific language and domain"""
    id: Optional[str] = Field(alias="_id", default=None)
    language: str
    domain: str
    cultural_note: Any  # Flexible to accept any content type (string, dict, list, object, etc.)
    created_at: datetime
    updated_at: datetime

class MongoDBClient:
    """MongoDB client for context storage and retrieval"""
    
    def __init__(self, connection_string: str, database_name: str = "localized_translator"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self._initialized = False
        
        # Collection references
        self.style_guides_collection: Optional[AsyncIOMotorCollection] = None
        self.cultural_notes_collection: Optional[AsyncIOMotorCollection] = None
    
    async def initialize(self) -> bool:
        """Initialize MongoDB connection and collections"""
        try:
            # Create client
            self.client = AsyncIOMotorClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ MongoDB connection established")
            
            # Get database
            self.database = self.client[self.database_name]
            
            # Get collections
            self.style_guides_collection = self.database["style_guides"]
            self.cultural_notes_collection = self.database["cultural_notes"]
            
            # Create indexes for efficient queries
            await self._create_indexes()
            
            self._initialized = True
            logger.info("✅ MongoDB collections initialized with indexes")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ MongoDB initialization failed: {e}")
            return False
    
    async def _create_indexes(self):
        """Create indexes for efficient querying"""
        try:
            # Style guides indexes
            await self.style_guides_collection.create_index("domain")
            
            # Cultural notes indexes
            await self.cultural_notes_collection.create_index("language")
            await self.cultural_notes_collection.create_index([("language", 1), ("domain", 1)])
            
            logger.info("✅ MongoDB indexes created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create MongoDB indexes: {e}")
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")
    
    async def get_collection(self, collection_name: str):
        """Get a collection by name"""
        if not self._initialized:
            await self.initialize()
        return self.database[collection_name]
    
    # Style Guides Methods
    async def get_style_guide(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get style guide for a specific domain"""
        try:
            if not self._initialized:
                raise RuntimeError("MongoDB client not initialized")
            
            doc = await self.style_guides_collection.find_one({"domain": domain})
            if doc:
                # Convert ObjectId to string for JSON serialization
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                logger.info(f"✅ Retrieved style guide for domain: {domain}")
                return doc  # Return the full document
            else:
                logger.info(f"ℹ️ No style guide found for domain: {domain}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get style guide for domain {domain}: {e}")
            return None
    
    async def add_style_guide(self, style_guide: StyleGuide) -> bool:
        """Add a new style guide"""
        try:
            if not self._initialized:
                raise RuntimeError("MongoDB client not initialized")
            
            doc = {
                "domain": style_guide.domain,
                "style_guide": style_guide.style_guide,
                "created_at": style_guide.created_at,
                "updated_at": style_guide.updated_at
            }
            
            result = await self.style_guides_collection.insert_one(doc)
            logger.info(f"✅ Added style guide for domain: {style_guide.domain}")
            return result.inserted_id is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to add style guide: {e}")
            return False
    
    # Cultural Notes Methods
    async def get_cultural_notes(self, language: str, domain: str = None, limit: int = 5) -> List[CulturalNote]:
        """Get cultural notes for a specific language and optionally domain"""
        try:
            if not self._initialized:
                raise RuntimeError("MongoDB client not initialized")
            
            # Build query filter
            query_filter = {"language": language}
            if domain:
                query_filter["domain"] = domain
            
            cursor = self.cultural_notes_collection.find(query_filter).limit(limit)
            notes = []
            
            async for doc in cursor:
                # Handle flexible cultural_note content (any object structure)
                cultural_note_content = doc.get("cultural_note", "")
                
                # Ensure we have valid content - if None or empty, use empty string
                if cultural_note_content is None:
                    cultural_note_content = ""
                
                notes.append(CulturalNote(
                    id=str(doc.get("_id")) if doc.get("_id") else None,
                    language=doc.get("language", language),
                    domain=doc.get("domain", ""),
                    cultural_note=cultural_note_content,  # Accept any content type (string, dict, list, object, etc.)
                    created_at=doc.get("created_at", datetime.utcnow()),
                    updated_at=doc.get("updated_at", datetime.utcnow())
                ))
            
            logger.info(f"✅ Retrieved {len(notes)} cultural notes for language: {language}, domain: {domain or 'all'}")
            return notes
            
        except Exception as e:
            logger.error(f"❌ Failed to get cultural notes for language {language}, domain {domain}: {e}")
            return []
    
    async def add_cultural_note(self, note: CulturalNote) -> bool:
        """Add a new cultural note"""
        try:
            if not self._initialized:
                raise RuntimeError("MongoDB client not initialized")
            
            doc = {
                "language": note.language,
                "domain": note.domain,
                "cultural_note": note.cultural_note,
                "created_at": note.created_at,
                "updated_at": note.updated_at
            }
            
            result = await self.cultural_notes_collection.insert_one(doc)
            logger.info(f"✅ Added cultural note for language: {note.language}, domain: {note.domain}")
            return result.inserted_id is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to add cultural note: {e}")
            return False
    
    # Utility Methods
    async def get_domains(self) -> List[str]:
        """Get all available domains"""
        try:
            if not self._initialized:
                logger.error("❌ MongoDB client not initialized")
                return []
            
            logger.info("🔍 Starting domain retrieval...")
            
            # Get domains from both style guides and cultural notes
            style_guide_domains = []
            cultural_note_domains = []
            
            try:
                logger.info("🔍 Querying style_guides collection for domains...")
                style_guide_domains = await self.style_guides_collection.distinct("domain")
                logger.info(f"✅ Found {len(style_guide_domains)} style guide domains: {style_guide_domains}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to get style guide domains: {e}")
                import traceback
                logger.warning(f"⚠️ Traceback: {traceback.format_exc()}")
            
            try:
                logger.info("🔍 Querying cultural_notes collection for domains...")
                cultural_note_domains = await self.cultural_notes_collection.distinct("domain")
                logger.info(f"✅ Found {len(cultural_note_domains)} cultural note domains: {cultural_note_domains}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to get cultural note domains: {e}")
                import traceback
                logger.warning(f"⚠️ Traceback: {traceback.format_exc()}")
            
            # Combine and deduplicate, filter out None values
            all_domains = []
            for domain in style_guide_domains + cultural_note_domains:
                if domain is not None and isinstance(domain, str):
                    all_domains.append(domain)
            
            unique_domains = list(set(all_domains))
            logger.info(f"✅ Final domains list: {unique_domains}")
            return sorted(unique_domains)
            
        except Exception as e:
            logger.error(f"❌ Failed to get domains: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []
    
    async def get_languages(self) -> List[str]:
        """Get all available languages"""
        try:
            if not self._initialized:
                logger.error("❌ MongoDB client not initialized")
                return []
            
            languages = await self.cultural_notes_collection.distinct("language")
            
            # Filter out None values and ensure they are strings
            valid_languages = []
            for lang in languages:
                if lang is not None and isinstance(lang, str):
                    valid_languages.append(lang)
            
            return sorted(valid_languages)
            
        except Exception as e:
            logger.error(f"❌ Failed to get languages: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            if not self._initialized:
                return {"status": "not_initialized"}
            
            stats = {
                "status": "initialized",
                "database": self.database_name,
                "collections": {
                    "style_guides": await self.style_guides_collection.count_documents({}),
                    "cultural_notes": await self.cultural_notes_collection.count_documents({})
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}
    
    async def clear_all_collections(self) -> None:
        """Clear all collections (for data refresh)"""
        try:
            if self.style_guides_collection is not None:
                await self.style_guides_collection.delete_many({})
                logger.info("✅ Cleared style_guides collection")
            
            if self.cultural_notes_collection is not None:
                await self.cultural_notes_collection.delete_many({})
                logger.info("✅ Cleared cultural_notes collection")
                
        except Exception as e:
            logger.error(f"❌ Failed to clear collections: {e}")
            raise
    

# Factory function
def create_mongodb_client(connection_string: str, database_name: str = "LocalizationDB") -> MongoDBClient:
    """Create MongoDB client instance"""
    return MongoDBClient(connection_string, database_name)
