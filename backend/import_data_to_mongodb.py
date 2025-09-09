#!/usr/bin/env python3
"""
Data Import Script for MongoDB
Imports writing traits, user guidelines, and cultural notes from CSV files into MongoDB
"""

import asyncio
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.mongodb_client import MongoDBClient, WritingTrait, UserGuideline, CulturalNote
from src.config.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataImporter:
    """Handles importing data from CSV files to MongoDB"""
    
    def __init__(self, mongodb_client: MongoDBClient):
        self.mongodb_client = mongodb_client
        
    async def import_writing_traits(self, csv_file_path: str) -> int:
        """Import writing traits from CSV file"""
        logger.info(f"📝 Importing writing traits from {csv_file_path}")
        
        traits = []
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                if not row.get('Trait') or not row.get('Category'):
                    continue
                    
                # Extract domain from category
                category = row['Category'].strip()
                if 'Game - Music' in category:
                    domain = 'music_game'
                elif 'Game - Casual' in category:
                    domain = 'casual_game'
                elif 'Entertainment' in category:
                    domain = 'entertainment'
                else:
                    domain = 'general'
                
                # Determine priority based on trait importance
                trait_text = row['Trait'].strip()
                priority = 1  # High priority by default
                if 'optional' in trait_text.lower() or 'sometimes' in trait_text.lower():
                    priority = 3  # Low priority
                elif 'important' in trait_text.lower() or 'essential' in trait_text.lower():
                    priority = 1  # High priority
                else:
                    priority = 2  # Medium priority
                
                trait = WritingTrait(
                    name=trait_text,
                    description=f"Example: {row.get('Example', 'N/A')} | Vibe: {row.get('Vibe', 'N/A')}",
                    domain=domain,
                    priority=priority,
                    tags=[category.strip()]
                )
                traits.append(trait)
        
        # Insert into MongoDB
        collection = await self.mongodb_client.get_collection("writing_traits")
        if traits:
            # Convert to dict for insertion, excluding None values
            trait_docs = [trait.model_dump(by_alias=True, exclude_none=True) for trait in traits]
            result = await collection.insert_many(trait_docs)
            logger.info(f"✅ Imported {len(result.inserted_ids)} writing traits")
            return len(result.inserted_ids)
        else:
            logger.warning("⚠️ No writing traits found to import")
            return 0
    
    async def create_domain_aware_writing_traits(self) -> int:
        """Create domain-aware writing traits for different industries"""
        logger.info("📝 Creating domain-aware writing traits for multiple domains")
        
        traits = []
        
        # Define writing traits for different domains
        domain_traits = {
            "gaming": [
                {
                    "name": "Action-driven verbs",
                    "description": "Use energetic, directive verbs that create excitement and urgency. Examples: TAP, SCAN, BUILD, CLIMB, CONQUER",
                    "priority": 1,
                    "tags": ["action", "energy", "gaming"]
                },
                {
                    "name": "Hype claims and rankings",
                    "description": "Use bold, promotional language with superlatives and achievement-focused messaging. Examples: TOP 1, UNBELIEVABLE, 93% positive reviews",
                    "priority": 1,
                    "tags": ["marketing", "hype", "gaming"]
                },
                {
                    "name": "Short and punchy phrases",
                    "description": "Keep text concise and mobile-optimized. Use fragments and action words. Examples: SWING, EXPLORE, COLLECT",
                    "priority": 2,
                    "tags": ["concise", "mobile", "gaming"]
                },
                {
                    "name": "Progress and competition",
                    "description": "Emphasize advancement, achievement, and competitive elements. Examples: CLIMB RANK, COMPETE & WIN, CONQUÉRIR LE CLASSEMENT",
                    "priority": 2,
                    "tags": ["progress", "competition", "gaming"]
                }
            ],
            "software": [
                {
                    "name": "Clear and precise language",
                    "description": "Use unambiguous, technical language that leaves no room for misinterpretation. Avoid jargon when possible.",
                    "priority": 1,
                    "tags": ["clarity", "precision", "software"]
                },
                {
                    "name": "User-friendly terminology",
                    "description": "Use familiar terms and avoid overly technical language. Make complex concepts accessible to all users.",
                    "priority": 1,
                    "tags": ["accessibility", "user_friendly", "software"]
                },
                {
                    "name": "Consistent terminology",
                    "description": "Maintain consistent terminology across all interface elements and documentation. Use the same term for the same concept.",
                    "priority": 1,
                    "tags": ["consistency", "terminology", "software"]
                },
                {
                    "name": "Action-oriented instructions",
                    "description": "Use clear, actionable language for user instructions. Start with verbs and be specific about what users should do.",
                    "priority": 2,
                    "tags": ["instructions", "actions", "software"]
                }
            ],
            "healthcare": [
                {
                    "name": "Professional and respectful tone",
                    "description": "Use formal, respectful language appropriate for medical contexts. Maintain dignity and professionalism.",
                    "priority": 1,
                    "tags": ["professional", "respectful", "healthcare"]
                },
                {
                    "name": "Clear and accessible language",
                    "description": "Use simple, clear language that patients can understand. Avoid medical jargon when possible.",
                    "priority": 1,
                    "tags": ["clarity", "accessibility", "healthcare"]
                },
                {
                    "name": "Safety-focused messaging",
                    "description": "Prioritize patient safety in all communications. Use clear warnings and emphasize important safety information.",
                    "priority": 1,
                    "tags": ["safety", "patient_care", "healthcare"]
                },
                {
                    "name": "Empathetic and supportive",
                    "description": "Use compassionate language that shows understanding and support for patients and their families.",
                    "priority": 2,
                    "tags": ["empathy", "support", "healthcare"]
                }
            ],
            "ecommerce": [
                {
                    "name": "Trust-building language",
                    "description": "Use language that builds confidence and trust. Emphasize security, quality, and customer satisfaction.",
                    "priority": 1,
                    "tags": ["trust", "confidence", "ecommerce"]
                },
                {
                    "name": "Persuasive but honest",
                    "description": "Use compelling language that encourages purchases while maintaining honesty and transparency.",
                    "priority": 1,
                    "tags": ["persuasive", "honest", "ecommerce"]
                },
                {
                    "name": "Clear product information",
                    "description": "Provide clear, detailed information about products. Use specific details and avoid vague descriptions.",
                    "priority": 1,
                    "tags": ["product_info", "clarity", "ecommerce"]
                },
                {
                    "name": "Urgency and scarcity",
                    "description": "Use appropriate urgency language to encourage action. Examples: Limited time, While supplies last, Only 3 left",
                    "priority": 2,
                    "tags": ["urgency", "scarcity", "ecommerce"]
                }
            ],
            "legal": [
                {
                    "name": "Precise and unambiguous",
                    "description": "Use exact, unambiguous language that leaves no room for misinterpretation. Every word matters.",
                    "priority": 1,
                    "tags": ["precision", "unambiguous", "legal"]
                },
                {
                    "name": "Formal and professional",
                    "description": "Use formal, professional language appropriate for legal documents. Maintain dignity and authority.",
                    "priority": 1,
                    "tags": ["formal", "professional", "legal"]
                },
                {
                    "name": "Comprehensive coverage",
                    "description": "Cover all necessary legal bases and potential scenarios. Leave no gaps in legal protection.",
                    "priority": 1,
                    "tags": ["comprehensive", "coverage", "legal"]
                },
                {
                    "name": "Clear structure and organization",
                    "description": "Use clear headings, numbering, and logical organization to make legal documents navigable.",
                    "priority": 2,
                    "tags": ["structure", "organization", "legal"]
                }
            ]
        }
        
        # Create writing traits for each domain
        for domain, trait_list in domain_traits.items():
            for trait_data in trait_list:
                trait = WritingTrait(
                    name=trait_data["name"],
                    description=trait_data["description"],
                    domain=domain,
                    priority=trait_data["priority"],
                    tags=trait_data["tags"]
                )
                traits.append(trait)
        
        # Insert into MongoDB
        collection = await self.mongodb_client.get_collection("writing_traits")
        if traits:
            trait_docs = [trait.model_dump(by_alias=True, exclude_none=True) for trait in traits]
            result = await collection.insert_many(trait_docs)
            logger.info(f"✅ Created {len(result.inserted_ids)} domain-aware writing traits across {len(domain_traits)} domains")
            return len(result.inserted_ids)
        else:
            logger.warning("⚠️ No writing traits created")
            return 0
    
    async def import_user_guidelines(self, csv_file_path: str) -> int:
        """Import user guidelines from style guide CSV file"""
        logger.info(f"📋 Importing user guidelines from {csv_file_path}")
        
        guidelines = []
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                if not row.get('Aspect') or not row.get('Category'):
                    continue
                    
                # Extract domain from category
                category = row['Category'].strip()
                if 'Game - Music' in category:
                    domain = 'music_game'
                elif 'Game - Casual' in category:
                    domain = 'casual_game'
                elif 'Entertainment' in category:
                    domain = 'entertainment'
                else:
                    domain = 'general'
                
                aspect = row['Aspect'].strip()
                guideline_content = row.get('Guideline', '').strip()
                
                if not guideline_content:
                    continue
                
                # Determine priority based on aspect type
                priority = 2  # Medium priority by default
                if 'Tone of Voice' in aspect or 'Best Practices' in aspect:
                    priority = 1  # High priority
                elif 'Language Traits' in aspect or 'Vocabulary' in aspect:
                    priority = 1  # High priority
                elif 'Structure' in aspect:
                    priority = 2  # Medium priority
                else:
                    priority = 3  # Low priority
                
                guideline = UserGuideline(
                    name=aspect,
                    content=guideline_content,
                    domain=domain,
                    priority=priority,
                    tags=[category.strip(), aspect.strip()]
                )
                guidelines.append(guideline)
        
        # Insert into MongoDB
        collection = await self.mongodb_client.get_collection("user_guidelines")
        if guidelines:
            guideline_docs = [guideline.model_dump(by_alias=True, exclude_none=True) for guideline in guidelines]
            result = await collection.insert_many(guideline_docs)
            logger.info(f"✅ Imported {len(result.inserted_ids)} user guidelines")
            return len(result.inserted_ids)
        else:
            logger.warning("⚠️ No user guidelines found to import")
            return 0
    
    async def create_domain_aware_user_guidelines(self) -> int:
        """Create domain-aware user guidelines for different industries"""
        logger.info("📋 Creating domain-aware user guidelines for multiple domains")
        
        guidelines = []
        
        # Define user guidelines for different domains
        domain_guidelines = {
            "gaming": [
                {
                    "name": "Tone of Voice",
                    "content": "High-energy, competitive, fan-driven. Feels urgent, bold, and genre-savvy. Use caps & exclamatory tone for excitement.",
                    "priority": 1,
                    "tags": ["tone", "energy", "gaming"]
                },
                {
                    "name": "Language Traits",
                    "content": "Use informal, spoken-like phrasing. Favor exclamations and positive reinforcement. Use 2nd person direct address (invite tes potes).",
                    "priority": 1,
                    "tags": ["language", "informal", "gaming"]
                },
                {
                    "name": "Vocabulary",
                    "content": "Use action verbs: TAP, SCAN, BUILD, CLIMB. Social phrases: potes, rire, surprise. Fun adjectives: amusant, relaxant, fou.",
                    "priority": 1,
                    "tags": ["vocabulary", "action_verbs", "gaming"]
                },
                {
                    "name": "Structure",
                    "content": "Often 3-part rhythm: Action + Reward + Social twist. Use ellipses or exclamations. Short sentences or catchy fragments.",
                    "priority": 2,
                    "tags": ["structure", "rhythm", "gaming"]
                }
            ],
            "software": [
                {
                    "name": "Tone of Voice",
                    "content": "Professional, clear, and helpful. Be concise but comprehensive. Use active voice and present tense for instructions.",
                    "priority": 1,
                    "tags": ["tone", "professional", "software"]
                },
                {
                    "name": "Language Traits",
                    "content": "Use clear, direct language. Avoid jargon when possible. Use consistent terminology throughout the interface.",
                    "priority": 1,
                    "tags": ["language", "clarity", "software"]
                },
                {
                    "name": "Vocabulary",
                    "content": "Use familiar terms: Click, Select, Save, Cancel. Avoid technical jargon in user-facing text. Use action verbs for buttons.",
                    "priority": 1,
                    "tags": ["vocabulary", "familiar_terms", "software"]
                },
                {
                    "name": "Structure",
                    "content": "Use numbered lists for steps. Group related information. Use headings and subheadings for organization.",
                    "priority": 2,
                    "tags": ["structure", "organization", "software"]
                }
            ],
            "healthcare": [
                {
                    "name": "Tone of Voice",
                    "content": "Professional, empathetic, and reassuring. Use warm but professional language. Show understanding and support.",
                    "priority": 1,
                    "tags": ["tone", "professional", "healthcare"]
                },
                {
                    "name": "Language Traits",
                    "content": "Use clear, simple language that patients can understand. Avoid medical jargon when possible. Use second person (you) for directness.",
                    "priority": 1,
                    "tags": ["language", "clarity", "healthcare"]
                },
                {
                    "name": "Vocabulary",
                    "content": "Use familiar medical terms. Explain complex concepts simply. Use positive language when possible.",
                    "priority": 1,
                    "tags": ["vocabulary", "medical_terms", "healthcare"]
                },
                {
                    "name": "Structure",
                    "content": "Use clear headings and bullet points. Group related information. Use step-by-step instructions for procedures.",
                    "priority": 2,
                    "tags": ["structure", "clarity", "healthcare"]
                }
            ],
            "ecommerce": [
                {
                    "name": "Tone of Voice",
                    "content": "Friendly, trustworthy, and persuasive. Build confidence while encouraging purchases. Use positive, engaging language.",
                    "priority": 1,
                    "tags": ["tone", "friendly", "ecommerce"]
                },
                {
                    "name": "Language Traits",
                    "content": "Use clear, honest language. Be specific about products and services. Use active voice and present tense.",
                    "priority": 1,
                    "tags": ["language", "honest", "ecommerce"]
                },
                {
                    "name": "Vocabulary",
                    "content": "Use product-specific terms. Include size, color, material details. Use action words: Buy, Add to Cart, Checkout.",
                    "priority": 1,
                    "tags": ["vocabulary", "product_terms", "ecommerce"]
                },
                {
                    "name": "Structure",
                    "content": "Use clear product descriptions. Group features and benefits. Use bullet points for specifications.",
                    "priority": 2,
                    "tags": ["structure", "product_info", "ecommerce"]
                }
            ],
            "legal": [
                {
                    "name": "Tone of Voice",
                    "content": "Formal, precise, and authoritative. Use professional language appropriate for legal documents. Maintain dignity and authority.",
                    "priority": 1,
                    "tags": ["tone", "formal", "legal"]
                },
                {
                    "name": "Language Traits",
                    "content": "Use precise, unambiguous language. Avoid contractions and colloquialisms. Use formal sentence structure.",
                    "priority": 1,
                    "tags": ["language", "precise", "legal"]
                },
                {
                    "name": "Vocabulary",
                    "content": "Use established legal terminology. Define technical terms when necessary. Use consistent legal language throughout.",
                    "priority": 1,
                    "tags": ["vocabulary", "legal_terms", "legal"]
                },
                {
                    "name": "Structure",
                    "content": "Use clear headings and numbering. Group related clauses. Use consistent formatting and organization.",
                    "priority": 2,
                    "tags": ["structure", "organization", "legal"]
                }
            ]
        }
        
        # Create user guidelines for each domain
        for domain, guideline_list in domain_guidelines.items():
            for guideline_data in guideline_list:
                guideline = UserGuideline(
                    name=guideline_data["name"],
                    content=guideline_data["content"],
                    domain=domain,
                    priority=guideline_data["priority"],
                    tags=guideline_data["tags"]
                )
                guidelines.append(guideline)
        
        # Insert into MongoDB
        collection = await self.mongodb_client.get_collection("user_guidelines")
        if guidelines:
            guideline_docs = [guideline.model_dump(by_alias=True, exclude_none=True) for guideline in guidelines]
            result = await collection.insert_many(guideline_docs)
            logger.info(f"✅ Created {len(result.inserted_ids)} domain-aware user guidelines across {len(domain_guidelines)} domains")
            return len(result.inserted_ids)
        else:
            logger.warning("⚠️ No user guidelines created")
            return 0
    
    async def create_domain_aware_cultural_notes(self) -> int:
        """Create cultural notes for different domains and languages"""
        logger.info("🌍 Creating domain-aware cultural notes for multiple domains and languages")
        
        cultural_notes = []
        
        # Define domains and their cultural considerations
        domains = {
            "gaming": {
                "jp": [
                    {
                        "name": "Gaming Honorifics and Politeness",
                        "content": "Japanese gaming text should use appropriate honorifics (-san, -chan) and polite forms (desu/masu) for user-facing text. Avoid overly casual language unless targeting young audiences.",
                        "category": "language_formality",
                        "tags": ["honorifics", "politeness", "gaming"]
                    },
                    {
                        "name": "Katakana for Gaming Terms",
                        "content": "Use katakana (カタカナ) for foreign game terms, music genres, and technical concepts. This makes the text feel more natural to Japanese gamers.",
                        "category": "writing_system",
                        "tags": ["katakana", "foreign_terms", "gaming"]
                    },
                    {
                        "name": "Gaming UI Layout",
                        "content": "Consider vertical text layout for mobile games. Some UI elements may need to be adjusted for vertical reading patterns common in Japanese mobile interfaces.",
                        "category": "ui_layout",
                        "tags": ["vertical_text", "mobile", "ui", "gaming"]
                    },
                    {
                        "name": "Japanese Gaming Terminology",
                        "content": "Use established Japanese gaming terminology: ゲーム (game), プレイ (play), レベル (level), スコア (score). Avoid direct translations that don't sound natural.",
                        "category": "terminology",
                        "tags": ["gaming_terms", "localization", "japanese"]
                    }
                ],
                "fr": [
                    {
                        "name": "French Gaming Tone",
                        "content": "French gaming text should balance formal and informal tones. Use 'vous' for instructions but 'tu' for casual game interactions. Avoid overly academic language.",
                        "category": "language_formality",
                        "tags": ["formality", "tone", "gaming"]
                    },
                    {
                        "name": "Gaming Anglicisms in French",
                        "content": "French gamers are comfortable with English gaming terms (score, level, boss) but prefer French for UI elements and instructions. Use 'score' not 'pointage'.",
                        "category": "terminology",
                        "tags": ["anglicisms", "gaming_terms", "ui"]
                    },
                    {
                        "name": "French Gaming Cultural References",
                        "content": "French gamers appreciate cultural references to French music, cinema, and pop culture. Use familiar expressions and avoid overly American references.",
                        "category": "cultural_context",
                        "tags": ["cultural_references", "music", "pop_culture", "gaming"]
                    },
                    {
                        "name": "French Mobile Gaming",
                        "content": "French mobile gaming uses shorter, punchier text. Avoid long sentences and use action verbs. 'Tapez' instead of 'Appuyez sur' for 'Tap'.",
                        "category": "mobile_optimization",
                        "tags": ["mobile", "short_text", "action_verbs", "gaming"]
                    }
                ],
                "vn": [
                    {
                        "name": "Vietnamese Gaming Diacritics",
                        "content": "Vietnamese text must include proper tone marks (dấu thanh) as they change meaning. Always use full diacritics in gaming text for clarity and professionalism.",
                        "category": "writing_system",
                        "tags": ["tone_marks", "diacritics", "clarity", "gaming"]
                    },
                    {
                        "name": "Vietnamese Gaming Terminology",
                        "content": "Vietnamese gamers use a mix of Vietnamese and English terms. Use 'trò chơi' for game, 'chơi' for play, but keep 'level', 'score' in English as they're commonly used.",
                        "category": "terminology",
                        "tags": ["gaming_terms", "mixed_language", "localization"]
                    },
                    {
                        "name": "Vietnamese Gaming Community",
                        "content": "Vietnamese culture values family and community. Gaming text should be inclusive and avoid individualistic language. Use 'chúng ta' (we) instead of 'bạn' (you) when possible.",
                        "category": "cultural_values",
                        "tags": ["family", "community", "inclusive_language", "gaming"]
                    },
                    {
                        "name": "Vietnamese Mobile Gaming",
                        "content": "Vietnamese users are primarily mobile-first. Keep text concise and use familiar mobile gaming conventions. Avoid complex sentences and technical jargon.",
                        "category": "mobile_optimization",
                        "tags": ["mobile_first", "concise", "simple_language", "gaming"]
                    }
                ]
            },
            "software": {
                "jp": [
                    {
                        "name": "Japanese Software Formality",
                        "content": "Japanese software text should use appropriate keigo (polite language) for user-facing interfaces. Use masu-form for buttons and instructions, but plain form for error messages.",
                        "category": "language_formality",
                        "tags": ["keigo", "formality", "software", "ui"]
                    },
                    {
                        "name": "Japanese Technical Terms",
                        "content": "Use established Japanese technical terminology: ソフトウェア (software), アプリケーション (application), データベース (database). Avoid katakana for common terms.",
                        "category": "terminology",
                        "tags": ["technical_terms", "software", "localization"]
                    }
                ],
                "fr": [
                    {
                        "name": "French Software Terminology",
                        "content": "French software should use established technical terms: 'logiciel' for software, 'application' for app, 'base de données' for database. Avoid anglicisms when French equivalents exist.",
                        "category": "terminology",
                        "tags": ["technical_terms", "software", "french"]
                    },
                    {
                        "name": "French UI Conventions",
                        "content": "French UI text should follow French typography rules: spaces before punctuation marks, proper capitalization, and familiar interface terminology.",
                        "category": "ui_conventions",
                        "tags": ["typography", "ui", "french", "software"]
                    }
                ],
                "vn": [
                    {
                        "name": "Vietnamese Software Localization",
                        "content": "Vietnamese software should use clear, simple language. Avoid overly technical terms and use familiar Vietnamese expressions for common actions.",
                        "category": "language_simplicity",
                        "tags": ["simple_language", "software", "vietnamese"]
                    }
                ]
            },
            "healthcare": {
                "jp": [
                    {
                        "name": "Japanese Medical Formality",
                        "content": "Japanese medical text requires high formality and respect. Use appropriate honorifics and formal language, especially for patient-facing content.",
                        "category": "medical_formality",
                        "tags": ["medical", "formality", "respect", "japanese"]
                    },
                    {
                        "name": "Japanese Medical Terminology",
                        "content": "Use proper Japanese medical terminology and avoid direct translations. Consult medical dictionaries for accurate terms.",
                        "category": "medical_terminology",
                        "tags": ["medical_terms", "accuracy", "japanese"]
                    }
                ],
                "fr": [
                    {
                        "name": "French Medical Precision",
                        "content": "French medical text must be precise and clear. Use established medical terminology and avoid ambiguity that could affect patient safety.",
                        "category": "medical_precision",
                        "tags": ["medical", "precision", "safety", "french"]
                    }
                ],
                "vn": [
                    {
                        "name": "Vietnamese Medical Clarity",
                        "content": "Vietnamese medical text should be clear and accessible to patients. Use simple language while maintaining medical accuracy.",
                        "category": "medical_clarity",
                        "tags": ["medical", "clarity", "accessibility", "vietnamese"]
                    }
                ]
            },
            "ecommerce": {
                "jp": [
                    {
                        "name": "Japanese E-commerce Politeness",
                        "content": "Japanese e-commerce text should be polite and respectful. Use appropriate honorifics and formal language for customer-facing content.",
                        "category": "ecommerce_politeness",
                        "tags": ["ecommerce", "politeness", "customer_service", "japanese"]
                    }
                ],
                "fr": [
                    {
                        "name": "French E-commerce Trust",
                        "content": "French e-commerce text should build trust and confidence. Use clear, honest language and familiar French expressions for shopping.",
                        "category": "ecommerce_trust",
                        "tags": ["ecommerce", "trust", "confidence", "french"]
                    }
                ],
                "vn": [
                    {
                        "name": "Vietnamese E-commerce Simplicity",
                        "content": "Vietnamese e-commerce text should be simple and clear. Use familiar Vietnamese expressions for shopping and avoid complex terms.",
                        "category": "ecommerce_simplicity",
                        "tags": ["ecommerce", "simplicity", "clarity", "vietnamese"]
                    }
                ]
            }
        }
        
        # Create cultural notes for each domain and language combination
        for domain, languages in domains.items():
            for language, notes in languages.items():
                for note_data in notes:
                    cultural_note = CulturalNote(
                        name=note_data["name"],
                        content=note_data["content"],
                        target_language=language,
                        category=note_data["category"],
                        tags=note_data["tags"] + [domain, language]
                    )
                    cultural_notes.append(cultural_note)
        
        # Insert into MongoDB
        collection = await self.mongodb_client.get_collection("cultural_notes")
        if cultural_notes:
            note_docs = [note.model_dump(by_alias=True, exclude_none=True) for note in cultural_notes]
            result = await collection.insert_many(note_docs)
            logger.info(f"✅ Created {len(result.inserted_ids)} domain-aware cultural notes across {len(domains)} domains and {len(set([note.target_language for note in cultural_notes]))} languages")
            return len(result.inserted_ids)
        else:
            logger.warning("⚠️ No cultural notes created")
            return 0
    
    async def import_all_data(self, data_dir: str) -> Dict[str, int]:
        """Import all relevant data from the data directory"""
        logger.info("🚀 Starting data import process...")
        
        results = {
            'writing_traits': 0,
            'user_guidelines': 0,
            'cultural_notes': 0
        }
        
        try:
            # Initialize MongoDB connection
            await self.mongodb_client.initialize()
            
            # Import existing writing traits from CSV (gaming-focused)
            writing_traits_file = os.path.join(data_dir, "FRi - Localization Ref For AI - Writing Traits.csv")
            if os.path.exists(writing_traits_file):
                csv_traits = await self.import_writing_traits(writing_traits_file)
                results['writing_traits'] += csv_traits
            
            # Import existing user guidelines from CSV (gaming-focused)
            style_guide_file = os.path.join(data_dir, "FRi - Localization Ref For AI - Style Guide.csv")
            if os.path.exists(style_guide_file):
                csv_guidelines = await self.import_user_guidelines(style_guide_file)
                results['user_guidelines'] += csv_guidelines
            
            # Create domain-aware writing traits for multiple domains
            domain_traits = await self.create_domain_aware_writing_traits()
            results['writing_traits'] += domain_traits
            
            # Create domain-aware user guidelines for multiple domains
            domain_guidelines = await self.create_domain_aware_user_guidelines()
            results['user_guidelines'] += domain_guidelines
            
            # Create domain-aware cultural notes for multiple domains and languages
            results['cultural_notes'] = await self.create_domain_aware_cultural_notes()
            
            # Print summary
            logger.info("\n📊 Import Summary:")
            logger.info(f"   Writing Traits: {results['writing_traits']} (CSV + Domain-aware)")
            logger.info(f"   User Guidelines: {results['user_guidelines']} (CSV + Domain-aware)")
            logger.info(f"   Cultural Notes: {results['cultural_notes']} (Domain-aware)")
            logger.info(f"   Total Records: {sum(results.values())}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Import failed: {e}")
            raise
        finally:
            await self.mongodb_client.close()

async def main():
    """Main function to run the import process"""
    try:
        # Load settings
        settings = Settings()
        
        # Create MongoDB client
        mongodb_client = MongoDBClient(
            connection_string=settings.mongo_connection_string,
            database_name=settings.mongo_database
        )
        
        # Create importer
        importer = DataImporter(mongodb_client)
        
        # Get data directory path
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        
        if not data_dir.exists():
            logger.error(f"❌ Data directory not found: {data_dir}")
            return
        
        # Run import
        results = await importer.import_all_data(str(data_dir))
        
        logger.info("🎉 Data import completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Import process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
