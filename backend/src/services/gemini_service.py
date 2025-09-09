"""
Gemini Service for intelligent translation using RAG context
"""
from typing import List, Dict, Any, Optional
from ..config import get_settings
from .prompt_service import PromptService
import google.generativeai as genai
from ..core.logger import get_logger

logger = get_logger("gemini_service", "translation")

# Mock mode for testing - set to True to avoid LLM costs
MOCK_MODE = False


class GeminiService:
    """Service for interacting with Google Gemini Pro for intelligent translation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.prompt_service = PromptService()
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Initialize Gemini client"""
        try:
            # Validate API key
            if not self.settings.gemini_api_key or self.settings.gemini_api_key == "your_gemini_api_key_here":
                raise ValueError("Gemini API key not configured. Please set TRANSLATION_GEMINI_API_KEY")
            
            genai.configure(api_key=self.settings.gemini_api_key)
            
            # Use configurable settings with fallbacks
            model_name = self.settings.gemini_model
            temperature = self.settings.gemini_temperature
            max_tokens = self.settings.gemini_max_tokens
            
            # Configure safety settings based on config
            safety_settings = {}
            if self.settings.enable_safety_filters:
                # Map safety threshold string to Gemini enum
                threshold_map = {
                    "NONE": genai.types.HarmBlockThreshold.BLOCK_NONE,
                    "LOW_AND_ABOVE": genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    "MEDIUM_AND_ABOVE": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    "HIGH_AND_ABOVE": genai.types.HarmBlockThreshold.BLOCK_HIGH_AND_ABOVE,
                    "ONLY_HIGH": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }
                threshold = threshold_map.get(self.settings.safety_threshold, genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
                
                # SAFETY FIRST: Block harmful content to protect users
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
                }
                logger.info(f"🛡️ Safety filters enabled with threshold: {self.settings.safety_threshold}")
            else:
                logger.warning("⚠️ Safety filters DISABLED - this is dangerous for production use!")
            
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                safety_settings=safety_settings
            )
            logger.info(f"✅ Gemini service initialized with model: {model_name}")
            logger.info(f"   Temperature: {temperature}, Max tokens: {max_tokens}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini service: {e}")
            raise
    
    def generate_translation_with_context(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        retrieved_translations: List[Dict[str, Any]],
        glossaries: List[Dict[str, Any]],
        domain_style_guide: Dict[str, Any] = None,
        cultural_notes: List[Dict[str, Any]] = None,
        domain: str = "general",
        attachments: List[Dict[str, Any]] = None,
        context_notes: Optional[str] = None
        # User context takes priority over RAG content
    ) -> Dict[str, Any]:
        """
        Generate intelligent translation using combined MongoDB and Qdrant context
        
        Args:
            source_text: Text to translate
            source_language: Source language code
            target_language: Target language code
            retrieved_translations: Translation memory from Qdrant
            glossaries: Glossaries and terminology from Qdrant
            domain_style_guide: Complete domain style guide from MongoDB
            cultural_notes: Cultural notes from MongoDB
            domain: Domain for context (e.g., "gaming", "software", "healthcare")
            attachments: Multi-modal attachments
            context_notes: User-provided context (takes priority)
            
        Returns:
            Dict with translation, reasoning, cultural_notes, and style_applied
        """
        try:
            # Input validation for safety
            if not source_text or not source_text.strip():
                return {
                    "translated_text": "[ERROR] No text provided for translation",
                    "source": "validation_error",
                    "reasoning": "Empty or missing source text",
                    "cultural_notes": "",
                    "style_applied": "",
                    "domain_considerations": "",
                    "error": "EMPTY_INPUT",
                    "error_message": "No text provided for translation"
                }
            
            # Basic content length validation
            if len(source_text) > 10000:  # 10k character limit
                return {
                    "translated_text": "[ERROR] Text too long for translation",
                    "source": "validation_error", 
                    "reasoning": "Input text exceeds maximum length limit",
                    "cultural_notes": "",
                    "style_applied": "",
                    "domain_considerations": "",
                    "error": "TEXT_TOO_LONG",
                    "error_message": "Text exceeds maximum length limit (10,000 characters)"
                }
            # Build context prompt with user priority
            context_prompt = self._build_context_prompt(
                source_text, source_language, target_language, 
                retrieved_translations, glossaries, domain_style_guide,
                cultural_notes, domain,
                attachments, context_notes
            )
            
            # 🆕 LOGGING: Final prompt being sent to Gemini
            print(f"\n🧠 ===== FINAL PROMPT SENT TO GEMINI =====")
            print(f"📝 Source Text: '{source_text}'")
            print(f"🌍 Language: {source_language} → {target_language}")
            print(f"📝 User Context: {context_notes or 'None (RAG will infer)'}")
            print(f"🔍 RAG Context:")
            print(f"   - Similar Translations: {len(retrieved_translations)}")
            print(f"   - Glossaries: {len(glossaries)}")
            print(f"📋 Full Prompt:")
            print(f"{context_prompt}")
            print(f"==========================================\n")
            
            # Mock mode for testing - return fake translation after showing prompt
            if MOCK_MODE:
                logger.info("🧪 MOCK MODE: Returning fake translation to avoid LLM costs")
                return {
                    "translation": f"[MOCK] {source_text} → {target_language}",
                    "reasoning": "Mock translation for testing purposes",
                    "cultural_notes": "Mock cultural notes",
                    "style_applied": "Mock style guide applied",
                    "confidence": 0.95,
                    "source_language": source_language,
                    "target_language": target_language,
                    "domain": domain,
                    "mock_mode": True,
                    "full_prompt": context_prompt
                }
            
            # Generate translation using Gemini
            response = self.model.generate_content(context_prompt)
            
            # Check for response issues
            if not response.text:
                # Check finish reason
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    safety_ratings = getattr(candidate, 'safety_ratings', [])
                    
                    # Handle safety blocks appropriately
                    if finish_reason == 2:  # SAFETY
                        logger.warning(f"⚠️ Content blocked by safety filters - Finish reason: {finish_reason}")
                        if safety_ratings:
                            blocked_categories = []
                            for rating in safety_ratings:
                                if hasattr(rating, 'probability') and rating.probability >= 2:  # MEDIUM or HIGH
                                    blocked_categories.append(rating.category.name)
                            logger.warning(f"Blocked categories: {blocked_categories}")
                        
                        # Return appropriate error message for safety blocks
                        return {
                            "translated_text": "[CONTENT BLOCKED] This content cannot be translated due to safety policies.",
                            "source": "safety_blocked",
                            "reasoning": "Content was blocked by safety filters. Please review the input text for potentially harmful content.",
                            "cultural_notes": "Translation blocked for safety reasons",
                            "style_applied": "No style guide applied - content blocked",
                            "domain_considerations": "Content blocked by safety filters",
                            "error": "SAFETY_BLOCKED",
                            "error_message": "Content blocked by safety filters. Please review input text.",
                            "full_prompt": context_prompt
                        }
                    else:
                        logger.error(f"❌ Gemini response blocked - Finish reason: {finish_reason}")
                        if safety_ratings:
                            logger.error(f"Safety ratings: {safety_ratings}")
                        
                        # For other types of blocks, try simplified prompt
                        try:
                            logger.info("🔄 Attempting simplified prompt as fallback...")
                            simple_prompt = f"Translate this text from {source_language} to {target_language}: '{source_text}'. Respond with only the translation."
                            simple_response = self.model.generate_content(simple_prompt)
                            
                            if simple_response.text:
                                logger.info("✅ Simplified prompt succeeded")
                                return {
                                    "translated_text": simple_response.text.strip(),
                                    "source": "gemini_ai_simplified",
                                    "reasoning": "Used simplified prompt due to context filter",
                                    "cultural_notes": "No cultural adaptation - simplified translation only",
                                    "style_applied": "No style guide applied - simplified translation only",
                                    "domain_considerations": "No domain-specific considerations - simplified translation only",
                                    "full_prompt": context_prompt
                                }
                        except Exception as e:
                            logger.warning(f"Simplified prompt also failed: {e}")
                        
                        # Return fallback translation for non-safety blocks
                        return self._fallback_translation(source_text, target_language, context_prompt)
                else:
                    logger.error("❌ Gemini returned empty response")
                    return self._fallback_translation(source_text, target_language, context_prompt)
            
            # 🆕 LOGGING: Raw Gemini response
            print(f"\n🤖 ===== RAW GEMINI RESPONSE =====")
            print(f"📤 Response Status: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'Success'}")
            print(f"📝 Raw Response Text: {response.text}")
            print(f"==========================================\n")
            
            # Parse response
            translation_result = self._parse_translation_response(response.text)
            
            # 🆕 LOGGING: Parsed result
            print(f"\n🔍 ===== PARSED TRANSLATION RESULT =====")
            print(f"📝 Translated Text: '{translation_result.get('translated_text', 'N/A')}'")
            print(f"🧠 Reasoning: {translation_result.get('reasoning', 'N/A')[:100]}...")
            print(f"🌍 Cultural Notes: {translation_result.get('cultural_notes', 'N/A')[:100]}...")
            print(f"🎨 Style Applied: {translation_result.get('style_applied', 'N/A')[:100]}...")
            print(f"🏢 Domain Considerations: {translation_result.get('domain_considerations', 'N/A')[:100]}...")
            print(f"==========================================\n")
            
            logger.info(f"✅ Translation generated for {source_language} -> {target_language}")
            
            # Add the full prompt to the response for frontend display
            translation_result["full_prompt"] = context_prompt
            
            return translation_result
            
        except Exception as e:
            logger.error(f"❌ Gemini translation failed: {e}")
            # Fallback to basic translation
            return self._fallback_translation(source_text, target_language, context_prompt)
    
    def _build_context_prompt(
        self,
        source_text: str,
        source_language: str,
        target_language: str,
        retrieved_translations: List[Dict[str, Any]],
        glossaries: List[Dict[str, Any]],
        domain_style_guide: Dict[str, Any] = None,
        cultural_notes: List[Dict[str, Any]] = None,
        domain: str = "general",
        attachments: List[Dict[str, Any]] = None,
        context_notes: Optional[str] = None
        # User context takes priority over RAG content
    ) -> str:
        """Build comprehensive prompt with user context priority over RAG content"""
        
        # Build context sections
        retrieved_translations_context = self._format_retrieved_translations(retrieved_translations)
        glossaries_context = self._format_style_guides(glossaries)  # These are actually glossaries from Qdrant
        domain_style_guide_context = self._format_style_guide(domain_style_guide) if domain_style_guide else "No domain style guide"
        cultural_notes_context = self._format_cultural_notes(cultural_notes) if cultural_notes else "No cultural notes"
        
        # PRIORITY LOGIC: User context takes precedence over RAG content
        # If user provides context_notes, it overrides RAG-inferred context
        user_context_priority = ""
        if context_notes:
            user_context_priority = f"""
IMPORTANT: USER PROVIDED CONTEXT (TAKES PRIORITY):
{context_notes}

This user-provided context should take precedence over any RAG-retrieved content below.
If there are conflicts between user context and RAG content, prioritize the user context.
"""
        
        # Use prompt template from PromptService
        prompt = self.prompt_service.get_translation_prompt().format(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            domain=domain,
            retrieved_translations_context=retrieved_translations_context,
            glossaries_context=glossaries_context,
            domain_style_guide_context=domain_style_guide_context,
            cultural_notes_context=cultural_notes_context,
            context_notes=user_context_priority or "No additional user context - rely on RAG content"
        )
        
        return prompt
    
    def _format_retrieved_translations(self, retrieved_translations: List[Dict[str, Any]]) -> str:
        """Format retrieved translations for prompt context"""
        if not retrieved_translations:
            return "No similar translations found."
        
        context = ""
        for i, trans in enumerate(retrieved_translations[:3], 1):
            payload = trans.get('payload', {})
            content = trans.get('content', 'N/A')
            content_type = payload.get('content_type', 'unknown')
            
            # Handle unified data structure
            if content_type == 'translation_pair':
                # Translation pairs
                source_text = payload.get('translation_source', 'N/A')
                target_text = payload.get('translation_target', 'N/A')
                
                context += f"""
{i}. [Translation] Source: "{source_text}"
   Translation: "{target_text}"
   Context: {content}"""
            else:
                # Other content types (style guides, etc.)
                context += f"""
{i}. [{content_type.title()}] Content: {content}"""
        
        return context
    
    def _format_style_guides(self, style_guides: List[Dict[str, Any]]) -> str:
        """Format style guides for prompt context - simplified"""
        if not style_guides:
            return "No specific style guides found."
        
        context = ""
        for i, guide in enumerate(style_guides[:6], 1):
            content = guide.get('content', 'N/A')
            payload = guide.get('payload', {})
            category = payload.get('category', 'General')
            aspect = payload.get('aspect', 'General')
            
            # Only show category/aspect if they're meaningful (not 'General' or 'N/A')
            if category != 'General' and aspect != 'General':
                context += f"\n{i}. [{category} - {aspect}] {content}"
            else:
                context += f"\n{i}. {content}"
        
        return context
    
    def _format_style_guide(self, style_guide: Dict[str, Any]) -> str:
        """Format complete style guide for prompt context - pass raw JSON to LLM"""
        if not style_guide:
            return "No style guide available."
        
        # Extract the actual style guide content from MongoDB document
        style_guide_content = style_guide.get('style_guide', {}) if isinstance(style_guide, dict) else style_guide
        
        if not style_guide_content:
            return "No style guide available."
        
        # Return the complete JSON object as a formatted string for the LLM
        import json
        return json.dumps(style_guide_content, indent=2, ensure_ascii=False)
    
    
    
    
    def _parse_translation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured format"""
        try:
            # Try to extract JSON from response
            import json
            import re
            
            # Find JSON in the response - look for JSON blocks that might be wrapped in code blocks
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON in code block
                r'```\s*(\{.*?\})\s*```',     # JSON in generic code block
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # More precise JSON matching
            ]
            
            parsed_json = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    try:
                        json_text = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                        # Clean up the JSON text
                        json_text = json_text.strip()
                        parsed_json = json.loads(json_text)
                        logger.info(f"✅ Successfully parsed JSON using pattern: {pattern}")
                        break
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON with pattern {pattern}: {e}")
                        continue
            
            if parsed_json:
                # Map the LLM response fields to our expected structure
                return {
                    "translated_text": parsed_json.get("translation", "Translation failed"),
                    "source": "gemini_ai",
                    "reasoning": parsed_json.get("reasoning", ""),
                    "cultural_notes": parsed_json.get("cultural_notes", ""),
                    "style_applied": parsed_json.get("style_applied", ""),
                    "domain_considerations": parsed_json.get("domain_considerations", "")
                }
            else:
                # Fallback if no JSON found - try to extract translation from text
                logger.warning("No valid JSON found in Gemini response, using fallback parsing")
                
                # Try to extract translation from common patterns
                translation_patterns = [
                    r'"translation":\s*"([^"]+)"',  # "translation": "text"
                    r'translation:\s*"([^"]+)"',    # translation: "text"
                    r'Translated Text:\s*"([^"]+)"', # Translated Text: "text"
                    r'Translation:\s*"([^"]+)"',     # Translation: "text"
                ]
                
                extracted_translation = None
                for pattern in translation_patterns:
                    match = re.search(pattern, response_text, re.IGNORECASE)
                    if match:
                        extracted_translation = match.group(1)
                        break
                
                return {
                    "translated_text": extracted_translation or response_text.strip(),
                    "source": "gemini_ai_fallback",
                    "reasoning": "Response parsed without structured format - extracted translation from text patterns",
                    "cultural_notes": "",
                    "style_applied": "",
                    "domain_considerations": ""
                }
                
        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return {
                "translated_text": response_text.strip(),
                "source": "gemini_ai_raw",
                "reasoning": f"Raw response due to parsing error: {e}",
                "cultural_notes": "",
                "style_applied": "",
                "domain_considerations": ""
            }
    

    def _fallback_translation(self, source_text: str, target_language: str, context_prompt: str = None) -> Dict[str, Any]:
        """Fallback translation when Gemini fails"""
        # Simple fallback translations for common languages
        fallback_translations = {
            "ja": f"翻訳: {source_text}",
            "fr": f"Traduction: {source_text}",
            "es": f"Traducción: {source_text}",
            "de": f"Übersetzung: {source_text}",
            "ko": f"번역: {source_text}",
            "zh": f"翻译: {source_text}",
            "pt": f"Tradução: {source_text}",
            "it": f"Traduzione: {source_text}",
            "ru": f"Перевод: {source_text}",
            "ar": f"ترجمة: {source_text}",
        }
        
        fallback_text = fallback_translations.get(target_language, f"Translation: {source_text}")
        
        result = {
            "translated_text": fallback_text,
            "source": "fallback",
            "reasoning": f"Gemini service unavailable or blocked (finish_reason=2), using basic fallback for {target_language}",
            "cultural_notes": f"Fallback translation - no cultural adaptation applied",
            "style_applied": "No style guide applied - basic fallback only",
            "domain_considerations": "No domain-specific considerations - basic fallback only"
        }
        
        if context_prompt:
            result["full_prompt"] = context_prompt
            
        return result
    
    
    
    
    
    
    
    def _format_cultural_notes(self, cultural_notes: List[Dict[str, Any]]) -> str:
        """Format cultural notes for prompt context"""
        if not cultural_notes:
            return "No cultural notes available."
        
        context = ""
        for i, note in enumerate(cultural_notes[:5], 1):
            # Use the correct field names from CulturalNote model
            language = note.get('language', 'Unknown')
            domain = note.get('domain', 'general')
            cultural_note = note.get('cultural_note', 'N/A')
            
            # Ensure domain is a string before calling upper()
            domain_str = str(domain) if domain is not None else 'general'
            context += f"{i}. [{domain_str.upper()}] {language}: {cultural_note}\n"
        
        return context
