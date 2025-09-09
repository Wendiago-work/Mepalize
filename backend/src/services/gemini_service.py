"""
Gemini Service for intelligent translation using RAG context
"""
from typing import List, Dict, Any, Optional
from ..config import get_settings
from .prompt_service import PromptService
from google import genai
from google.genai import types
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
                logger.error("Gemini API key not configured")
                raise ValueError("Gemini API key not configured. Please set TRANSLATION_GEMINI_API_KEY")
            
            # Configure the client with API key
            self.client = genai.Client(api_key=self.settings.gemini_api_key)
            
            # Use configurable settings with fallbacks
            model_name = self.settings.gemini_model
            temperature = self.settings.gemini_temperature
            max_tokens = self.settings.gemini_max_tokens
            
            # Configure very permissive safety settings for gaming content translation
            # Allow most content that might be flagged as "Dangerous" due to game nature
            safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ]
            
            # Store configuration for use in generation
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens or 2048
            self.safety_settings = safety_settings
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
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
        # Initialize context_prompt to avoid scope issues
        context_prompt = None
        
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
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=context_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        response_mime_type="application/json",
                        safety_settings=self.safety_settings
                    )
                )
                
            except Exception as api_error:
                logger.error(f"Gemini API call failed: {api_error}")
                raise api_error
            
            # Check for response issues
            if not response.text:
                # Check finish reason
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    safety_ratings = getattr(candidate, 'safety_ratings', [])
                    
                    # Use proper constants instead of magic numbers for better maintainability
                    # Check if it's a safety-related block (finish_reason can be 2 for SAFETY, 3 for RECITATION, etc.)
                    is_safety_block = finish_reason in [2, 3]  # SAFETY or RECITATION
                    
                    if is_safety_block:
                        logger.warning(f"Content blocked by safety filters - Finish reason: {finish_reason}")
                        
                        # Try without safety settings as a last resort
                        try:
                            no_safety_response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=context_prompt,
                                config=types.GenerateContentConfig(
                                    temperature=self.temperature,
                                    max_output_tokens=self.max_tokens,
                                    response_mime_type="application/json"
                                    # No safety_settings - most permissive
                                )
                            )
                            
                            if no_safety_response.text:
                                return self._parse_translation_response(no_safety_response.text)
                        except Exception as e:
                            logger.warning(f"Translation without safety settings also failed: {e}")
                        
                        # Try one more time with just the basic translation request
                        try:
                            basic_prompt = f"Translate '{source_text}' from {source_language} to {target_language}. Return only the translation."
                            basic_response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=basic_prompt,
                                config=types.GenerateContentConfig(
                                    temperature=0.1,
                                    max_output_tokens=1000
                                    # No safety_settings, no JSON response
                                )
                            )
                            
                            if basic_response.text:
                                return {
                                    "translated_text": basic_response.text.strip(),
                                    "translation": basic_response.text.strip(),  # Add backward compatibility
                                    "source": "gemini_ai_basic",
                                    "reasoning": "Used basic translation without context due to safety blocks",
                                    "cultural_notes": "No cultural adaptation - basic translation only",
                                    "style_applied": "No style guide applied - basic translation only",
                                    "domain_considerations": "No domain-specific considerations - basic translation only",
                                    "full_prompt": context_prompt
                                }
                        except Exception as e:
                            logger.warning(f"Basic translation also failed: {e}")
                        
                        # Return appropriate error message for safety blocks
                        return {
                            "translated_text": "[CONTENT BLOCKED] This content cannot be translated due to safety policies.",
                            "translation": "[CONTENT BLOCKED] This content cannot be translated due to safety policies.",  # Add backward compatibility
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
                        logger.error(f"Gemini response blocked - Finish reason: {finish_reason}")
                        
                        # For other types of blocks, try simplified prompt
                        try:
                            simple_prompt = f"Translate this text from {source_language} to {target_language}: '{source_text}'. Respond with only the translation."
                            simple_response = self.client.models.generate_content(
                                model=self.model_name,
                                contents=simple_prompt,
                                config=types.GenerateContentConfig(
                                    temperature=self.temperature,
                                    max_output_tokens=self.max_tokens,
                                    safety_settings=self.safety_settings
                                )
                            )
                            
                            if simple_response.text:
                                return {
                                    "translated_text": simple_response.text.strip(),
                                    "translation": simple_response.text.strip(),  # Add backward compatibility
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
                        return self._fallback_translation(source_text, target_language, context_prompt, finish_reason)
                else:
                    logger.error("Gemini returned empty response")
                    return self._fallback_translation(source_text, target_language, context_prompt, finish_reason)
            
            # Parse response - use parsed JSON when available
            response_data = None
            if hasattr(response, 'parsed') and response.parsed:
                response_data = response.parsed
            elif hasattr(response, 'to_json_dict') and response.to_json_dict():
                response_data = response.to_json_dict()
            elif response.text:
                response_data = response.text
            else:
                logger.error("No usable response data found")
                return self._fallback_translation(source_text, target_language, context_prompt, finish_reason)
            
            translation_result = self._parse_translation_response(response_data)
            
            
            # Add the full prompt to the response for frontend display
            translation_result["full_prompt"] = context_prompt
            
            return translation_result
            
        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            # Fallback to basic translation
            return self._fallback_translation(source_text, target_language, context_prompt, None)
    
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
        for i, trans in enumerate(retrieved_translations, 1):
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
        for i, guide in enumerate(style_guides, 1):
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
    
    
    
    
    def _parse_translation_response(self, response_data) -> Dict[str, Any]:
        """Parse Gemini's response into structured format - handles both JSON objects and text strings"""
        try:
            import json
            import re
            
            
            # Handle different response data types
            if isinstance(response_data, dict):
                # Check if this is a Gemini response structure with nested JSON
                if 'candidates' in response_data and response_data['candidates']:
                    candidate = response_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        text_parts = candidate['content']['parts']
                        if text_parts and 'text' in text_parts[0]:
                            # Extract the JSON text from the nested structure
                            json_text = text_parts[0]['text']
                            try:
                                parsed_json = json.loads(json_text)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON from candidate text: {e}")
                                parsed_json = None
                        else:
                            parsed_json = None
                    else:
                        parsed_json = None
                else:
                    # Already parsed JSON object
                    parsed_json = response_data
            else:
                # String response - try to extract JSON
                response_text = str(response_data)
            
                # Find JSON in the response - look for JSON blocks that might be wrapped in code blocks
                json_patterns = [
                    r'```json\s*(\{.*?\})\s*```',  # JSON in code block
                    r'```\s*(\{.*?\})\s*```',     # JSON in generic code block
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # More precise JSON matching
                ]
                
                parsed_json = None
                for i, pattern in enumerate(json_patterns):
                    json_match = re.search(pattern, response_text, re.DOTALL)
                    if json_match:
                        try:
                            json_text = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                            # Clean up the JSON text
                            json_text = json_text.strip()
                            parsed_json = json.loads(json_text)
                            break
                        except json.JSONDecodeError as e:
                            continue
            
            if parsed_json:
                # Map the LLM response fields to our expected structure
                # Support both "translation" and "translated_text" keys for backward compatibility
                translated_text = parsed_json.get("translated_text") or parsed_json.get("translation", "Translation failed")
                return {
                    "translated_text": translated_text,
                    "translation": translated_text,  # Add backward compatibility
                    "source": "gemini_ai",
                    "reasoning": parsed_json.get("reasoning", ""),
                    "cultural_notes": parsed_json.get("cultural_notes", ""),
                    "style_applied": parsed_json.get("style_applied", ""),
                    "domain_considerations": parsed_json.get("domain_considerations", "")
                }
            else:
                # Fallback if no JSON found - try to extract translation from text
                logger.warning("No valid JSON found in Gemini response, using fallback parsing")
                
                # Get the response text for fallback parsing
                response_text = str(response_data) if not isinstance(response_data, dict) else str(response_data)
                
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
                
                # Strip translation prefixes from the extracted text
                final_text = extracted_translation or response_text.strip()
                final_text = re.sub(r'^(翻訳|Translation)\s*:\s*', '', final_text, flags=re.IGNORECASE)
                
                result = {
                    "translated_text": final_text,
                    "translation": final_text,  # Add backward compatibility
                    "source": "gemini_ai_fallback",
                    "reasoning": "Response parsed without structured format - extracted translation from text patterns",
                    "cultural_notes": "",
                    "style_applied": "",
                    "domain_considerations": ""
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Exception in parsing: {e}")
            # Strip translation prefixes from raw response
            final_text = str(response_data).strip()
            final_text = re.sub(r'^(翻訳|Translation)\s*:\s*', '', final_text, flags=re.IGNORECASE)
            
            result = {
                "translated_text": final_text,
                "translation": final_text,  # Add backward compatibility
                "source": "gemini_ai_raw",
                "reasoning": f"Raw response due to parsing error: {e}",
                "cultural_notes": "",
                "style_applied": "",
                "domain_considerations": ""
            }
            
            return result
    

    def _fallback_translation(self, source_text: str, target_language: str, context_prompt: str = None, finish_reason: int = None) -> Dict[str, Any]:
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
        
        # Strip translation prefixes from fallback text
        import re
        clean_fallback_text = re.sub(r'^(翻訳|Translation)\s*:\s*', '', fallback_text, flags=re.IGNORECASE)
        
        result = {
            "translated_text": clean_fallback_text,
            "translation": clean_fallback_text,  # Add backward compatibility
            "source": "fallback",
            "reasoning": f"Gemini service unavailable or blocked (finish_reason={finish_reason}), using basic fallback for {target_language}",
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
            
            # Handle any object structure intelligently
            if isinstance(cultural_note, str):
                cultural_note_str = cultural_note
            elif isinstance(cultural_note, dict):
                # Convert dictionary to readable format with proper formatting
                cultural_note_str = self._format_object_content(cultural_note)
            elif isinstance(cultural_note, list):
                # Convert list to readable format
                cultural_note_str = self._format_object_content(cultural_note)
            else:
                # Convert any other object type to string
                cultural_note_str = str(cultural_note)
            
            # Ensure domain is a string before calling upper()
            domain_str = str(domain) if domain is not None else 'general'
            context += f"{i}. [{domain_str.upper()}] {language}: {cultural_note_str}\n"
        
        return context
    
    def _format_object_content(self, content: Any) -> str:
        """Format any object content (dict, list, etc.) into a readable string"""
        import json
        
        try:
            # Try to format as JSON for better readability
            if isinstance(content, (dict, list)):
                return json.dumps(content, indent=2, ensure_ascii=False)
            else:
                return str(content)
        except (TypeError, ValueError):
            # Fallback to string representation if JSON serialization fails
            return str(content)
