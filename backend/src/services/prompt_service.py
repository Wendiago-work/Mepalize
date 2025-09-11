"""
Prompt Service for loading and managing prompt templates
"""

import os
from pathlib import Path
from typing import Dict, Optional
from ..core.logger import get_logger

logger = get_logger("prompt_service", "template_management")


class PromptService:
    """Service for loading and managing prompt templates"""
    
    def __init__(self):
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompt templates from the prompts directory"""
        self.prompts: Dict[str, str] = {}
        
        try:
            if not self.prompts_dir.exists():
                logger.warning(f"Prompts directory not found: {self.prompts_dir}")
                return
            
            # Load all .txt files in the prompts directory
            for prompt_file in self.prompts_dir.glob("*.txt"):
                prompt_name = prompt_file.stem  # filename without extension
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.prompts[prompt_name] = f.read().strip()
                    logger.info(f"✅ Loaded prompt template: {prompt_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load prompt {prompt_name}: {e}")
            
            logger.info(f"📚 Loaded {len(self.prompts)} prompt templates")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize prompt service: {e}")
    
    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """Get a prompt template by name"""
        return self.prompts.get(prompt_name)
    
    def get_translation_prompt(self) -> str:
        """Get the translation prompt template"""
        prompt = self.get_prompt("translation_prompt")
        if not prompt:
            raise ValueError("Translation prompt template not found")
        return prompt
    
    
    def list_available_prompts(self) -> list:
        """List all available prompt names"""
        return list(self.prompts.keys())
    
    def reload_prompts(self):
        """Reload prompts from disk (useful for development)"""
        logger.info("🔄 Reloading prompt templates...")
        self._load_prompts()
    
    def generate_final_prompt(self, 
                            source_text: str,
                            source_language: str,
                            target_language: str,
                            domain: str,
                            retrieved_translations: list,
                            glossaries: list,
                            domain_style_guide: dict = None,
                            cultural_notes: list = None,
                            context_notes: str = None) -> str:
        """Generate the final prompt with Chroma DB context for translation"""
        
        # Handle empty source text
        source_text_section = self._format_source_text_section(source_text)
        
        # Format retrieved translations
        retrieved_translations_context = self._format_retrieved_translations(retrieved_translations)
        
        # Format glossaries
        glossaries_context = self._format_glossaries(glossaries)
        
        # Format domain style guide
        domain_style_guide_context = self._format_style_guide(domain_style_guide) if domain_style_guide else "No domain style guide available"
        
        # Format cultural notes
        cultural_notes_context = self._format_cultural_notes(cultural_notes) if cultural_notes else "No cultural notes available"
        
        # Get the base prompt template
        prompt_template = self.get_translation_prompt()
        
        # Format the prompt with all context
        final_prompt = prompt_template.format(
            source_text_section=source_text_section,
            source_language=source_language,
            target_language=target_language,
            domain=domain,
            retrieved_translations_context=retrieved_translations_context,
            glossaries_context=glossaries_context,
            domain_style_guide_context=domain_style_guide_context,
            cultural_notes_context=cultural_notes_context,
            context_notes=context_notes or "No additional user context - rely on RAG content"
        )
        
        return final_prompt
    
    def _format_source_text_section(self, source_text: str) -> str:
        """Format the source text section, handling empty text appropriately"""
        if not source_text or not source_text.strip():
            return "NOTE: No source text provided. Please focus on translating any text content found in the attached images only."
        
        return f'SOURCE TEXT: "{source_text}"'
    
    def _format_retrieved_translations(self, retrieved_translations: list) -> str:
        """Format retrieved translation memory for prompt"""
        if not retrieved_translations:
            return "No relevant translation memory found."
        
        formatted = []
        for i, tm in enumerate(retrieved_translations, 1):
            # Handle both RetrievedDocument objects and dictionaries
            if hasattr(tm, 'content'):
                # RetrievedDocument object
                source = tm.content
                target = tm.metadata.get('target_text', '') if tm.metadata else ''
                domain = tm.metadata.get('domain', '') if tm.metadata else ''
                use_case = tm.metadata.get('use_case', '') if tm.metadata else ''
            else:
                # Dictionary format
                source = tm.get('content', '')
                target = tm.get('metadata', {}).get('target_text', '')
                domain = tm.get('metadata', {}).get('domain', '')
                use_case = tm.get('metadata', {}).get('use_case', '')
            
            if source and target:
                context_info = f" (Domain: {domain}, Use Case: {use_case})" if domain or use_case else ""
                formatted.append(f"{i}. EN: \"{source}\" → {target}{context_info}")
        
        return "\n".join(formatted) if formatted else "No relevant translation memory found."
    
    def _format_glossaries(self, glossaries: list) -> str:
        """Format retrieved glossaries for prompt"""
        if not glossaries:
            return "No relevant glossary terms found."
        
        formatted = []
        for i, glossary in enumerate(glossaries, 1):
            # Handle both RetrievedDocument objects and dictionaries
            if hasattr(glossary, 'content'):
                # RetrievedDocument object
                term = glossary.metadata.get('term', '') if glossary.metadata else ''
                definition = glossary.metadata.get('definition', '') if glossary.metadata else ''
                full_name_jp = glossary.metadata.get('full_name_japanese', '') if glossary.metadata else ''
                common_display_jp = glossary.metadata.get('common_display_japanese', '') if glossary.metadata else ''
            else:
                # Dictionary format
                term = glossary.get('metadata', {}).get('term', '')
                definition = glossary.get('metadata', {}).get('definition', '')
                full_name_jp = glossary.get('metadata', {}).get('full_name_japanese', '')
                common_display_jp = glossary.get('metadata', {}).get('common_display_japanese', '')
            
            if term and definition:
                jp_info = f" (JP: {full_name_jp} / {common_display_jp})" if full_name_jp or common_display_jp else ""
                formatted.append(f"{i}. {term}: {definition}{jp_info}")
        
        return "\n".join(formatted) if formatted else "No relevant glossary terms found."
    
    def _format_style_guide(self, style_guide: dict) -> str:
        """Format domain style guide for prompt - return as-is without parsing structure"""
        if not style_guide:
            return "No style guide available."
        
        # Return the style guide as-is without trying to parse the structure
        return str(style_guide)
    
    def _format_cultural_notes(self, cultural_notes: list) -> str:
        """Format cultural notes for prompt - return as-is without parsing structure"""
        if not cultural_notes:
            return "No cultural notes available."
        
        # Return the cultural notes as-is without trying to parse the structure
        return str(cultural_notes)