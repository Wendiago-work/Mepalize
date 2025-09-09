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
