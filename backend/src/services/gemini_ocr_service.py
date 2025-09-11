"""
Google Gemini OCR Service
Uses Google's Gemini AI model for image text extraction
"""
import base64
import io
from typing import Dict, Any, Optional
from google import genai
from PIL import Image
from ..core.logger import get_logger
from ..config.config import get_settings

logger = get_logger("gemini_ocr_service", "image_processing")

class GeminiOCRService:
    """OCR service using Google Gemini AI"""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.gemini_api_key
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set - Gemini OCR service will not work")
            self.client = None
        else:
            try:
                # Initialize the client with API key
                self.client = genai.Client(api_key=self.api_key)
                logger.info("✅ Gemini OCR service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini OCR service: {e}")
                self.client = None
    
    async def extract_text_from_image(
        self, 
        base64_data: str, 
        filename: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Extract text from base64-encoded image data using Google Gemini.
        This service only handles text extraction - translation is handled by the main pipeline.
        
        Args:
            base64_data: Base64-encoded image data
            filename: Original filename for logging
            
        Returns:
            Dictionary with OCR results containing only the extracted text
        """
        if not self.client:
            return {
                "text": "",
                "filename": filename,
                "extraction_method": "gemini_ai",
                "success": False,
                "error": "Gemini client not initialized - check GEMINI_API_KEY"
            }
        
        try:
            # Decode base64 to get file size and image
            image_bytes = base64.b64decode(base64_data)
            file_size = len(image_bytes)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary (Gemini works better with RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"🔍 Processing image with Gemini: {filename} ({file_size} bytes, {image.size})")
            
            # Create the prompt for text extraction only
            prompt = """Extract all text from this image. Please provide the text exactly as it appears, preserving line breaks and formatting. If there are multiple text elements, separate them with line breaks. Focus on accuracy and completeness."""
            
            # Generate content using Gemini with the new API
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image]
            )
            
            if response and response.text:
                extracted_text = response.text.strip()
                
                if extracted_text:
                    logger.info(f"✅ Gemini extracted text: '{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}'")
                    
                    # Return only the extracted text
                    return {
                        "text": extracted_text,
                        "filename": filename,
                        "extraction_method": "gemini_ai",
                        "success": True
                    }
                else:
                    logger.warning(f"⚠️ Gemini found no text in image: {filename}")
                    return {
                        "text": "",
                        "filename": filename,
                        "extraction_method": "gemini_ai",
                        "success": True,  # API call succeeded, just no text found
                        "error": "No text detected in image"
                    }
            else:
                logger.warning(f"⚠️ Gemini returned no response for: {filename}")
                return {
                    "text": "",
                    "filename": filename,
                    "extraction_method": "gemini_ai",
                    "success": False,
                    "error": "No response from Gemini API"
                }
                
        except Exception as e:
            logger.error(f"Gemini OCR extraction failed: {e}")
            return {
                "text": "",
                "filename": filename,
                "extraction_method": "gemini_ai",
                "success": False,
                "error": str(e)
            }

# Global instance
_gemini_ocr_service = None

def get_gemini_ocr_service() -> GeminiOCRService:
    """Get the Gemini OCR service instance"""
    global _gemini_ocr_service
    if _gemini_ocr_service is None:
        _gemini_ocr_service = GeminiOCRService()
    return _gemini_ocr_service

# Convenience function for async usage
async def extract_text_from_image_gemini(
    base64_data: str, 
    filename: str = "unknown"
) -> Dict[str, Any]:
    """Extract text from image using Google Gemini AI"""
    service = get_gemini_ocr_service()
    return await service.extract_text_from_image(
        base64_data, filename
    )

