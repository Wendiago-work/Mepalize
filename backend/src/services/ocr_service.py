from __future__ import annotations
import io
from typing import List, Tuple, Dict, Any, Optional
import base64
from PIL import Image, ImageOps, ImageFilter
from rapidocr_onnxruntime import RapidOCR

from ..core.types import Attachment
from ..core.logger import get_logger

logger = get_logger("ocr_service", "image_processing")

# ---------- Pre-processing for poster/UI images ----------
def _preprocess_image(src_bytes: bytes) -> bytes:
    """
    Preprocess image for better OCR results on UI/poster images.
    - Convert RGBA → RGB (flatten transparency)
    - Apply autocontrast to improve text visibility
    - Light sharpening to enhance text edges
    - Optional upscaling for small images
    """
    im = Image.open(io.BytesIO(src_bytes)).convert("RGB")
    
    # Autocontrast to improve text visibility
    im = ImageOps.autocontrast(im, cutoff=1)
    
    # Light sharpening to enhance text edges
    im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Upscale small images for better OCR
    if max(im.size) < 1600:
        im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    
    # Save as PNG to preserve quality
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()

def _rotate_image_variants(image_bytes: bytes) -> List[Tuple[int, bytes]]:
    """Generate 0/90/180/270 degree rotation variants for CJK vertical text detection."""
    variants = []
    im = Image.open(io.BytesIO(image_bytes))
    
    for deg in [0, 90, 180, 270]:
        b = io.BytesIO()
        im.rotate(deg, expand=True).save(b, format="PNG", optimize=True)
        variants.append((deg, b.getvalue()))
    return variants

def _group_text_by_proximity(detections: List) -> str:
    """Group detected text by spatial proximity to create more coherent text blocks"""
    if not detections:
        return ""
    
    # Extract text with bounding box information
    text_items = []
    for detection in detections:
        if isinstance(detection, (list, tuple)) and len(detection) >= 3:
            bbox, text, confidence = detection[0], detection[1], detection[2]
            if isinstance(text, str) and text.strip():
                # Calculate center point of bounding box
                if isinstance(bbox, list) and len(bbox) >= 4:
                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    text_items.append({
                        'text': text.strip(),
                        'x': center_x,
                        'y': center_y,
                        'confidence': confidence
                    })
    
    if not text_items:
        return ""
    
    # Sort by Y coordinate (top to bottom), then by X coordinate (left to right)
    text_items.sort(key=lambda item: (item['y'], item['x']))
    
    # Group nearby text items
    groups = []
    current_group = [text_items[0]]
    
    for i in range(1, len(text_items)):
        current_item = text_items[i]
        last_item = current_group[-1]
        
        # Group if items are close vertically (within 50 pixels) and horizontally (within 200 pixels)
        if (abs(current_item['y'] - last_item['y']) < 50 and 
            abs(current_item['x'] - last_item['x']) < 200):
            current_group.append(current_item)
        else:
            # Start a new group
            groups.append(current_group)
            current_group = [current_item]
    
    # Add the last group
    if current_group:
        groups.append(current_group)
    
    # Convert groups to text
    result_lines = []
    for group in groups:
        # Sort group by X coordinate (left to right)
        group.sort(key=lambda item: item['x'])
        group_text = " ".join([item['text'] for item in group])
        result_lines.append(group_text)
    
    return "\n".join(result_lines)

# ---------- PaddleOCR (RapidOCR) Implementation ----------
def _make_rapidocr() -> RapidOCR:
    """Create RapidOCR instance with optimized settings for UI/poster images."""
    return RapidOCR(
        use_angle_cls=True,  # Enable text angle classification
        use_text_det=True,   # Enable text detection
        use_text_rec=True,   # Enable text recognition
        text_score=0.3,      # Lower threshold for text detection
        rec_score=0.3,       # Lower threshold for text recognition
    )

# Single global OCR instance (thread-safe)
try:
    _OCR = _make_rapidocr()
except Exception as e:
    logger.error(f"Failed to create OCR instance: {e}")
    _OCR = None

def ocr_image_bytes(image_bytes: bytes, try_cjk_vertical: bool = False) -> str:
    """
    OCR an image using RapidOCR directly.
    - Only handles images (PNG/JPG/TIFF); PDFs not considered here.
    - try_cjk_vertical=True enables 4-way rotation to catch vertical JP/CJK.
    """
    if not image_bytes or len(image_bytes) == 0:
        logger.warning("Empty image data provided to OCR")
        return ""
    
    if _OCR is None:
        logger.error("OCR service not initialized - cannot process image")
        return ""
    
    try:
        preprocessed = _preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return ""
    
    if try_cjk_vertical:
        # Try all rotation variants for CJK vertical text
        variants = _rotate_image_variants(preprocessed)
        best_result = ""
        best_confidence = 0.0
        
        for deg, variant_bytes in variants:
            try:
                result = _OCR(variant_bytes)
                if result and len(result) > 0:
                    # Calculate average confidence - handle different result structures
                    confidences = []
                    text_parts = []
                    
                    # Handle the actual RapidOCR result structure: (detections_list, timing_info)
                    if len(result) >= 2 and isinstance(result[0], list):
                        # result[0] is the list of detections
                        detections = result[0]
                        
                        # Use spatial grouping for better text organization
                        grouped_text = _group_text_by_proximity(detections)
                        if grouped_text:
                            # Extract confidence scores for scoring
                            for detection in detections:
                                if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                                    confidence = detection[2]
                                    if isinstance(confidence, (int, float)):
                                        confidences.append(confidence)
                            
                            # Use grouped text instead of concatenated text
                            text_result = grouped_text
                        else:
                            # Fallback to simple concatenation
                            for detection in detections:
                                if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                                    # Each detection: [bbox, text, confidence]
                                    bbox, text, confidence = detection[0], detection[1], detection[2]
                                    if isinstance(confidence, (int, float)) and isinstance(text, str) and text.strip():
                                        confidences.append(confidence)
                                        text_parts.append(text.strip())
                            text_result = " ".join(text_parts).strip()
                    else:
                        # Fallback to old parsing method
                        for item in result:
                            if isinstance(item, (list, tuple)):
                                if len(item) >= 3:
                                    # Standard format: [bbox, text, confidence]
                                    bbox, text, confidence = item[0], item[1], item[2]
                                    if isinstance(confidence, (int, float)) and isinstance(text, str):
                                        confidences.append(confidence)
                                        text_parts.append(text)
                                elif len(item) == 2:
                                    # Alternative format: [text, confidence] or [bbox, text]
                                    if isinstance(item[1], (int, float)):
                                        # [text, confidence] format
                                        text, confidence = item[0], item[1]
                                        if isinstance(text, str):
                                            confidences.append(confidence)
                                            text_parts.append(text)
                                    else:
                                        # [bbox, text] format
                                        text = item[1]
                                        if isinstance(text, str):
                                            text_parts.append(text)
                                elif len(item) == 1:
                                    # Single item: just text
                                    text = item[0]
                                    if isinstance(text, str):
                                        text_parts.append(text)
                            elif isinstance(item, dict):
                                # Dictionary format
                                if 'confidence' in item and isinstance(item['confidence'], (int, float)):
                                    confidences.append(item['confidence'])
                                if 'text' in item:
                                    text_parts.append(str(item['text']))
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    text_result = " ".join(text_parts).strip()
                    
                    # Calculate a coherence score based on text quality
                    coherence_score = 0
                    if text_result:
                        # Bonus for longer text (more content detected)
                        coherence_score += len(text_result) * 0.1
                        # Bonus for common words that suggest coherent text
                        common_words = ['play', 'download', 'rating', 'total', 'average', 'hit', 'endless', 'kpop']
                        for word in common_words:
                            if word.lower() in text_result.lower():
                                coherence_score += 10
                        # Bonus for numbers (ratings, downloads)
                        if any(char.isdigit() for char in text_result):
                            coherence_score += 5
                        # Penalty for very short results
                        if len(text_result) < 10:
                            coherence_score -= 5
                    
                    # Combined score: confidence + coherence
                    combined_score = avg_confidence * 100 + coherence_score
                    
                    # Only consider results with actual text content (not just coordinates)
                    if (combined_score > best_confidence and 
                        text_result and 
                        len(text_result) > 3 and  # Minimum text length
                        not text_result.startswith('[[')):  # Avoid bounding box coordinates
                        best_confidence = combined_score
                        best_result = text_result
                        
            except Exception as e:
                logger.warning(f"OCR failed for rotation {deg}°: {e}")
                continue
        
        if best_result:
            logger.info(f"✅ OCR with rotation found text: '{best_result[:100]}{'...' if len(best_result) > 100 else ''}'")
            return best_result
        else:
            logger.info("⚠️ No text found with rotation variants, trying without rotation...")
            # Fallback to non-rotation OCR
            try:
                result = _OCR(preprocessed)
                if result and len(result) > 0:
                    text_parts = []
                    for item in result:
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            text = item[1]
                            if isinstance(text, str):
                                text_parts.append(text)
                        elif isinstance(item, dict) and 'text' in item:
                            text_parts.append(str(item['text']))
                    fallback_result = " ".join(text_parts).strip()
                    if fallback_result:
                        logger.info(f"✅ OCR without rotation found text: '{fallback_result[:100]}{'...' if len(fallback_result) > 100 else ''}'")
                        return fallback_result
            except Exception as e:
                logger.warning(f"Fallback OCR also failed: {e}")
            
            # Try a completely different approach as final fallback
            logger.info("🔄 Trying alternative OCR approach...")
            try:
                # Try with the original image without preprocessing
                logger.info("🔍 Trying OCR on original image without preprocessing...")
                original_result = _OCR(image_bytes)
                logger.info(f"📝 Original image OCR result: {original_result}")
                
                if original_result and len(original_result) > 0:
                    text_parts = []
                    # Handle the actual RapidOCR result structure: (detections_list, timing_info)
                    if len(original_result) >= 2 and isinstance(original_result[0], list):
                        # result[0] is the list of detections
                        detections = original_result[0]
                        for detection in detections:
                            if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                                # Each detection: [bbox, text, confidence]
                                bbox, text, confidence = detection[0], detection[1], detection[2]
                                if isinstance(text, str) and text.strip():
                                    text_parts.append(text.strip())
                    else:
                        # Fallback to old parsing method
                        for item in original_result:
                            if isinstance(item, (list, tuple)):
                                if len(item) >= 3:
                                    # Standard format: [bbox, text, confidence]
                                    text = item[1]
                                    if isinstance(text, str) and text.strip():
                                        text_parts.append(text)
                                elif len(item) == 2:
                                    # Alternative format: [text, confidence] or [bbox, text]
                                    if isinstance(item[1], (int, float)):
                                        # [text, confidence] format
                                        text = item[0]
                                        if isinstance(text, str) and text.strip():
                                            text_parts.append(text)
                                    else:
                                        # [bbox, text] format
                                        text = item[1]
                                        if isinstance(text, str) and text.strip():
                                            text_parts.append(text)
                                elif len(item) == 1:
                                    # Single item: just text
                                    text = item[0]
                                    if isinstance(text, str) and text.strip():
                                        text_parts.append(text)
                            elif isinstance(item, dict) and 'text' in item:
                                text = str(item['text'])
                                if text.strip():
                                    text_parts.append(text)
                    if text_parts:
                        result_text = " ".join(text_parts).strip()
                        return result_text
                
            except Exception as e:
                logger.warning(f"Alternative OCR approach failed: {e}")
            
            return ""
    else:
        # Standard OCR without rotation
        try:
            result = _OCR(preprocessed)
            if result and len(result) > 0:
                text_parts = []
                # Handle the actual RapidOCR result structure: (detections_list, timing_info)
                if len(result) >= 2 and isinstance(result[0], list):
                    # result[0] is the list of detections
                    detections = result[0]
                    for detection in detections:
                        if isinstance(detection, (list, tuple)) and len(detection) >= 3:
                            # Each detection: [bbox, text, confidence]
                            bbox, text, confidence = detection[0], detection[1], detection[2]
                            if isinstance(text, str) and text.strip():
                                text_parts.append(text.strip())
                else:
                    # Fallback to old parsing method
                    for item in result:
                        if isinstance(item, (list, tuple)):
                            if len(item) >= 3:
                                # Standard format: [bbox, text, confidence]
                                text = item[1]
                                if isinstance(text, str):
                                    text_parts.append(text)
                            elif len(item) == 2:
                                # Alternative format: [text, confidence] or [bbox, text]
                                if isinstance(item[1], (int, float)):
                                    # [text, confidence] format
                                    text = item[0]
                                    if isinstance(text, str):
                                        text_parts.append(text)
                                else:
                                    # [bbox, text] format
                                    text = item[1]
                                    if isinstance(text, str):
                                        text_parts.append(text)
                            elif len(item) == 1:
                                # Single item: just text
                                text = item[0]
                                if isinstance(text, str):
                                    text_parts.append(text)
                        elif isinstance(item, dict) and 'text' in item:
                            text_parts.append(str(item['text']))
                return " ".join(text_parts)
            return ""
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            logger.debug(f"OCR error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(f"OCR traceback: {traceback.format_exc()}")
            return ""

def get_ocr_service():
    """Get the OCR service instance."""
    return _OCR

# ---------- FastAPI Integration ----------
async def extract_text_from_image(
    base64_data: str, 
    filename: str = "unknown", 
    try_cjk_vertical: bool = False
) -> Dict[str, Any]:
    """
    Extract text from base64-encoded image data.
    """
    try:
        # Decode base64 data
        image_bytes = base64.b64decode(base64_data)
        
        # Run OCR
        text = ocr_image_bytes(image_bytes, try_cjk_vertical=try_cjk_vertical)
        
        return {
            "text": text,
            "filename": filename,
            "file_size": len(image_bytes),
            "cjk_rotation_used": try_cjk_vertical,
            "extraction_method": "rapidocr_direct",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return {
            "text": "",
            "filename": filename,
            "file_size": 0,
            "cjk_rotation_used": try_cjk_vertical,
            "extraction_method": "rapidocr_direct",
            "success": False,
            "error": str(e)
        }

# ---------- Legacy compatibility functions ----------
def ocr_image(image_bytes: bytes) -> str:
    """Legacy function name for compatibility."""
    return ocr_image_bytes(image_bytes, try_cjk_vertical=False)

def preprocess_image(src_bytes: bytes) -> bytes:
    """Legacy function name for compatibility."""
    return _preprocess_image(src_bytes)

def test_ocr_service() -> bool:
    """Test the OCR service with a simple generated image"""
    try:
        import io
        from PIL import Image, ImageDraw, ImageFont
        
        if _OCR is None:
            logger.error("OCR service not initialized - cannot run test")
            return False
        
        # Create a simple test image with text
        img = Image.new("RGB", (300, 100), "white")
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw text with high contrast
        draw.text((20, 30), "Test OCR", fill="black", font=font)
        draw.text((20, 60), "Hello World", fill="black", font=font)
        
        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, "PNG")
        test_bytes = buf.getvalue()
        
        # Test OCR
        result = ocr_image_bytes(test_bytes, try_cjk_vertical=False)
        
        if result and ("Test" in result or "Hello" in result):
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"OCR service test error: {e}")
        return False