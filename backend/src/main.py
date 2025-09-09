from fastapi import FastAPI, Header, HTTPException, status, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Optional, Union, Dict, Any
import os
import json
from datetime import datetime, timezone
import uuid

# Import types from centralized location
from .core.types import Attachment, TranslateReq, RunStatus, RunPass, RunFail

# Import services
from .services.hybrid_retrieval_service import HybridRetrievalService
from .services.gemini_service import GeminiService
from .services.prompt_service import PromptService
from .services.context_service import ContextService, create_context_service
from .services.langchain_orchestrator import LangChainOrchestrator, TranslationRequest, TranslationResult
from .services.ocr_service import get_ocr_service, ocr_image_bytes, extract_text_from_image
from .database.mongodb_client import create_mongodb_client
from .database.qdrant_client import QdrantVectorStore
from .core.model_manager import get_model_manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    global mongodb_client, qdrant_client, context_service, hybrid_retrieval_service, gemini_service, prompt_service, orchestrator, model_manager
    
    print("🚀 Starting Localized Translator with MongoDB + Qdrant Architecture...")
    
    try:
        # Initialize model manager first
        model_manager = get_model_manager()
        await model_manager.initialize()
        
        # Initialize MongoDB
        from .config.config import get_settings
        settings = get_settings()
        mongodb_client = create_mongodb_client(
            connection_string=settings.mongo_connection_string,
            database_name=settings.mongo_database
        )
        await mongodb_client.initialize()
        
        # Initialize Qdrant
        qdrant_client = QdrantVectorStore(collection_name=settings.qdrant_collection_name)
        await qdrant_client.initialize()
        await qdrant_client.setup_localization_indexes()
        
        # Initialize services
        context_service = create_context_service(mongodb_client)
        hybrid_retrieval_service = HybridRetrievalService(qdrant_client)
        hybrid_retrieval_service.initialize_services()
        gemini_service = GeminiService()
        prompt_service = PromptService()
        
        # Initialize LangChain orchestrator
        orchestrator = LangChainOrchestrator(
            qdrant_service=hybrid_retrieval_service,
            gemini_service=gemini_service,
            prompt_service=prompt_service,
            context_service=context_service
        )
        
        # Initialize OCR service and warm up models
        ocr_service = get_ocr_service()
        await warmup_ocr()
        
        # Test OCR service
        from .services.ocr_service import test_ocr_service
        ocr_test_result = test_ocr_service()
        if not ocr_test_result:
            print("⚠️ OCR service test failed - OCR may not work properly")
        
        print("✅ All services initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    print("🛑 Shutting down services...")
    if mongodb_client:
        await mongodb_client.close()
    if qdrant_client:
        await qdrant_client.close()
    if model_manager:
        await model_manager.close()

app = FastAPI(title="Localized Translator MVP API", lifespan=lifespan)

# CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",  
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_origin_regex=r"http://.*:(8000|5173|3000)"  # Allow common dev ports
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global services
mongodb_client = None
qdrant_client = None
context_service = None
hybrid_retrieval_service = None
gemini_service = None
prompt_service = None
orchestrator = None
model_manager = None

AUDIT_DIR = os.environ.get("AUDIT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "audit"))
AUDIT_DIR = os.path.abspath(AUDIT_DIR)
os.makedirs(AUDIT_DIR, exist_ok=True)


def generate_run_id() -> str:
    return str(uuid.uuid4())


async def warmup_ocr():
    """Warm up OCR models at startup to avoid first-request delay"""
    try:
        import io
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (220, 60), "white")
        d = ImageDraw.Draw(im)
        d.text((10, 20), "Warmup")
        buf = io.BytesIO()
        im.save(buf, "PNG")
        buf.seek(0)
        _ = ocr_image_bytes(buf.read(), try_cjk_vertical=False)
        print("✅ OCR models warmed up successfully")
    except Exception as e:
        print(f"⚠️ OCR warmup failed (non-critical): {e}")


def write_audit_stub(run_id: str, payload: dict) -> None:
    record = {
        "runId": run_id,
        "status": "received",
        "receivedAt": datetime.now(timezone.utc).isoformat(),
        "request": payload,
        "history": [
            {"status": "received", "at": datetime.now(timezone.utc).isoformat()}
        ],
    }
    with open(os.path.join(AUDIT_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def _audit_path(run_id: str) -> str:
    return os.path.join(AUDIT_DIR, f"{run_id}.json")


def _load_audit(run_id: str) -> dict:
    with open(_audit_path(run_id), "r", encoding="utf-8") as f:
        return json.load(f)


def _save_audit(run_id: str, record: dict) -> None:
    with open(_audit_path(run_id), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def advance_status(run_id: str, new_status: str, details: Optional[Dict] = None) -> None:
    try:
        rec = _load_audit(run_id)
    except FileNotFoundError:
        return
    rec["status"] = new_status
    history = rec.get("history", [])
    history_entry = {"status": new_status, "at": datetime.now(timezone.utc).isoformat()}
    
    if details:
        history_entry.update(details)
    
    history.append(history_entry)
    rec["history"] = history
    _save_audit(run_id, rec)

async def process_translation_pipeline(run_id: str, request: TranslateReq) -> None:
    """Process translation using the complete MongoDB + Qdrant pipeline"""
    try:
        print(f"\n🔄 ===== TRANSLATION PIPELINE START =====")
        print(f"📝 Run ID: {run_id}")
        print(f"📄 Source Text: '{request.text}'")
        print(f"🌍 Source Language: {request.source_language}")
        print(f"🎯 Target Language: {request.target_language}")
        print(f"🏷️  Domain: {request.domain}")
        print(f"📝 Context Notes: {request.context_notes or 'None'}")
        print(f"🕐 Started at: {datetime.now(timezone.utc).isoformat()}")
        print(f"==========================================\n")
        
        advance_status(run_id, "processing", {"message": "Processing attachments and executing translation pipeline"})
        
        # Process image attachments with OCR
        processed_text = request.text
        ocr_context = ""
        
        if request.attachments:
            print(f"📎 Processing {len(request.attachments)} attachments...")
            for attachment in request.attachments:
                if attachment.type == "image" and attachment.base64_data:
                    print(f"🔍 Processing image: {attachment.filename or 'unknown'}")
                    try:
                        # Extract text from image using OCR
                        ocr_result = await extract_text_from_image(
                            attachment.base64_data,
                            attachment.filename or "unknown",
                            try_cjk_vertical=True  # Enable CJK vertical text detection
                        )
                        
                        if ocr_result["success"] and ocr_result["text"]:
                            extracted_text = ocr_result["text"].strip()
                            print(f"✅ OCR extracted text: '{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}'")
                            ocr_context += f"\n{extracted_text}"
                        else:
                            error_msg = ocr_result.get('error', 'Unknown error')
                            print(f"⚠️ OCR failed for {attachment.filename or 'image'}: {error_msg}")
                            print(f"   OCR result: {ocr_result}")
                    except Exception as e:
                        print(f"❌ OCR error for {attachment.filename or 'image'}: {e}")
                        import traceback
                        print(f"   Error details: {traceback.format_exc()}")
                        continue
            
            # Combine original text with OCR context
            if ocr_context:
                processed_text = f"{request.text}{ocr_context}"
                print(f"📝 Combined text with OCR context: {len(processed_text)} characters")
        
        # Convert to LangChain TranslationRequest
        langchain_request = TranslationRequest(
            text=processed_text,
            source_language=request.source_language,
            target_language=request.target_language,
            domain=request.domain,
            attachments=request.attachments,
            context_notes=request.context_notes
        )
        
        # Execute translation pipeline
        result = await orchestrator.execute_pipeline(langchain_request)
        
        # Update audit with results
        advance_status(run_id, "completed", {"message": "Translation completed successfully"})
        
        rec = _load_audit(run_id)
        rec["status"] = "pass"
        rec["finalText"] = result.translated_text
        rec["full_prompt"] = result.full_prompt  # Add full_prompt to audit record
        rec["execution_time"] = result.execution_time
        rec["context"] = result.rag_context
        rec["history"].append({
            "status": "pass",
            "at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "mongodb_qdrant_orchestrator",
            "execution_time": result.execution_time
        })
        _save_audit(run_id, rec)
        
        print(f"\n✅ ===== TRANSLATION COMPLETED =====")
        print(f"📝 Run ID: {run_id}")
        print(f"📄 Translation: '{result.translated_text}'")
        print(f"⏱️  Execution Time: {result.execution_time:.3f}s")
        print(f"🕐 Completed at: {datetime.now(timezone.utc).isoformat()}")
        print(f"==========================================\n")
        
    except Exception as e:
        print(f"\n❌ ===== TRANSLATION FAILED =====")
        print(f"📝 Run ID: {run_id}")
        print(f"❌ Error: {e}")
        print(f"🕐 Failed at: {datetime.now(timezone.utc).isoformat()}")
        print(f"==========================================\n")
        
        # Create failure response
        rec = _load_audit(run_id)
        rec["status"] = "fail"
        rec["error"] = str(e)
        
        history = rec.get("history", [])
        history.append({
            "status": "fail", 
            "at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "pipeline": "mongodb_qdrant_orchestrator"
        })
        rec["history"] = history
        
        _save_audit(run_id, rec)


@app.get("/")
async def root():
    """Serve the main interface"""
    static_file = os.path.join(static_dir, "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    else:
        return {"message": "Localized Translator MVP API", "docs": "/docs"}


@app.post("/runs/translate", status_code=status.HTTP_202_ACCEPTED)
async def post_translate(
    req: TranslateReq,
    background_tasks: BackgroundTasks,
    x_project_token: Optional[str] = Header(default=None, alias="X-Project-Token"),
):
    """
    Background translation using LangChain orchestrator with OCR support
    
    Supports image attachments in the request. Images will be processed with OCR
    to extract text, which will be combined with the main text for translation.
    
    Example request body:
    {
        "text": "Translate this text",
        "source_language": "en",
        "target_language": "ja",
        "domain": "general",
        "attachments": [
            {
                "type": "image",
                "base64_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "filename": "screenshot.png",
                "mime_type": "image/png"
            }
        ]
    }
    """
    if not x_project_token:
        raise HTTPException(status_code=401, detail="Missing X-Project-Token")
    
    if not req.text:
        raise HTTPException(status_code=400, detail="Text is required for translation")
    
    run_id = generate_run_id()
    write_audit_stub(run_id, req.model_dump())
    
    # Add background task with translation pipeline
    background_tasks.add_task(process_translation_pipeline, run_id, req)
    
    return {"runId": run_id, "status": "accepted"}


@app.post("/chat/translate")
async def chat_translate(request: TranslateReq):
    """
    Translation endpoint with MongoDB + Qdrant context and OCR support
    
    Supports image attachments in the request. Images will be processed with OCR
    to extract text, which will be combined with the main text for translation.
    """
    session_id = str(uuid.uuid4())
    
    print(f"\n🔄 TRANSLATION REQUEST")
    print(f"==========================================")
    print(f"Session ID: {session_id}")
    print(f"Text: '{request.text}'")
    print(f"Source: {request.source_language} → Target: {request.target_language}")
    print(f"Domain: {request.domain}")
    print(f"Context: {request.context_notes or 'None'}")
    print(f"==========================================\n")
    
    try:
        # Process image attachments with OCR
        processed_text = request.text
        ocr_context = ""
        
        if request.attachments:
            print(f"📎 Processing {len(request.attachments)} attachments...")
            for attachment in request.attachments:
                if attachment.type == "image" and attachment.base64_data:
                    print(f"🔍 Processing image: {attachment.filename or 'unknown'}")
                    try:
                        # Extract text from image using OCR
                        ocr_result = await extract_text_from_image(
                            attachment.base64_data,
                            attachment.filename or "unknown",
                            try_cjk_vertical=True  # Enable CJK vertical text detection
                        )
                        
                        if ocr_result["success"] and ocr_result["text"]:
                            extracted_text = ocr_result["text"].strip()
                            print(f"✅ OCR extracted text: '{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}'")
                            ocr_context += f"\n{extracted_text}"
                        else:
                            error_msg = ocr_result.get('error', 'Unknown error')
                            print(f"⚠️ OCR failed for {attachment.filename or 'image'}: {error_msg}")
                            print(f"   OCR result: {ocr_result}")
                    except Exception as e:
                        print(f"❌ OCR error for {attachment.filename or 'image'}: {e}")
                        import traceback
                        print(f"   Error details: {traceback.format_exc()}")
                        continue
            
            # Combine original text with OCR context
            if ocr_context:
                processed_text = f"{request.text}{ocr_context}"
                print(f"📝 Combined text with OCR context: {len(processed_text)} characters")
        
        # Convert to LangChain TranslationRequest
        langchain_request = TranslationRequest(
            text=processed_text,
            source_language=request.source_language,
            target_language=request.target_language,
            domain=request.domain,
            attachments=request.attachments,
            context_notes=request.context_notes
        )
        
        # Execute translation pipeline
        result = await orchestrator.execute_pipeline(langchain_request)
        
        # Debug: Check if full_prompt is in result
        print(f"🔍 DEBUG: result type: {type(result)}")
        print(f"🔍 DEBUG: result attributes: {dir(result)}")
        print(f"🔍 DEBUG: hasattr full_prompt: {hasattr(result, 'full_prompt')}")
        if hasattr(result, 'full_prompt'):
            print(f"🔍 DEBUG: full_prompt length: {len(result.full_prompt)}")
            print(f"🔍 DEBUG: full_prompt preview: {result.full_prompt[:200]}...")
        
        # Format response
        translation_response = {
            "session_id": session_id,
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "translation": result.translated_text,
            "reasoning": result.reasoning,
            "cultural_notes": result.cultural_notes,
            "style_applied": result.style_applied,
            "domain_considerations": result.domain_considerations,
            "full_prompt": result.full_prompt,  # Add full_prompt for frontend display
            "rag_context": result.rag_context,
            "execution_time": result.execution_time,
            "pipeline": "mongodb_qdrant_orchestrator"
        }
        
        # Debug: Check final response
        print(f"🔍 DEBUG: translation_response keys: {list(translation_response.keys())}")
        print(f"🔍 DEBUG: full_prompt in response: {'full_prompt' in translation_response}")
        if 'full_prompt' in translation_response:
            print(f"🔍 DEBUG: full_prompt length in response: {len(translation_response['full_prompt'])}")
        
        print(f"   ✅ Translation completed!")
        print(f"   ⏱️  Execution time: {result.execution_time:.3f}s")
        print(f"   🎯 Context: {request.context_notes or 'Using RAG context'}")
        
        return translation_response
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "session_id": session_id,
            "error": True,
            "message": f"Translation failed: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get chat session information and history"""
    try:
        memory_path = _audit_path(f"chat_session_{session_id}")
        
        if not os.path.exists(memory_path):
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        with open(memory_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        
        return {
            "session_id": session_id,
            "session_info": session_data,
            "status": "active"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all active chat sessions"""
    try:
        audit_dir = os.path.dirname(_audit_path("dummy"))
        sessions = []
        
        if os.path.exists(audit_dir):
            for file in os.listdir(audit_dir):
                if file.startswith("chat_session_"):
                    session_id = file.replace("chat_session_", "").replace(".json", "")
                    sessions.append({
                        "session_id": session_id,
                        "file": file
                    })
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


# Run status models moved to core/types.py for better organization


@app.get("/runs/{run_id}", response_model=Union[RunStatus, RunPass, RunFail])
def get_run_status(run_id: str):
    path = _audit_path(run_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run not found")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    status_val = data.get("status", "received")
    
    if status_val == "pass":
        return {
            "status": "pass",
            "runId": data.get("runId", run_id),
            "finalText": data.get("finalText", ""),
            "execution_time": data.get("execution_time", 0.0),
            "context": data.get("context", {}),
        }
    
    if status_val == "fail":
        return {
            "status": "fail",
            "runId": data.get("runId", run_id),
            "error": data.get("error", "Unknown error"),
        }
    
    # Calculate progress for in-progress runs
    progress = None
    if status_val in ["retrieving", "generating", "reranking", "validating", "finalizing"]:
        status_order = ["received", "retrieving", "generating", "reranking", "validating", "finalizing"]
        try:
            current_index = status_order.index(status_val)
            progress = (current_index / (len(status_order) - 1)) * 100
        except ValueError:
            progress = 0
    
    return {
        "runId": data.get("runId", run_id), 
        "status": status_val, 
        "history": data.get("history"),
        "progress": progress
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), try_cjk_vertical: bool = Query(False)):
    """OCR endpoint for extracting text from images using Docling + RapidOCR"""
    try:
        data = await file.read()
        text = ocr_image_bytes(data, try_cjk_vertical=try_cjk_vertical)
        return {
            "text": text,
            "filename": file.filename,
            "file_size": len(data),
            "cjk_rotation_used": try_cjk_vertical,
            "extraction_method": "docling_rapidocr"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@app.post("/ocr/base64")
async def ocr_base64(
    base64_data: str,
    filename: str = "unknown",
    try_cjk_vertical: bool = Query(False)
):
    """OCR endpoint for base64-encoded images"""
    try:
        ocr_service = get_ocr_service()
        result = await ocr_service.extract_text_from_image(
            base64_data, 
            filename, 
            try_cjk_vertical=try_cjk_vertical
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if services are initialized
        if not all([mongodb_client, qdrant_client, context_service, hybrid_retrieval_service, gemini_service, orchestrator, model_manager]):
            return {"status": "unhealthy", "message": "Services not initialized"}
        
        # Get service stats
        mongodb_stats = await context_service.get_stats()
        qdrant_stats = await qdrant_client.get_collection_info()
        
        return {
            "status": "healthy",
            "message": "Localized Translator with MongoDB + Qdrant",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "mongodb_client": "initialized",
                "qdrant_client": "initialized", 
                "context_service": "initialized",
                "hybrid_retrieval_service": "initialized",
                "gemini_service": "initialized",
                "orchestrator": "initialized",
                "model_manager": "initialized"
            },
            "mongodb": mongodb_stats,
            "qdrant": qdrant_stats,
        }
        
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


@app.get("/data/summary")
async def get_data_summary():
    """Get summary of available data"""
    try:
        if not all([context_service, qdrant_client]):
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Get combined summary
        mongodb_summary = await context_service.get_stats()
        qdrant_summary = await qdrant_client.get_collection_info()
  
        return {
            "mongodb": mongodb_summary,
            "qdrant": qdrant_summary,
            "total_knowledge": {
                "mongodb_documents": mongodb_summary.get("collections", {}).get("style_guides", 0) + 
                                   mongodb_summary.get("collections", {}).get("cultural_notes", 0),
                "qdrant_documents": qdrant_summary.get("points_count", 0),
                "collection_name": qdrant_summary.get("name", "knowledge_base"),
                "indexed_vectors": qdrant_summary.get("indexed_vectors_count", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data summary: {str(e)}")


@app.get("/orchestrator/stats")
async def get_orchestrator_stats():
    """Get LangChain orchestrator statistics and pipeline health"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        stats = await orchestrator.get_pipeline_stats()
        return {
            "status": "success",
            "orchestrator_stats": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get orchestrator stats: {str(e)}")


@app.get("/context/cultural-notes/{language}")
async def get_cultural_notes(language: str):
    """Get cultural notes for a specific language"""
    try:
        if not context_service:
            raise HTTPException(status_code=503, detail="Context service not initialized")
        
        notes = await context_service.get_cultural_notes(language)
        return {
            "language": language,
            "cultural_notes": notes,
            "count": len(notes),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cultural notes: {str(e)}")


@app.get("/context/style-guide/{domain}")
async def get_style_guide(domain: str):
    """Get style guide for a specific domain"""
    try:
        if not context_service:
            raise HTTPException(status_code=503, detail="Context service not initialized")
        
        style_guide = await context_service.get_style_guide(domain)
        return {
            "domain": domain,
            "style_guide": style_guide,
            "found": style_guide is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get style guide: {str(e)}")


@app.get("/context/domains")
async def get_available_domains():
    """Get list of available domains"""
    try:
        if not context_service:
            raise HTTPException(status_code=503, detail="Context service not initialized")
        
        domains = await context_service.get_available_domains()
        return {
            "domains": domains,
            "count": len(domains),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get domains: {str(e)}")


@app.get("/context/languages")
async def get_available_languages():
    """Get list of available languages"""
    try:
        if not context_service:
            raise HTTPException(status_code=503, detail="Context service not initialized")
        
        languages = await context_service.get_available_languages()
        return {
            "languages": languages,
            "count": len(languages),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get languages: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


