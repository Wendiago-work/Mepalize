# Localized Translator

An AI-powered translation system with cultural context awareness, built using RAG (Retrieval-Augmented Generation) architecture with MongoDB and Qdrant vector database integration.

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Databases     │
│   (React/Vite)  │◄──►│   (FastAPI)     │◄──►│   MongoDB       │
│                 │    │                 │    │   Qdrant        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI Services   │
                       │   Gemini Pro    │
                       │   OCR (RapidOCR)│
                       └─────────────────┘
```

### Core Architecture

#### 1. **Frontend (React + TypeScript)**
- **Framework**: React 19 with Vite
- **UI Library**: Radix UI components with Tailwind CSS
- **State Management**: React hooks
- **Key Features**:
  - Multi-language translation interface
  - Image upload with OCR support
  - Real-time prompt preview
  - Cultural context display
  - Style guide integration

#### 2. **Backend API (FastAPI + Python)**
- **Framework**: FastAPI with Uvicorn
- **Architecture**: LangChain-based orchestration
- **Key Services**:
  - **Translation Orchestrator**: Coordinates the entire translation pipeline
  - **Gemini Service**: Handles LLM interactions with Google Gemini Pro
  - **Hybrid Retrieval Service**: Combines vector and BM25 search
  - **Context Service**: Manages MongoDB data (style guides, cultural notes)
  - **OCR Service**: Extracts text from images using RapidOCR
  - **Prompt Service**: Manages translation prompts and templates

#### 3. **Database Layer**

**MongoDB**:
- Stores structured data (style guides, cultural notes)
- Collections: `style_guides`, `cultural_notes`
- Used for domain-specific context and cultural adaptation

**Qdrant Vector Database**:
- Stores embeddings for semantic search
- Collections: `translation_embeddings`
- Handles translation memory and glossaries
- Supports hybrid search (vector + BM25)

#### 4. **AI/ML Services**

**Google Gemini Pro**:
- Primary translation engine
- Handles multi-modal input (text + images)
- Safety filtering and content moderation

**Embedding Models**:
- `intfloat/multilingual-e5-large`: Dense embeddings
- `prithvida/Splade_PP_en_v1`: Sparse embeddings
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: Reranking

**OCR Engine**:
- RapidOCR with ONNX Runtime
- Supports multiple languages including CJK
- Handles vertical text detection

## Data Flow

### Translation Pipeline

1. **Input Processing**:
   - User submits text and optional images
   - OCR extracts text from images
   - Combined text is processed for translation

2. **Context Retrieval**:
   - Query embedding is generated
   - Hybrid search retrieves relevant translations and glossaries
   - MongoDB provides style guides and cultural notes

3. **Prompt Construction**:
   - RAG context is assembled
   - User context takes priority over retrieved context
   - Complete prompt is sent to Gemini Pro

4. **Translation Generation**:
   - Gemini Pro processes the full context
   - Response includes translation, reasoning, and cultural notes
   - Safety filters ensure appropriate content

5. **Response Formatting**:
   - Translation result is structured
   - Full prompt is included for transparency
   - Response is returned to frontend

## Deployment Requirements

### Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **Python 3.13+**: For backend development
- **Node.js 18+**: For frontend development
- **MongoDB**: Database for structured data
- **Qdrant**: Vector database for embeddings

### Environment Variables

#### Backend Configuration

```bash
# Database Configuration
TRANSLATION_MONGO_CONNECTION_STRING=mongodb://localhost:27017
TRANSLATION_MONGO_DATABASE=LocalizationDB

# Qdrant Configuration
TRANSLATION_QDRANT_HOST=localhost
TRANSLATION_QDRANT_PORT=6333
TRANSLATION_QDRANT_COLLECTION_NAME=translation_embeddings

# Gemini AI Configuration
TRANSLATION_GEMINI_API_KEY=your_gemini_api_key_here
TRANSLATION_GEMINI_MODEL=gemini-2.5-pro
TRANSLATION_GEMINI_TEMPERATURE=0.1
TRANSLATION_GEMINI_MAX_TOKENS=4000

# Safety Configuration
TRANSLATION_ENABLE_SAFETY_FILTERS=true
TRANSLATION_SAFETY_THRESHOLD=MEDIUM_AND_ABOVE

# Performance Configuration
UVICORN_WORKERS=2
```

#### Frontend Configuration

```bash
# API Configuration
VITE_API_BASE=http://localhost:8000
VITE_PROJECT_TOKEN=dev-token
```

### Database Setup

#### MongoDB Collections

```javascript
// Style Guides Collection
{
  "_id": ObjectId,
  "domain": "gaming",
  "style_guide": {
    "tone": "casual",
    "guidelines": ["Use simple language", "Avoid technical jargon"]
  },
  "created_at": ISODate,
  "updated_at": ISODate
}

// Cultural Notes Collection
{
  "_id": ObjectId,
  "language": "ja",
  "domain": "gaming",
  "cultural_note": "Japanese gamers prefer honorifics in character names",
  "created_at": ISODate
}
```

#### Qdrant Collections

```python
# Translation Memory Collection
{
  "id": "uuid",
  "vector": [0.1, 0.2, ...],  # 1024-dimensional embedding
  "payload": {
    "dataset": "translation_memory",
    "content_type": "translation_pair",
    "translation_source": "Hello world",
    "translation_target": "こんにちは世界",
    "source_language": "en",
    "target_language": "ja",
    "domain": "general"
  }
}

# Glossaries Collection
{
  "id": "uuid",
  "vector": [0.1, 0.2, ...],
  "payload": {
    "dataset": "glossaries",
    "content_type": "glossary_entry",
    "term": "user interface",
    "translation": "ユーザーインターフェース",
    "definition": "The space where interactions between humans and machines occur"
  }
}
```

### Deployment Options

#### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd localized-translator

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# Services will be available at:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8000
# - Qdrant: http://localhost:6333
```

#### Option 2: Manual Deployment

**Backend Setup**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Frontend Setup**:
```bash
cd frontend
npm install
npm run dev
```

**Database Setup**:
```bash
# Start MongoDB
mongod --dbpath /path/to/data

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:latest
```

### Production Considerations

#### Security
- Set strong API keys and database credentials
- Enable HTTPS in production
- Configure CORS properly
- Use environment-specific configuration files

#### Performance
- Configure appropriate worker counts
- Set up database connection pooling
- Enable model caching
- Monitor resource usage

#### Monitoring
- Set up logging aggregation
- Monitor API response times
- Track translation quality metrics
- Database performance monitoring

#### Scaling
- Use load balancers for multiple API instances
- Consider database sharding for large datasets
- Implement caching layers
- Monitor and optimize embedding generation

### API Endpoints

#### Core Translation
- `POST /chat/translate` - Main translation endpoint
- `POST /runs/translate` - Background translation processing
- `GET /runs/{run_id}` - Check translation status

#### Context Management
- `GET /context/cultural-notes/{language}` - Get cultural notes
- `GET /context/style-guide/{domain}` - Get style guide
- `GET /context/domains` - List available domains
- `GET /context/languages` - List supported languages

#### OCR Services
- `POST /ocr` - Extract text from uploaded images
- `POST /ocr/base64` - Extract text from base64 images

#### System Health
- `GET /health` - System health check
- `GET /data/summary` - Database statistics
- `GET /orchestrator/stats` - Pipeline statistics

### Data Ingestion

The system supports various data formats for knowledge base population:

- **PDF, DOCX, HTML**: Document processing with Docling
- **CSV, XLSX**: Structured data import
- **JSON**: Direct data import
- **Images**: OCR text extraction

### Model Management

The system uses several ML models that are automatically downloaded and cached:

- **Embedding Model**: `intfloat/multilingual-e5-large` (1024 dimensions)
- **Sparse Model**: `prithvida/Splade_PP_en_v1`
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **OCR Models**: RapidOCR with ONNX Runtime

Models are cached locally and downloaded on first use.

## Development

### Project Structure

```
localized-translator/
├── backend/
│   ├── src/
│   │   ├── config/          # Configuration management
│   │   ├── core/            # Core utilities and types
│   │   ├── database/        # Database clients
│   │   ├── services/        # Business logic services
│   │   ├── prompts/         # LLM prompt templates
│   │   └── main.py          # FastAPI application
│   ├── docker-compose.yml   # Container orchestration
│   ├── Dockerfile          # Backend container
│   └── pyproject.toml      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API client
│   │   └── App.tsx         # Main application
│   └── package.json        # Node.js dependencies
└── README.md              # This file
```

### Key Features

- **Multi-modal Translation**: Text and image input support
- **Cultural Context**: Domain-specific style guides and cultural notes
- **RAG Integration**: Retrieval-augmented generation for better translations
- **Real-time Preview**: See the complete prompt sent to the LLM
- **Safety Filtering**: Content moderation and safety checks
- **OCR Support**: Extract text from images in multiple languages
- **Hybrid Search**: Combines vector and keyword search for better retrieval
- **Transparency**: Full prompt visibility for debugging and understanding

This architecture provides a robust, scalable foundation for AI-powered translation with cultural awareness and transparency.
