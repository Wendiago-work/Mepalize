# Localized Translator

An AI-powered translation system with cultural context awareness, built using RAG (Retrieval-Augmented Generation) architecture with MongoDB and Qdrant vector database integration.

## Architecture Overview

### System Components

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Cloud Services│
│   (React/Vite)  │◄──►│   (FastAPI)     │◄──►│   MongoDB Atlas │
│                 │    │                 │    │   Qdrant Cloud  │
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
- **MongoDB Atlas**: Cloud database for structured data (free tier available)
- **Qdrant Cloud**: Cloud vector database for embeddings (free tier available)
- **Google Gemini API Key**: For AI translation services (free tier available)

### Environment Variables

#### Backend Configuration

```bash
# =============================================================================
# REQUIRED CLOUD CREDENTIALS (Only 4 variables needed!)
# =============================================================================

# MongoDB Atlas 
TRANSLATION_MONGO_CONNECTION_STRING=your_mongodb_atlas_connection_string

# Qdrant Cloud 
TRANSLATION_QDRANT_CLOUD_URL=your_qdrant_cloud_url
TRANSLATION_QDRANT_CLOUD_API_KEY=your_qdrant_api_key

# Google Gemini 
TRANSLATION_GEMINI_API_KEY=your_gemini_api_key

TRANSLATION_MONGO_DATABASE=LocalizationDB
TRANSLATION_QDRANT_COLLECTION_NAME=translation_embeddings
```

#### Frontend Configuration

```bash
# API Configuration
VITE_API_BASE=http://localhost:8000
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

#### Quick Start (Docker Compose)

```bash
# 1. Clone repository
git clone <repository-url>
cd localized-translator

# 2. Set up cloud credentials
cp .env.example .env
# Edit .env with your MongoDB Atlas and Qdrant Cloud credentials

# 3. Start the application
docker-compose up -d

# 4. Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
```

#### Local Development

**Backend Setup**:

```bash
cd backend
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv
pip install uv
uv pip install -e .

# Set up cloud credentials
cp .env.example .env
# Edit .env with your MongoDB Atlas and Qdrant Cloud credentials

# Run the application
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend Setup**:

```bash
cd frontend
npm install
npm run dev
```

**Required Cloud Services** (All free tiers available):

- **MongoDB Atlas**: Free forever tier with 512MB storage
- **Qdrant Cloud**: Free tier with 1GB storage
- **Google Gemini**: Free tier with generous usage limits

#### Development with Public URLs

Use the provided script for local development with public URLs:

```bash
# Make sure both frontend and backend are running
# Frontend: npm run dev (port 5173)
# Backend: uvicorn src.main:app --reload (port 8000)

# Run the tunnel script
./run-local.sh
```

This will create public URLs for both services using LocalTunnel.

### Quick Setup Guide

#### 1. Get Free Cloud Credentials

**MongoDB Atlas** (Free forever tier):

1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free cluster
3. Get your connection string

**Qdrant Cloud** (Free tier available):

1. Go to [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a free cluster
3. Get your URL and API key

**Google Gemini** (Free tier available):

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create an API key

#### 2. Deploy

```bash
# Clone and setup
git clone <repository-url>
cd localized-translator
cp .env.example .env

# Edit .env with your cloud credentials
# TRANSLATION_MONGO_CONNECTION_STRING=your_mongodb_atlas_string
# TRANSLATION_QDRANT_CLOUD_URL=your_qdrant_url
# TRANSLATION_QDRANT_CLOUD_API_KEY=your_qdrant_key
# TRANSLATION_GEMINI_API_KEY=your_gemini_key

# Start the application
docker-compose up -d
```

That's it! Your translation service will be running at `http://localhost:5173`

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

- `POST /chat/translate` - Main translation endpoint with OCR support
- `POST /runs/translate` - Background translation processing (requires X-Project-Token header)
- `GET /runs/{run_id}` - Check translation status

#### Chat Sessions

- `GET /chat/sessions` - List all active chat sessions
- `GET /chat/sessions/{session_id}` - Get specific chat session information

#### Context Management

- `GET /context/cultural-notes/{language}` - Get cultural notes for specific language
- `GET /context/style-guide/{domain}` - Get style guide for specific domain
- `GET /context/domains` - List available domains
- `GET /context/languages` - List supported languages

#### OCR Services

- `POST /ocr` - Extract text from uploaded images (multipart/form-data)
- `POST /ocr/base64` - Extract text from base64-encoded images

#### System Health & Monitoring

- `GET /health` - System health check with service status
- `GET /data/summary` - Database statistics (MongoDB + Qdrant)
- `GET /orchestrator/stats` - LangChain orchestrator pipeline statistics

#### Root

- `GET /` - Serves static frontend or API documentation

### Data Ingestion

The system includes comprehensive data ingestion capabilities for populating the knowledge base:

#### Available Data Sources

The `backend/data/` directory contains pre-processed CSV files with localization data:

- **Cultural Notes**: `Cultural Notes - All Languages.csv` - Cultural context for different languages
- **Style Guides**: `FRi - Localization Ref For AI - Style Guide.csv` - Writing style guidelines
- **Translation Memory**: `JPi - Localization Ref For AI - Translation Memory.csv` - Historical translations
- **Glossaries**: Domain-specific glossaries for different game types:
  - `Glossary - Game - Music - *.csv` (EN, FR, JA)
  - `Glossary - Game - Casual - *.csv` (EN, FR, JA)
  - `Glossary - Entertainment - *.csv` (EN, FR, JA)

#### Data Ingestion Pipeline

The system provides several ingestion scripts:

```bash
# Import data to MongoDB
cd backend
python import_data_to_mongodb.py

# Import data to Qdrant vector database
python ingest_qdrant_only.py

# Run complete data ingestion pipeline
python data_ingestion_pipeline.py

# Migrate to Qdrant Cloud
python migrate_to_qdrant_cloud.py
```

#### Supported Data Formats

- **CSV**: Primary format for structured localization data
- **PDF, DOCX, HTML**: Document processing with Docling
- **JSON**: Direct data import
- **Images**: OCR text extraction with RapidOCR

### Model Management

The system uses several ML models that are automatically downloaded and cached:

- **Embedding Model**: `intfloat/multilingual-e5-large` (1024 dimensions)
- **Sparse Model**: `prithvida/Splade_PP_en_v1`
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **OCR Models**: RapidOCR with ONNX Runtime

Models are cached locally and downloaded on first use.

## Development

### Project Structure

```text
localized-translator/
├── backend/
│   ├── src/
│   │   ├── config/          # Configuration management
│   │   ├── core/            # Core utilities, types, and model management
│   │   ├── database/        # MongoDB and Qdrant clients
│   │   ├── services/        # Business logic services (OCR, Gemini, RAG, etc.)
│   │   ├── prompts/         # LLM prompt templates
│   │   └── main.py          # FastAPI application
│   ├── data/               # CSV data files for ingestion
│   │   ├── Cultural Notes - All Languages.csv
│   │   ├── Glossary - Game - Music - *.csv
│   │   ├── FRi - Localization Ref For AI - *.csv
│   │   └── JPi - Localization Ref For AI - *.csv
│   ├── audit/              # Translation audit logs
│   ├── logs/               # Application logs
│   ├── docker-compose.yml  # Container orchestration
│   ├── Dockerfile         # Backend container
│   ├── pyproject.toml     # Python dependencies (uv-based)
│   ├── requirements.txt   # Alternative pip requirements
│   ├── data_ingestion_pipeline.py  # Data ingestion scripts
│   ├── import_data_to_mongodb.py   # MongoDB data import
│   ├── ingest_qdrant_only.py      # Qdrant data import
│   └── migrate_to_qdrant_cloud.py # Cloud migration script
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── context/    # Cultural notes, style guide, prompt preview
│   │   │   ├── translation/# Translation form, results, selectors
│   │   │   └── ui/         # Reusable UI components
│   │   ├── services/       # API client
│   │   ├── lib/           # Utility functions
│   │   └── App.tsx        # Main application
│   ├── public/            # Static assets
│   ├── dist/              # Built frontend
│   ├── package.json       # Node.js dependencies
│   └── vite.config.ts     # Vite configuration
├── audit/                 # Shared audit directory
├── run-local.sh          # Local development with tunnels
└── README.md             # This file
```

### Key Features

- **Multi-modal Translation**: Text and image input support with OCR
- **Cultural Context**: Domain-specific style guides and cultural notes
- **RAG Integration**: Retrieval-augmented generation for better translations
- **Real-time Preview**: See the complete prompt sent to the LLM
- **Safety Filtering**: Content moderation and safety checks
- **OCR Support**: Extract text from images in multiple languages (including CJK)
- **Hybrid Search**: Combines vector and keyword search for better retrieval
- **Transparency**: Full prompt visibility for debugging and understanding
- **Audit Logging**: Complete translation pipeline audit trails
- **Cloud-Native**: Fully cloud-based with MongoDB Atlas and Qdrant Cloud
- **Modern UI**: React 19 with Radix UI components and Tailwind CSS

## Current Status

### ✅ Implemented Features

- Complete translation pipeline with MongoDB + Qdrant
- OCR integration with RapidOCR
- Cultural context and style guide integration
- Real-time prompt preview
- Audit logging system
- Docker containerization
- Local development with tunnel support
- Data ingestion pipeline for CSV files

### 🚧 Development Status

- **Backend**: Fully functional with all core services
- **Frontend**: Complete React application with modern UI
- **Data**: Pre-loaded with localization data for testing
- **Deployment**: Docker-ready with cloud-native architecture

### 📊 Data Available

- Cultural notes for multiple languages
- Style guides for different domains
- Translation memory and glossaries
- Game-specific terminology (Music, Casual, Entertainment)

This architecture provides a robust, scalable foundation for AI-powered translation with cultural awareness and transparency.
