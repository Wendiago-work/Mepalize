# Localized Translator

An AI-powered prompt generation system with cultural context awareness, built using RAG (Retrieval-Augmented Generation) architecture with MongoDB and Chroma DB cloud integration.

## Architecture Overview

### System Components

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Cloud Services│
│   (React/Vite)  │◄──►│   (FastAPI)     │◄──►│   MongoDB Atlas │
│                 │    │                 │    │   Chroma DB     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI Services   │
                       │   Prompt Gen    │
                       │   OCR (Gemini)  │
                       └─────────────────┘
```

### Core Architecture

#### 1. **Frontend (React + TypeScript)**

- **Framework**: React 19 with Vite
- **UI Library**: Radix UI components with Tailwind CSS
- **State Management**: React hooks
- **Key Features**:
  - Multi-language prompt generation interface
  - Image upload with OCR support
  - Real-time prompt preview
  - Cultural context display
  - Style guide integration

#### 2. **Backend API (FastAPI + Python)**

- **Framework**: FastAPI with Uvicorn
- **Architecture**: LangChain-based orchestration
- **Key Services**:
  - **LangChain Orchestrator**: Coordinates the entire prompt generation pipeline
  - **Chroma Retrieval Service**: Handles vector search across translation memory and glossaries
  - **Context Service**: Manages MongoDB data (style guides, cultural notes)
  - **OCR Service**: Extracts text from images using Google Gemini AI
  - **Prompt Service**: Generates comprehensive prompts with context

#### 3. **Database Layer**

**MongoDB**:

- Stores structured data (style guides, cultural notes)
- Collections: `style_guides`, `cultural_notes`
- Used for domain-specific context and cultural adaptation

**Chroma DB Cloud**:

- Stores embeddings for semantic search using native embedding functions
- Collections: `translation_memory`, `glossaries`
- Handles translation memory and glossaries as separate collections
- Supports semantic vector search with metadata filtering

#### 4. **AI/ML Services**

**Prompt Generation**:

- Generates comprehensive prompts with context from Chroma DB and MongoDB
- Combines translation memory, glossaries, style guides, and cultural notes
- No external LLM required - outputs ready-to-use prompts

**Embedding Models**:

- Chroma DB native embedding functions (no local models needed)
- Automatic embedding generation and storage
- Optimized for semantic search and retrieval

**OCR Engine**:

- Google Gemini AI multimodal model
- Cloud-based text extraction from images
- Supports multiple languages and formats
- High accuracy with advanced AI capabilities
- No local model downloads or storage required
- Faster processing and better accuracy than traditional OCR
- Automatic language detection and text formatting

## Data Flow

### Prompt Generation Pipeline

1. **Input Processing**:
   - User submits text and optional images
   - OCR extracts text from images
   - Combined text is processed for prompt generation

2. **Context Retrieval**:
   - Chroma DB performs semantic search on translation memory and glossaries
   - MongoDB provides style guides and cultural notes
   - Context is filtered by language pair and domain

3. **Prompt Construction**:
   - RAG context is assembled from multiple sources
   - Translation memory entries are formatted for reference
   - Glossary terms are included with definitions
   - Style guides and cultural notes are integrated

4. **Prompt Generation**:
   - Comprehensive prompt is generated with all context
   - Includes source text, target language, domain, and cultural considerations
   - Ready-to-use prompt for external LLM consumption

5. **Response Formatting**:
   - Generated prompt is returned to frontend
   - Context statistics are included for transparency
   - Full prompt can be copied for external use

## Deployment Requirements

### Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **MongoDB Atlas**: Cloud database for structured data (free tier available)
- **Chroma DB Cloud**: Cloud vector database for embeddings (free tier available)
- **Google Gemini API**: For OCR text extraction from images
- **No external LLM required**: System generates prompts for external use

### Environment Variables

#### Backend Configuration

```bash
# =============================================================================
# REQUIRED CLOUD CREDENTIALS (Only 5 variables needed!)
# =============================================================================

# MongoDB Atlas 
MONGO_CONNECTION_STRING=your_mongodb_atlas_connection_string
MONGO_DATABASE=LocalizationDB

# Chroma DB Cloud 
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database

# Google Gemini API (for OCR)
GEMINI_API_KEY=your_gemini_api_key
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

#### Chroma DB Collections

```python
# Translation Memory Collection
{
  "id": "uuid",
  "content": "Hello world",  # Source text (embedded)
  "metadata": {
    "target_text": "こんにちは世界",
    "source_language": "en",
    "target_language": "ja",
    "domain": "general",
    "date_created": "2023-07-01"
  }
}

# Glossaries Collection
{
  "id": "uuid",
  "content": "user interface: The space where interactions between humans and machines occur",  # Term + definition (embedded)
  "metadata": {
    "term": "user interface",
    "definition": "The space where interactions between humans and machines occur",
    "source_language": "en",
    "target_language": "ja",
    "domain": "general",
    "date_created": "2023-07-01"
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
# Edit .env with your MongoDB Atlas and Chroma DB Cloud credentials

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
# Edit .env with your MongoDB Atlas and Chroma DB Cloud credentials

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
- **Chroma DB Cloud**: Free tier with 1GB storage
- **Google Gemini API**: Free tier with generous usage limits
- **No external LLM required**: System generates prompts for external use

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

**Chroma DB Cloud** (Free tier available):

1. Go to [Chroma DB Cloud](https://www.trychroma.com/)
2. Create a free account
3. Get your API key, tenant, and database name

**Google Gemini API** (Free tier available):

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a free account
3. Generate an API key for Gemini

#### 2. Deploy

```bash
# Clone and setup
git clone <repository-url>
cd localized-translator
cp .env.example .env

# Edit .env with your cloud credentials
# MONGO_CONNECTION_STRING=your_mongodb_atlas_string
# MONGO_DATABASE=LocalizationDB
# CHROMA_API_KEY=your_chroma_api_key
# CHROMA_TENANT=your_chroma_tenant
# CHROMA_DATABASE=your_chroma_database
# GEMINI_API_KEY=your_gemini_api_key

# Start the application
docker-compose up -d
```

That's it! Your prompt generation service will be running at `http://localhost:5173`

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
- Track prompt generation quality metrics
- Database performance monitoring

#### Scaling

- Use load balancers for multiple API instances
- Consider database sharding for large datasets
- Implement caching layers
- Monitor and optimize Chroma DB performance

### API Endpoints

#### Core Prompt Generation

- `POST /chat/translate` - Main prompt generation endpoint with OCR support
- `POST /runs/translate` - Background prompt generation processing (requires X-Project-Token header)
- `GET /runs/{run_id}` - Check prompt generation status

#### Chat Sessions

- `GET /chat/sessions` - List all active chat sessions
- `GET /chat/sessions/{session_id}` - Get specific chat session information

#### Context Management

- `GET /context/cultural-notes/{language}` - Get cultural notes for specific language
- `GET /context/style-guide/{domain}` - Get style guide for specific domain
- `GET /context/domains` - List available domains
- `GET /context/languages` - List supported languages


#### System Health & Monitoring

- `GET /health` - System health check with service status
- `GET /data/summary` - Database statistics (MongoDB + Chroma DB)
- `GET /orchestrator/stats` - LangChain orchestrator pipeline statistics

#### Root

- `GET /` - Serves static frontend or API documentation

### Data Ingestion

The system includes data ingestion capabilities for populating the knowledge base:

#### Available Data Sources

The `backend/data/` directory contains CSV files with localization data:

- **Translation Memory**: `Optimized_TM_Music-Game_JA_v3.csv` - Historical translations
- **Glossaries**: `Optimized_Glossary_Music-Game_JA_v3.csv` - Domain-specific terminology

#### Data Ingestion Pipeline

The system provides ingestion scripts for Chroma DB:

```bash
# Ingest data to Chroma DB collections
cd backend
python ingest_chroma_data.py
```

#### Supported Data Formats

- **CSV**: Primary format for structured localization data
- **Images**: OCR text extraction with RapidOCR

### Model Management

The system uses Chroma DB's native embedding functions:

- **No local models required**: Chroma DB handles all embeddings natively
- **OCR Models**: Google Gemini AI for cloud-based image processing
- **Automatic embedding**: Chroma DB generates and manages embeddings automatically

## Development

### Project Structure

```text
localized-translator/
├── backend/
│   ├── src/
│   │   ├── config/          # Configuration management
│   │   ├── core/            # Core utilities, types, and model management
│   │   ├── database/        # MongoDB and Chroma DB clients
│   │   ├── services/        # Business logic services (OCR, RAG, etc.)
│   │   ├── prompts/         # LLM prompt templates
│   │   └── main.py          # FastAPI application
│   ├── data/               # CSV data files for ingestion
│   │   ├── Optimized_TM_Music-Game_JA_v3.csv
│   │   └── Optimized_Glossary_Music-Game_JA_v3.csv
│   ├── audit/              # Prompt generation audit logs
│   ├── logs/               # Application logs
│   ├── docker-compose.yml  # Container orchestration
│   ├── Dockerfile         # Backend container
│   ├── pyproject.toml     # Python dependencies (uv-based)
│   ├── requirements.txt   # Alternative pip requirements
│   └── ingest_chroma_data.py      # Chroma DB data import
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── context/    # Cultural notes, style guide, prompt preview
│   │   │   ├── translation/# Prompt generation form, results, selectors
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

- **Multi-modal Prompt Generation**: Text and image input support with OCR
- **Cultural Context**: Domain-specific style guides and cultural notes
- **RAG Integration**: Retrieval-augmented generation for comprehensive prompts
- **Real-time Preview**: See the complete generated prompt
- **OCR Support**: Extract text from images using Google Gemini AI
- **Semantic Search**: Chroma DB vector search for relevant context
- **Transparency**: Full prompt visibility for debugging and understanding
- **Audit Logging**: Complete prompt generation pipeline audit trails
- **Cloud-Native**: Fully cloud-based with MongoDB Atlas, Chroma DB Cloud, and Google Gemini
- **Modern UI**: React 19 with Radix UI components and Tailwind CSS
- **No External LLM**: Generates ready-to-use prompts for external consumption

## Current Status

### ✅ Implemented Features

- Complete prompt generation pipeline with MongoDB + Chroma DB
- OCR integration with Google Gemini AI
- Cultural context and style guide integration
- Real-time prompt preview
- Audit logging system
- Docker containerization
- Local development with tunnel support
- Data ingestion pipeline for CSV files

### 🚧 Development Status

- **Backend**: Fully functional with all core services
- **Frontend**: Complete React application with modern UI
- **Data**: Pre-loaded with translation memory and glossaries for testing
- **Deployment**: Docker-ready with cloud-native architecture

### 📊 Data Available

- Cultural notes for multiple languages
- Style guides for different domains
- Translation memory and glossaries
- Game-specific terminology (Music domain)

This architecture provides a robust, scalable foundation for AI-powered prompt generation with cultural awareness and transparency.
