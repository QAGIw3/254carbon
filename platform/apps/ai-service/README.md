# AI Service

**Version:** 2.0.0  
**Port:** 8020  
**Status:** Operational

## Overview

The AI Service is a consolidated platform that unifies all LLM-based AI capabilities previously scattered across 3 separate services. This consolidation enables shared LLM client connections, unified vector database integration, and common conversation management.

## Consolidated Services

This service consolidates the following previous services:

1. **ai-copilot** (Port 8017) → Copilot Module
2. **nlp-service** (Port 8014) → NLP Module
3. **regtech-ai** → RegTech Module

## Architecture

```
ai-service/
├── copilot/                  # Conversational AI
│   ├── engine.py
│   ├── rag.py
│   └── conversation_manager.py
├── nlp/                      # NLP query understanding
│   ├── query_parser.py
│   ├── entity_extractor.py
│   ├── insight_generator.py
│   └── report_generator.py
├── regtech/                  # Regulatory compliance AI
│   ├── compliance_engine.py
│   ├── regulation_tracker.py
│   └── penalty_assessor.py
└── routers/                  # API routers
    ├── copilot.py
    ├── nlp.py
    └── regtech.py
```

## Key Features

### AI Copilot
- Conversational AI for energy market intelligence
- Multi-model support (GPT-4, Claude, Mistral, Local LLMs)
- RAG (Retrieval Augmented Generation)
- Multi-language support (English, Spanish, Portuguese, etc.)
- WebSocket support for real-time chat
- Conversation history management

### NLP Query Understanding
- Natural language query parsing
- Entity extraction (markets, dates, metrics)
- SQL query generation from natural language
- API call generation
- Automated market insights
- Report generation

### RegTech Compliance AI
- Automated regulation tracking (60+ jurisdictions)
- NLP-based rule extraction
- Compliance gap analysis
- Automated report generation
- Penalty risk assessment
- Cross-jurisdictional mapping

## API Endpoints

### Copilot
- `POST /api/v1/copilot/chat` - Chat with AI Copilot
- `WebSocket /api/v1/copilot/ws/{conversation_id}` - Real-time chat
- `DELETE /api/v1/copilot/conversation/{id}` - Clear conversation

### NLP
- `POST /api/v1/nlp/query` - Parse natural language query
- `POST /api/v1/nlp/insights` - Generate market insights
- `POST /api/v1/nlp/reports` - Generate automated reports

### RegTech
- `GET /api/v1/regtech/regulations` - Get regulatory updates
- `GET /api/v1/regtech/compliance/gaps` - Analyze compliance gaps
- `POST /api/v1/regtech/reports/generate` - Generate compliance report
- `GET /api/v1/regtech/penalty-risk/{entity_id}` - Assess penalty risk

## Shared Infrastructure

- **LLM Clients**: Shared OpenAI/Anthropic client connections
- **Vector Database**: Unified ChromaDB for RAG across all features
- **Conversation Management**: Common conversation history storage
- **Rate Limiting**: Shared rate limiting for API calls
- **Embedding Models**: Single embedding model instance

## Environment Variables

```bash
OPENAI_API_KEY=<your-key>
ANTHROPIC_API_KEY=<your-key>
POSTGRES_HOST=postgresql
POSTGRES_PORT=5432
```

## Running the Service

### Local Development
```bash
cd platform/apps/ai-service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8020
```

### Docker
```bash
docker build -t 254carbon/ai-service:latest .
docker run -p 8020:8020 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  254carbon/ai-service:latest
```

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

## LLM Provider Configuration

### OpenAI
```python
model = ModelProvider.OPENAI_GPT4
# Uses GPT-4 Turbo for analysis
```

### Anthropic Claude
```python
model = ModelProvider.ANTHROPIC_CLAUDE
# Uses Claude 3 for detailed analysis
```

### Local LLM
```python
model = ModelProvider.LOCAL_LLAMA
# Uses local LLaMA for development
```

## Benefits of Consolidation

1. **Operational**: 3 services → 1 service (67% reduction)
2. **Performance**: Connection pooling, reduced network hops
3. **Development**: Single codebase, shared testing infrastructure
4. **Cost**: Reduced infrastructure, optimized API usage

## WebSocket Usage

```javascript
const ws = new WebSocket('ws://ai-service:8020/api/v1/copilot/ws/my-conversation');

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log(response.response);
};

ws.send('What drove prices up in PJM yesterday?');
```

## Migration Notes

The service maintains API compatibility with previous services. All existing endpoints remain accessible with the same request/response formats.

## Health Check

```bash
curl http://ai-service:8020/health
```

## Documentation

- API Docs: http://ai-service:8020/docs
- OpenAPI: http://ai-service:8020/openapi.json

## Supported Languages

- English (`en`)
- Spanish (`es`)
- Portuguese (`pt`)
- French (`fr`)
- German (`de`)
- Mandarin (`zh`)

## Jurisdictional Coverage (RegTech)

- FERC (US Federal Energy Regulatory Commission)
- NERC (North American Electric Reliability Corporation)
- REMIT (EU Regulation on Energy Market Integrity)
- FCA (UK Financial Conduct Authority)
- ACER (EU Agency for the Cooperation of Energy Regulators)
- CERC (India Central Electricity Regulatory Commission)
- NEA (China National Energy Administration)

