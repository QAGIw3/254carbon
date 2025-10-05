"""
AI Service API Routers

This package contains the FastAPI routers that expose the service’s HTTP and
WebSocket APIs. Each router groups related endpoints and delegates to an
underlying engine:

- `copilot`: conversational chat and streaming interactions
- `nlp`: natural language query parsing and insight generation
- `regtech`: regulatory tracking and compliance analytics

Routers are mounted by `main.py` and are designed to be independent and
composable. Keep request/response models as the contract boundary.
"""
