"""
AI Service package

This package aggregates LLM-powered capabilities used across the platform:
- Copilot: conversational assistant for market analysis and workflows
- NLP: natural language query parsing and insight generation
- RegTech: compliance monitoring, reporting, and penalty risk assessment

The subpackages expose engines used by the FastAPI routers in `routers/`.
All stateful components are kept minimal by default (integration points are
stubbed for local development) and should be wired to production services
like Redis, ClickHouse, and external LLM providers via environment variables.
"""
