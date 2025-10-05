"""
AI Copilot API Router
"""
import logging
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel

from copilot.engine import AICopilot, ModelProvider, Language

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])

# Initialize copilot engine
copilot = AICopilot()


class ConversationRequest(BaseModel):
    """Conversation request."""
    query: str
    conversation_id: Optional[str] = None
    language: Language = Language.ENGLISH
    model: ModelProvider = ModelProvider.OPENAI_GPT4
    context: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Conversation response."""
    conversation_id: str
    response: str
    sources: List[Dict]
    suggested_actions: List[str]
    data_citations: List[str]
    confidence: float
    language: Language


@router.post("/chat", response_model=ConversationResponse)
async def chat_with_copilot(request: ConversationRequest):
    """Chat with AI Copilot."""
    try:
        logger.info(f"Chat request: {request.query[:50]}...")
        
        result = await copilot.chat(
            request.query,
            request.conversation_id,
            request.language,
            request.model,
            request.context
        )
        
        return ConversationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            result = await copilot.chat(
                query=data,
                conversation_id=conversation_id,
                language=Language.ENGLISH,
                model=ModelProvider.OPENAI_GPT4,
            )
            
            await websocket.send_json(result)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history."""
    if conversation_id in copilot.conversations:
        del copilot.conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}
    return {"status": "not_found", "conversation_id": conversation_id}

