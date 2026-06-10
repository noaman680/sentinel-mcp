import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mcp_agent.agent import stream_agent

app = FastAPI(title="Databricks MCP Agent Streaming API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/api/v1/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streams agent state updates via Server-Sent Events (SSE).
    Allows the UI to render "Agent is searching..." or "Executing Python..." dynamically.
    """
    async def event_generator():
        try:
            async for update in stream_agent(request.message, session_id=request.session_id):
                # Sanitize the output for the network stream
                safe_update = {k: str(v) for k, v in update.items()}
                
                # Yield standard SSE format
                yield f"data: {json.dumps(safe_update)}\n\n"
                
                # Briefly yield control back to the event loop
                await asyncio.sleep(0.01)
                
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")