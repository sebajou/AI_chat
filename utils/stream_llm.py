import json
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Store for conversation histories (in-memory, per session)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

async def handle_streaming_chat(user_message: str, websocket, session_id: str = "default"):
    """
    Handle a streaming chat interaction with LangChain.

    Args:
        user_message: The user's input message
        websocket: WebSocket connection to send chunks
        session_id: Unique identifier for this conversation session
    """
    # Initialize the LLM
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.7,
        api_key=os.getenv("MISTRAL_API_KEY"),
        streaming=True
    )

    # Get session history
    history = get_session_history(session_id)

    # Add user message to history
    history.add_user_message(user_message)

    # Get all messages for the LLM
    langchain_messages = history.messages

    # Send start signal
    await websocket.send_text(json.dumps({"type": "start", "content": ""}))

    # Stream the response
    full_response = ""
    try:
        async for chunk in llm.astream(langchain_messages):
            if chunk.content:
                full_response += chunk.content
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": chunk.content
                }))

        # Send end signal
        await websocket.send_text(json.dumps({"type": "end", "content": ""}))

        # Add assistant response to history
        history.add_ai_message(full_response)

    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "content": f"Error: {str(e)}"
        }))
        raise