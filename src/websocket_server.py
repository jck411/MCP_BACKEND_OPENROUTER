"""
WebSocket Server for MCP Platform

This module provides a thin communication layer between the frontend and chat service.
It handles WebSocket connections and message routing only.
"""

import contextlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.chat import ChatOrchestrator
from src.chat.models import ChatMessage
from src.config import Configuration
from src.history import ChatRepository
from src.history.models import ChatEvent
from src.history.repository import SavedSessionsRepository

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from typing import Any as LLMClient
    from typing import Any as MCPClient
else:
    # At runtime, these will be the actual types passed in
    LLMClient = object
    MCPClient = object

logger = logging.getLogger(__name__)


# Pydantic models for WebSocket message validation
class ChatPayload(BaseModel):
    """Payload for chat messages with validation."""

    text: str
    streaming: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSocketMessage(BaseModel):
    """WebSocket message structure with validation."""

    request_id: str
    payload: ChatPayload
    message_type: str = "chat"


class WebSocketResponse(BaseModel):
    """WebSocket response structure."""

    request_id: str
    status: str  # "processing", "chunk", "completed", "init", "error"
    chunk: dict[str, Any] = Field(default_factory=dict)


class WebSocketServer:
    """
    Pure WebSocket communication server.

    This class only handles:
    - WebSocket connections
    - Message parsing and routing
    - Response streaming

    All business logic is delegated to ChatService.
    """

    def __init__(
        self,
        clients: list["MCPClient"],
        llm_client: "LLMClient",
        config: dict[str, Any],
        repo: ChatRepository,
        configuration: Configuration,
    ):
        service_config = ChatOrchestrator.ChatOrchestratorConfig(
            clients=clients,
            llm_client=llm_client,
            config=config,
            repo=repo,
            configuration=configuration,
        )
        self.chat_service = ChatOrchestrator(service_config)
        self.repo = repo
        self.config = config
        self.configuration = configuration
        self.app = self._create_app()
        self.active_connections: list[WebSocket] = []
        # Store conversation id per socket
        self.conversation_ids: dict[WebSocket, str] = {}

    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI app."""
        app = FastAPI(title="MCP WebSocket Chat Server")
        router = APIRouter()

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket):  # type: ignore
            await self._handle_websocket_connection(websocket)

        @app.get("/")
        async def root():  # type: ignore
            return {"message": "MCP WebSocket Chat Server"}

        @app.get("/health")
        async def health():  # type: ignore
            return {"status": "healthy"}

        # Session management endpoints (AutoPersistRepo)
        @router.post("/sessions/{conversation_id}/save")
        async def save_session(
            conversation_id: str, name: str | None = None
        ) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
            if isinstance(self.repo, SavedSessionsRepository):
                saved_id: str = await self.repo.save_session(conversation_id, name)
                return {"saved_id": saved_id}
            return {"error": "Saved sessions are disabled"}

        @router.get("/sessions")
        async def list_sessions() -> list[dict[str, Any]]:  # type: ignore
            if isinstance(self.repo, SavedSessionsRepository):
                sessions: list[dict[str, Any]] = await self.repo.list_saved_sessions()
                return sessions
            return []

        @router.get("/sessions/{saved_id}")
        async def load_session(saved_id: str) -> list[dict[str, Any]] | dict[str, str]:  # type: ignore
            if isinstance(self.repo, SavedSessionsRepository):
                events: list[ChatEvent] = await self.repo.load_saved_session(saved_id)
                # Return as plain dicts for API consumers
                return [ev.model_dump() for ev in events]
            return {"error": "Saved sessions are disabled"}

        # Prevent static analyzers from marking route handlers as unused
        __keep_for_pyright = (save_session, list_sessions, load_session)
        del __keep_for_pyright

        app.include_router(router)

        return app

    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle a WebSocket connection."""
        await self._connect_websocket(websocket)

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Handle message based on action
                if message_data.get("action") == "chat":
                    await self._handle_chat_message(websocket, message_data)
                elif message_data.get("action") == "clear_session":
                    await self._handle_clear_session(websocket, message_data)
                else:
                    # Unknown message format
                    logger.warning(f"Unknown message format: {message_data}")
                    await self._send_error_response(
                        websocket,
                        message_data.get("request_id", "unknown"),
                        (
                            "Unknown message format. "
                            "Expected 'action': 'chat' or 'clear_session'"
                        ),
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            with contextlib.suppress(Exception):
                await self._send_error_response(
                    websocket,
                    locals().get("message_data", {}).get("request_id", "unknown"),
                    f"Server error: {e!s}",
                )
        finally:
            self._disconnect_websocket(websocket)

    async def _handle_chat_message(
        self, websocket: WebSocket, message_data: dict[str, Any]
    ):
        """Handle a chat message from the frontend with Pydantic validation."""
        try:
            # Validate message structure using Pydantic
            ws_message = WebSocketMessage.model_validate(message_data)
        except ValidationError as e:
            await self._send_error_response(
                websocket,
                message_data.get("request_id", "unknown"),
                f"Invalid message format: {e}",
            )
            return

        request_id = ws_message.request_id
        payload = ws_message.payload
        user_message = payload.text

        # Check streaming configuration - FAIL FAST approach
        service_config = self.config.get("chat", {}).get("service", {})
        streaming_config = service_config.get("streaming", {})

        if payload.streaming is not None:
            # Client explicitly set streaming preference - use it
            streaming = payload.streaming
        elif streaming_config.get("enabled") is not None:
            # Use configured default - must be explicitly set
            streaming = streaming_config["enabled"]
        else:
            # FAIL FAST: No streaming configuration found
            await self._send_error_response(
                websocket,
                request_id,
                "Streaming configuration missing. "
                "Set 'chat.service.streaming.enabled' in config.yaml "
                "or specify 'streaming: true/false' in payload.",
            )
            return

        logger.info(f"Processing message with streaming={streaming}")
        logger.info(f"Received chat message: {user_message[:50]}...")

        try:
            # Send processing status
            response = WebSocketResponse(
                request_id=request_id,
                status="processing",
                chunk={"metadata": {"user_message": user_message}},
            )
            await websocket.send_text(response.model_dump_json())

            # get or assign conversation_id
            conversation_id = self.conversation_ids.get(websocket)
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                self.conversation_ids[websocket] = conversation_id

            if streaming:
                # Streaming mode: real-time chunks
                await self._handle_streaming_chat(
                    websocket, request_id, conversation_id, user_message
                )
            else:
                # Non-streaming mode: single final assistant message
                await self._handle_non_streaming_chat(
                    websocket, request_id, conversation_id, user_message
                )

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            await self._send_error_response(websocket, request_id, str(e))

    async def _handle_clear_session(
        self, websocket: WebSocket, message_data: dict[str, Any]
    ):
        """Handle a clear session request from the frontend."""
        request_id = message_data.get("request_id", str(uuid.uuid4()))

        try:
            # Get current conversation_id
            old_conversation_id = self.conversation_ids.get(websocket, "")

            # AutoPersist uses periodic full wipe logic
            # Returns True when a full wipe occurs
            full_wipe_occurred = await self.repo.handle_clear_session()
            if full_wipe_occurred:
                retention_conf = (
                    self.config.get("chat", {})
                    .get("storage", {})
                    .get("persistence", {})
                    .get("retention", {})
                )
                max_sessions = retention_conf.get("max_sessions")
                logger.info(
                    "Full history wipe occurred%s",
                    (
                        f" on {max_sessions}th clear session"
                        if isinstance(max_sessions, int)
                        else ""
                    ),
                )

            # Only create new conversation_id if full wipe occurred
            # Otherwise keep the same conversation_id to maintain LLM memory
            if full_wipe_occurred:
                new_conversation_id = str(uuid.uuid4())
                self.conversation_ids[websocket] = new_conversation_id
                logger.info(
                    "Full wipe: Session cleared: %s -> %s",
                    old_conversation_id,
                    new_conversation_id,
                )
            else:
                new_conversation_id = old_conversation_id  # Keep same conversation
                counter = getattr(self.repo, "_clear_session_counter", "?")
                max_sessions = getattr(self.repo, "max_sessions", "N/A")
                logger.info(
                    f"UI clear only: Session {old_conversation_id} "
                    f"(counter: {counter}/{max_sessions})"
                )

            # Send success response
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "completed",
                        "chunk": {
                            "type": "session_cleared",
                            "metadata": {
                                "new_conversation_id": new_conversation_id,
                                "old_conversation_id": old_conversation_id,
                                "full_wipe_occurred": full_wipe_occurred,
                            },
                        },
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            await self._send_error_response(
                websocket, request_id, f"Clear session failed: {e}"
            )

    async def _send_error_response(
        self, websocket: WebSocket, request_id: str, error_message: str
    ):
        """Send error response using Pydantic model."""
        response = WebSocketResponse(
            request_id=request_id, status="error", chunk={"error": error_message}
        )
        await websocket.send_text(response.model_dump_json())

    async def _handle_streaming_chat(
        self,
        websocket: WebSocket,
        request_id: str,
        conversation_id: str,
        user_message: str,
    ):
        """Handle streaming chat response."""
        # Stream response chunks using new signature (no external history needed)
        async for chat_message in self.chat_service.process_message(
            conversation_id, user_message, request_id
        ):
            await self._send_chat_response(websocket, request_id, chat_message)

        # Send completion signal
        await websocket.send_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "status": "completed",
                    "chunk": {},
                }
            )
        )

    async def _handle_non_streaming_chat(
        self,
        websocket: WebSocket,
        request_id: str,
        conversation_id: str,
        user_message: str,
    ):
        """Handle non-streaming chat response."""
        assistant_ev = await self.chat_service.chat_once(
            conversation_id=conversation_id,
            user_msg=user_message,
            request_id=request_id,
        )

        await websocket.send_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "status": "chunk",
                    "chunk": {
                        "type": "text",
                        "data": assistant_ev.content,
                        "metadata": {},
                    },
                }
            )
        )

        await websocket.send_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "status": "completed",
                    "chunk": {},
                }
            )
        )

    async def _send_chat_response(
        self, websocket: WebSocket, request_id: str, chat_message: ChatMessage
    ):
        """Send a chat response to the frontend."""
        logger.info(
            "Sending WebSocket message: "
            f"type={chat_message.type}, "
            f"content={chat_message.content[:50]}..."
        )

        # Convert chat service message to frontend format
        if chat_message.type == "text":
            # Only send text messages that aren't tool results
            if not chat_message.metadata.get("tool_result"):
                await websocket.send_text(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "status": "chunk",
                            "chunk": {
                                "type": "text",
                                "data": chat_message.content,
                                "metadata": chat_message.metadata,
                            },
                        }
                    )
                )

        elif chat_message.type == "tool_execution":
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "processing",
                        "chunk": {
                            "type": "tool_execution",
                            "data": chat_message.content,
                            "metadata": chat_message.metadata,
                        },
                    }
                )
            )

        elif chat_message.type == "error":
            await websocket.send_text(
                json.dumps(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "chunk": {
                            "error": chat_message.content,
                            "metadata": chat_message.metadata,
                        },
                    }
                )
            )

    async def _load_previous_conversation(self, websocket: WebSocket):
        """Load and send previous conversation history to the frontend."""
        try:
            # Get all conversations from the repository
            all_conversations = await self.repo.list_conversations()

            if not all_conversations:
                # No previous conversations, start fresh
                conversation_id = str(uuid.uuid4())
                self.conversation_ids[websocket] = conversation_id
                logger.info(f"Starting new conversation: {conversation_id}")
                return

            # Get the most recent conversation (last one in the list)
            # list_conversations() returns oldest -> newest by last activity
            recent_conversation_id = all_conversations[-1]
            self.conversation_ids[websocket] = recent_conversation_id

            # Load conversation history
            history = await self.repo.get_conversation_history(
                recent_conversation_id, limit=50
            )

            if not history:
                logger.info(f"Resuming empty conversation: {recent_conversation_id}")
                return

            logger.info(
                f"Loading {len(history)} messages from conversation: "
                f"{recent_conversation_id}"
            )

            # Send history to frontend as a special message
            history_messages: list[dict[str, Any]] = []

            def _collapse_double(text: str) -> str:
                # If content is an exact duplication (XX), collapse to X
                if not text:
                    return text
                n = len(text)
                if n % 2 == 0:
                    half = n // 2
                    if text[:half] == text[half:]:
                        return text[:half]
                return text

            for event in history:
                if event.type == "user_message":
                    history_messages.append(
                        {
                            "role": "user",
                            "content": str(event.content or ""),
                            "timestamp": event.created_at.isoformat(),
                        }
                    )
                elif event.type == "assistant_message":
                    history_messages.append(
                        {
                            "role": "assistant",
                            "content": _collapse_double(str(event.content or "")),
                            "timestamp": event.created_at.isoformat(),
                        }
                    )

            if history_messages:
                # Send history as a special initialization message
                await websocket.send_text(
                    json.dumps(
                        {
                            "request_id": str(uuid.uuid4()),
                            "status": "init",
                            "chunk": {
                                "type": "conversation_history",
                                "data": history_messages,
                                "metadata": {
                                    "conversation_id": recent_conversation_id,
                                    "message_count": len(history_messages),
                                },
                            },
                        }
                    )
                )

                logger.info(
                    f"Sent conversation history: {len(history_messages)} messages"
                )

        except Exception as e:
            logger.error(f"Error loading previous conversation: {e}")
            # Fall back to new conversation on error
            conversation_id = str(uuid.uuid4())
            self.conversation_ids[websocket] = conversation_id
            logger.info(f"Fallback to new conversation: {conversation_id}")

    async def _connect_websocket(self, websocket: WebSocket):
        """Connect a WebSocket and load previous conversation history."""
        try:
            logger.info(f"WebSocket connection attempt from {websocket.client}")
            await websocket.accept()
            self.active_connections.append(websocket)

            # Try to resume the most recent conversation
            await self._load_previous_conversation(websocket)

            logger.info(
                f"WebSocket connection established. Total connections: "
                f"{len(self.active_connections)}"
            )
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            raise

    def _disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Clean up conversation id
        if websocket in self.conversation_ids:
            del self.conversation_ids[websocket]
        logger.info(
            f"WebSocket connection closed. Total connections: "
            f"{len(self.active_connections)}"
        )

    async def start_server(self):
        """Start the WebSocket server with comprehensive cleanup."""
        # Initialize chat service
        await self.chat_service.initialize()

        # Start server
        chat_config = self.config.get("chat", {})
        websocket_config = chat_config.get("websocket", {})
        host = websocket_config.get("host", "localhost")
        port = websocket_config.get("port", 8000)

        logger.info(f"Starting WebSocket server on {host}:{port}")

        server_config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(server_config)

        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal, cleaning up...")
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            raise
        finally:
            logger.info("Shutting down WebSocket server and cleaning up resources...")
            try:
                await self.chat_service.cleanup()
                logger.info("Chat service cleanup completed")
            except Exception as e:
                logger.error(f"Error during chat service cleanup: {e}")
                # Don't re-raise cleanup errors to avoid masking the original exception


async def run_websocket_server(
    clients: list["MCPClient"],
    llm_client: "LLMClient",
    config: dict[str, Any],
    repo: ChatRepository,
    configuration: Configuration,
) -> None:
    """
    Run the WebSocket server.

    This function maintains the same interface as before but now uses
    the clean separation between communication and business logic.
    """
    server = WebSocketServer(clients, llm_client, config, repo, configuration)
    await server.start_server()
