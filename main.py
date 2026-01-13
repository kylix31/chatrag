"""
Main Application Entry Point
FastAPI application setup and configuration
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import router
from src.application.graph import ConversationGraph
from src.infrastructure.config import get_settings


# Global variable to store the graph instance
conversation_graph: ConversationGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to initialize resources on startup
    and cleanup on shutdown
    """
    global conversation_graph
    # Startup: Initialize the graph once
    conversation_graph = ConversationGraph()
    yield
    # Shutdown: Cleanup if needed
    conversation_graph = None


app = FastAPI(
    title="ChatRAG API",
    description="Chatbot with RAG using LangChain, LangGraph, Azure AI Search and OpenAI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, tags=["conversations"])


def get_conversation_graph() -> ConversationGraph:
    """Dependency to get the conversation graph instance"""
    if conversation_graph is None:
        raise RuntimeError("ConversationGraph not initialized")
    return conversation_graph


@app.get("/", tags=["root"])
async def root():
    settings = get_settings()
    """Root endpoint"""
    return {
        "message": "ChatRAG API",
        "docs": "/docs",
        "health": "/health",
        "model": settings.openai_chat_model,
    }
