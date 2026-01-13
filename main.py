"""
Main Application Entry Point
FastAPI application setup and configuration
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import router
from src.infrastructure import get_settings

app = FastAPI(
    title="ChatRAG API",
    description="Chatbot with RAG using LangChain, LangGraph, Azure AI Search and OpenAI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, tags=["conversations"])


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {"message": "ChatRAG API", "docs": "/docs", "health": "/health"}


def main():
    """Start the server"""
    settings = get_settings()
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=True)


if __name__ == "__main__":
    main()
