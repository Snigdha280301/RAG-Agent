# models.py
from pydantic import BaseModel, Field
from typing import List

# --- Data Models (FastAPI Schema) ---

class IndexRequest(BaseModel):
    """Model for the data ingestion request."""
    # Allows specifying a target (subreddit name) or an optional local file path
    target: str = Field(description="Subreddit name (e.g., 'rag') or 'local' to use mock data.")
    limit: int = Field(default=25, description="Maximum number of posts to fetch from the live API.")

class ChatRequest(BaseModel):
    """Model for the user chat query."""
    question: str = Field(..., description="The user's question to the RAG agent.")

class ChatResponse(BaseModel):
    """Model for the agent's response, including citations."""
    answer: str
    sources: List[str] = Field(default=[], description="List of unique source URLs for the answer.")