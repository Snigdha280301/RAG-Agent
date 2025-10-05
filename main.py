# main.py
import os
import asyncio
import sys
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables (must be first)
load_dotenv() 

# FastAPI Imports
from fastapi import FastAPI, HTTPException

# Local Imports
from models import IndexRequest, ChatRequest, ChatResponse
import rag_core 

# --- Configuration (Read from .env) ---
DEFAULT_SUBREDDIT = os.getenv("SUBREDDIT_NAME", "LangChain")
PRAW_FETCH_LIMIT = int(os.getenv("PRAW_FETCH_LIMIT", 25))


# --- Initialization on Startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the RAG application.
    Code before the 'yield' is run on startup.
    Code after the 'yield' is run on shutdown.
    """
    print("Application Startup: Checking RAG index status...", file=sys.stderr)
    
    # --- STARTUP LOGIC (REPLACES @app.on_event("startup")) ---
    if not rag_core.load_existing_index():
        print(f"Existing index not found. Running initial data ingestion from r/{DEFAULT_SUBREDDIT}...", file=sys.stderr)
        
        # Run initial indexing with the default subreddit
        await asyncio.to_thread(rag_core.index_data, DEFAULT_SUBREDDIT, PRAW_FETCH_LIMIT)
        
    yield  # Application is ready to handle requests

    # --- SHUTDOWN LOGIC (Run on exit, optional for RAG) ---
    print("Application Shutdown complete.", file=sys.stderr)


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Reddit RAG Agent API (PRAW Ready)",
    description="Scalable RAG agent using LangChain, FastAPI, and data sourced from Reddit (or local mock data).",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/status")
def get_status():
    """Check the health and readiness of the RAG agent."""
    is_ready = rag_core.get_current_retriever() is not None
    return {"status": "ready" if is_ready else "indexing", "model": rag_core.RAG_MODEL}


@app.post("/index", summary="Rebuild RAG knowledge base from specified target", response_model=dict)
async def reindex_data(request: IndexRequest):
    """Endpoint to trigger the indexing process from a specified subreddit or 'local' file."""
    
    # Run the indexing function in a separate thread
    await asyncio.to_thread(rag_core.index_data, request.target, request.limit)
    return {"message": f"Successfully indexed data from target: {request.target}. Fetch limit: {request.limit}"}


# main.py (Revised /chat endpoint)

@app.post("/chat", summary="Query the RAG Agent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """The main endpoint for sending a question to the RAG agent."""
    
    if rag_core.get_current_retriever() is None:
        raise HTTPException(status_code=503, detail="RAG system is still indexing or failed to load.")
    
    try:
        rag_chain = rag_core.get_rag_chain()
        retriever_instance = rag_core.get_current_retriever()
        
        # 1. Retrieve documents (to get sources for citation)
        retrieved_docs = await retriever_instance.ainvoke(request.question)
        
        # 2. Invoke the RAG chain for the final answer
        answer_object = await rag_chain.ainvoke({"question": request.question})

        # --- CRITICAL FIX: Extracting the clean string ---
        # The chain ends with '| str' but might wrap the final text.
        # We ensure it is a simple string by casting, which discards the verbose parts.
        # If the output is already a string (which it should be), this is safe.
        final_answer_text = str(answer_object)
        
        # 3. Extract unique sources for the response model
        sources = list(set([doc.metadata.get("source", "N/A") for doc in retrieved_docs]))
        
        # 4. Return the clean Pydantic model
        return ChatResponse(answer=final_answer_text, sources=sources)
    
    except Exception as e:
        # Note: Added sys import if you don't have it
        import sys 
        print(f"RAG Chain Error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")