# frontend.py
import streamlit as st
import requests
import json
from typing import List, Dict, Any

# --- Configuration ---
# The URL where your FastAPI backend is running
API_URL = "http://localhost:8000" 
CHAT_ENDPOINT = f"{API_URL}/chat"

# Initialize chat history using Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message from the AI
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am your Reddit RAG Agent. Ask me a question about RAG, vector databases, or LLMs."})

# --- Helper Functions ---

def call_rag_backend(query: str) -> Dict[str, Any]:
    """Sends the user query to the FastAPI RAG endpoint."""
    try:
        # FastAPI expects a JSON payload matching the ChatRequest Pydantic model
        payload = {"question": query}
        
        # We use a POST request to send the data
        response = requests.post(
            CHAT_ENDPOINT, 
            json=payload,
            timeout=120 # Set a generous timeout for LLM generation
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # The response matches the ChatResponse model: {answer: str, sources: List[str]}
        return response.json()
        
    except requests.exceptions.RequestException as e:
        return {"answer": f"ERROR: Could not connect to the backend (FastAPI). Is the server running at {API_URL}?", "sources": []}
    except Exception as e:
        return {"answer": f"An unexpected error occurred: {e}", "sources": []}


def display_sources(sources: List[str]):
    """Displays the unique source links in a collapsible expander."""

    if not sources:
        return
    clean_sources = [s for s in sources if s and s.startswith("http")]

    if not clean_sources:
        return
    
    with st.expander("📚 Sources Retrieved (Click to Expand)"):
        for source in clean_sources:
            st.markdown(f"- [Source Link]({source})")
            
# --- Streamlit UI ---

st.set_page_config(page_title="Reddit RAG Chatbot", page_icon="🤖")
st.title("Reddit RAG Agent")

# --- Display Chat History ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") and message["sources"]: 
            display_sources(message["sources"])
            

# --- Handle User Input ---

if prompt := st.chat_input("Ask a question based on the subreddit..."):
    
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the RAG backend
    with st.spinner("Searching knowledge base..."):
        rag_response = call_rag_backend(prompt)

    # 3. Process and display AI response
    assistant_response_raw = rag_response["answer"]
    sources = rag_response["sources"]

    # --- CRITICAL SANITIZATION STEP ---
    clean_response = assistant_response_raw
    
    # 1. Strip common LangChain/LLM wrapper prefixes (content=, additional_kwargs, etc.)
    if clean_response.startswith("content='"):
        clean_response = clean_response.split("content='", 1)[1]
        
    if 'additional_kwargs' in clean_response:
        clean_response = clean_response.split('additional_kwargs', 1)[0]
    
    # 2. Remove Newline Characters (The requested change)
    # Replaces double newlines with a single space, then removes single newlines.
    # This keeps the text flowing without large gaps.
    clean_response = clean_response.replace('\n\n', ' ').replace('\n', ' ')

    # 3. Final strip to remove trailing quotes, newlines, or spaces
    final_answer = clean_response.strip().strip("'").strip('"').strip()

    with st.chat_message("assistant"):
        st.markdown(final_answer) # Display the cleaned text
        if sources:
            display_sources(sources) # Display sources after the answer

    # 4. Add AI response to session history
    st.session_state.messages.append({"role": "assistant", "content": final_answer, "sources": sources})