# rag_core.py
import os
import sys
import json
import praw
from typing import List, Optional, Any
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from dotenv import load_dotenv

# Load all variables from .env
load_dotenv() 

# --- Configuration (Read from .env) ---
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
RAG_MODEL = os.getenv("RAG_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MOCK_DATA_PATH = os.getenv("MOCK_DATA_PATH", "data/mock_r_rag_data.json")

# Global instance of the retriever
rag_retriever: Optional[Any] = None

# --- Custom Loaders ---

class RedditJSONLoader(BaseLoader):
    """Loads mock data as a fallback."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
             print(f"Mock file not found at {self.file_path}.", file=sys.stderr)
             return []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for post in data:
            page_content = f"Title: {post.get('title', '')}\nContent: {post.get('content', '')}"
            metadata = {
                "source": post.get("url", "N/A"),
                "author": post.get("author", "N/A"),
                "date": post.get("date", "N/A"),
                "type": "mock-reddit-post"
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

class PRAWLoader(BaseLoader):
    """Loads live data from Reddit using PRAW."""
    def __init__(self, subreddit_name: str, limit: int):
        self.subreddit_name = subreddit_name
        self.limit = limit
        self.reddit = self._authenticate()

    def _authenticate(self):
        """Authenticates with Reddit using environment variables."""
        return praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent="rag_fastapi_agent_v1.0"
        )

    def load(self) -> List[Document]:
        """Fetches posts and converts them to LangChain Documents."""
        subreddit = self.reddit.subreddit(self.subreddit_name)
        documents = []
        
        for submission in subreddit.top(limit=self.limit):
            page_content = f"Title: {submission.title}\nBody: {submission.selftext}"
            
            metadata = {
                "source": f"https://www.reddit.com{submission.permalink}",
                "author": str(submission.author),
                "date": submission.created_utc,
                "score": submission.score,
                "type": "live-reddit-post"
            }
            if page_content and page_content.strip():
                 documents.append(Document(page_content=page_content, metadata=metadata))
            
        return documents

# --- RAG Core Functions ---

def get_rag_chain():
    """Initializes and returns the complete RAG Chain."""
    global rag_retriever
    if rag_retriever is None:
        raise ValueError("RAG retriever is not initialized.")
        
    llm = ChatOpenAI(model=RAG_MODEL, temperature=0)

    def format_docs(docs: List[Document]) -> str:
        """Formats the retrieved documents for the LLM prompt."""
        formatted_list = []
        for doc in docs:
            source = doc.metadata.get("source", "N/A")
            content = doc.page_content
            formatted_list.append(f"{content}\nSource: {source}")
        return "\n\n---\n\n".join(formatted_list)
        
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Reddit RAG support agent. Use the provided "
            "forum posts (CONTEXT) to answer the user's QUESTION. "
            "You MUST cite the 'Source' URL provided in the context for every relevant quote or fact used."
            "\n\nCONTEXT:\n{context}"
        ),
        ("human", "{question}"),
    ])
    
    rag_chain = (
        {"context": itemgetter("question") | rag_retriever | RunnableLambda(format_docs),
         "question": itemgetter("question")}
        | rag_prompt
        | llm
        | str
    )
    return rag_chain


def index_data(target: str, limit: int):
    """Decides which loader to use and indexes the data."""
    global rag_retriever
    
    print(f"--- Indexing target: {target} (Limit: {limit})... ---", file=sys.stderr)

    if target.lower() == 'local':
        loader = RedditJSONLoader(MOCK_DATA_PATH)
    else:
        try:
            # PRAWLoader raises RuntimeError on auth failure
            loader = PRAWLoader(target, limit)
        except RuntimeError as e:
            # Fallback to local if PRAW auth fails
            print(f"PRAW failed for {target}. Falling back to local mock data. Error: {e}", file=sys.stderr)
            loader = RedditJSONLoader(MOCK_DATA_PATH)

    docs = loader.load()
    if not docs:
        raise ValueError("No documents loaded from the source.")

    # Split, Embed, and Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print(f"Indexing Complete. Indexed {len(all_splits)} chunks.", file=sys.stderr)
    
def load_existing_index():
    """Attempts to load the vector store from disk."""
    global rag_retriever
    if not os.path.exists(PERSIST_DIRECTORY):
        return False
        
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        rag_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("Existing RAG index loaded successfully from disk.", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        return False

def get_current_retriever() -> Optional[Any]:
    return rag_retriever