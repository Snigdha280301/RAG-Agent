# 🤖 Reddit RAG Agent: FastAPI + LangChain + Streamlit

This project implements a **Retrieval-Augmented Generation (RAG) agent** that provides customer support and technical answers by retrieving real-world knowledge from Reddit forum posts (specifically `r/rag`).

The application is built with a clean, decoupled architecture:

- **Backend (FastAPI):** Handles all core business logic (indexing, LLM calls, and vector search).
- **Core Logic (LangChain):** Manages the RAG pipeline, the custom PRAW data loader, and vectorization.
- **Vector Store (ChromaDB):** Stores all forum post embeddings locally for quick retrieval.
- **Frontend (Streamlit):** Provides an interactive chat interface that communicates with the FastAPI backend via API calls.

---

## 🚀 1. Setup and Installation

Follow these steps to set up your environment, install dependencies, and configure API keys.

### 1.1 Prerequisites

- **Python 3.9+**
- **OpenAI API Key** (Required for the LLM and Embeddings)
- **Reddit API Credentials** (Client ID, Secret, etc. Required for live data access via PRAW)

### 1.2 Environment & Dependencies

1.  **Create and Activate Virtual Environment:**

    ```bash
    python -m venv venv
    # For macOS/Linux
    source venv/bin/activate
    # For Windows
    .\venv\Scripts\activate
    ```

2.  **Install Packages:**
    (Ensure you have created the `requirements.txt` file from the previous steps.)
    ```bash
    pip install -r requirements.txt
    ```

### 1.3 Configuration (`.env` file)

Fill out your `.env` file with all required keys and configuration.

| Variable               | Description                                                                     |
| :--------------------- | :------------------------------------------------------------------------------ |
| `OPENAI_API_KEY`       | Your primary key for LLM and Embeddings.                                        |
| `REDDIT_CLIENT_ID`     | Your Reddit application Client ID.                                              |
| `REDDIT_CLIENT_SECRET` | Your Reddit application Client Secret.                                          |
| `REDDIT_USERNAME`      | Your Reddit account username.                                                   |
| `REDDIT_PASSWORD`      | Your Reddit account password.                                                   |
| `SUBREDDIT_NAME`       | The subreddit to index (e.g., `rag`).                                           |
| `PERSIST_DIRECTORY`    | Local folder for ChromaDB (default: `./chroma_db`).                             |
| `FASTAPI_BASE_URL`     | The host URL for the frontend to connect to (default: `http://localhost:8000`). |

---

## ⚙️ 2. Running the Application

The system requires two separate terminals: one for the backend API and one for the frontend UI.

### 2.1 Start the FastAPI Backend (Terminal 1)

This step automatically runs the indexing process (`rag_core.index_data`) on startup if the `./chroma_db` folder doesn't exist.

```bash
# Ensure venv is active!
uvicorn main:app --reload --port 8000
```

Access API Docs: `http://127.0.0.1:8000/docs`

---

### 2.2 Start the Streamlit Frontend (Terminal 2)

Open a second terminal window, activate your venv, and then run the frontend application.

```bash
# Ensure venv is active!
streamlit run frontend.py
```

Access Chat UI: `http://localhost:8501` (or the URL provided by Streamlit)

---

## 🔧 Key Endpoints & Customization

The RAG Process
**Start**: User submits a question via the Streamlit UI.

**API**: Streamlit sends the question to the `POST /chat` endpoint on the FastAPI backend.

**Retrieve**: The RAG chain uses the query to search the ChromaDB vector store for the top 5 most relevant Reddit posts.

**Augment & Generate**: The retrieved post text and the user's question are sent to GPT-4o-mini (the LLM) via the RAG prompt, which synthesizes a grounded answer.

**Return**: FastAPI sends the clean text answer and citation URLs back to Streamlit.
