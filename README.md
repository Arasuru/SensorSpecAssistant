# SensorSpecAssistant

**SensorSpecAssistant** is a Retrieval-Augmented Generation (RAG) application designed to act as an expert assistant for sensor specifications. Specifically tuned for the **BME280** sensor, it allows users to chat with the datasheet using natural language.

The application leverages **LangChain**, **ChromaDB** for vector storage, **HuggingFace** for embeddings, and **Groq** (using Llama 3.1) for high-speed inference.

## 🚀 Features

* **PDF to Markdown Conversion**: Uses `pymupdf4llm` to preserve structure and headers from the datasheet.
* **Vector Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (runs efficiently on CPU) to create semantic embeddings.
* **Persistent Vector Store**: Stores embeddings in a local ChromaDB instance to avoid reprocessing the PDF every time.
* **Context-Aware Chat**: Implements a history-aware retriever that rewrites user queries based on previous conversation context.
* **Fast Inference**: Powered by the Groq API running `llama-3.1-8b-instant`.

## 📂 Project Structure

* **`src/config.py`**: The main config file to setup path variables to run smoothly on any system.
* **`src/ingest.py`**: The ingestion script. It loads the PDF, splits it into markdown chunks, creates embeddings, and saves them to the vector database.
* **`src/chat.py`**: The main chat application. It loads the database, initializes the LLM, and handles the RAG chat loop.
* **`src/utils.py`**: Helper functions for PDF processing, formatting documents and chat history strings.


## 🛠️ Prerequisites

* Python 3.10 or higher
* A [Groq API Key](https://console.groq.com/)

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Arasuru/SensorSpecAssistant.git
    cd SensorSpecAssistant
    ```

2.  **Install dependencies:**
    You can install the required packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=gsk_your_groq_api_key_here
    ```

## 🏃 Usage

### 1. Create the Vector Database
Before chatting, you must process the PDF and create the embeddings. Ensure the file `bst-bme280-ds002.pdf` is in the INPUT_DIR.

Run the creation script:
```bash
python -m src.ingest
```

### 2. Start the Chat Assistant
Once the database is ready, start the chat interface:

```Bash
python -m src.chat
```
Type your question when prompted with User:.
Type `exit` or `quit` to end the session.

## ⚙️ Configuration

### Adjusting Chunking
In `src/config.py`, you can modify how the text is split:

```Python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

### Changing the Model Settings
* In `src/config.py`
* you can swap the model (e.g., to a larger Llama model or Mixtral) by changing the `LLM_MODEL_NAME`.
* you can swap the embedding function by changing the `EMBEDDING_MODEL_NAME`.

```Python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
LLM_MODEL_NAME = "llama-3.1-8b-instant"
```

## 📝 Notes
* The embedding model is configured to run on cpu by default in `src/config.py` (`EMBEDDING_MODEL_KWARGS`). If you have a CUDA-enabled GPU, you can change device to cuda.

* The system keeps a memory of the last 6 messages (3 interactions) to maintain context without exceeding token limits.
