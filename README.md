# SensorSpec Assistant v1.0 📟

**SensorSpec Assistant** is a Retrieval-Augmented Generation (RAG) application designed to act as an expert assistant for technical datasheets. 

It comes pre-loaded with knowledge about the **BME280** sensor but also allows users to **upload their own PDF datasheets** to chat with them instantly.

## 🚀 Features

* **Interactive Web UI**: Built with **Streamlit** for a clean, chat-like experience.
* **Multi-Document Support**: 
    * **Default Mode**: Chat with the pre-indexed BME280 datasheet.
    * **Custom Mode**: Upload any PDF datasheet; the app indexes it in-memory for that session.
* **Smart Context**: Maintains chat history and rewrites queries to ensure the LLM understands follow-up questions.
* **Transparent RAG**: "View Retrieved Context" expander lets you see exactly what data the AI is using to answer.
* **High-Performance**: Powered by **Groq** (Llama 3.1) for near-instant inference and **HuggingFace** for efficient CPU-based embeddings.

## 📂 Project Structure

```text
SensorSpecAssistant/
├── .env                    # API Keys (GROQ_API_KEY)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   
│   ├── inputs/             # Store default PDFs here (e.g., bst-bme280-ds002.pdf)
│   └── vector_store/       # Persistent Vector Database (ChromaDB)
└── src/                    # Source Code
    ├── app.py              # Main Streamlit Application
    ├── chat.py             # CLI Chat Interface (optional)
    ├── config.py           # Configuration settings (Paths, Models)
    ├── ingest.py           # Script to process default PDFs
    └── utils.py            # Helper functions (PDF processing, Formatting)
```

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

## Local Usage
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

## Streamlit app
### 1. Start the Chat Assistant with Web UI
start the chat interface with UI Element using streamlit:

```Bash
streamlit run src/app.py
```

### 2. Accessing the app on cloud 
    `<https://sensorspec-assistant.streamlit.app/>`

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
