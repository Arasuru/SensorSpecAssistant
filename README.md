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

* **`sensorspec_vectors_creation.py`**: The ingestion script. It loads the PDF, splits it into markdown chunks, creates embeddings, and saves them to the vector database.
* **`sensorspec_llm.py`**: The main chat application. It loads the database, initializes the LLM, and handles the RAG chat loop.
* **`utils.py`**: Helper functions for PDF processing (single doc vs. page-by-page) and loading the similarity database.
* **`llm_utils.py`**: Helper functions for formatting documents and chat history strings.
* **`test_setup.py`**: A utility script to verify library versions.
* **`bst-bme280-ds002.pdf`**: The source datasheet file (ensure this file exists in the root directory).
* **`chroma_db/`**: The directory where the vector database is persisted.

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
    pip install -U langchain langchain-chroma langchain-huggingface langchain-groq pymupdf pymupdf4llm python-dotenv
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=gsk_your_groq_api_key_here
    ```

## 🏃 Usage

### 1. Create the Vector Database
Before chatting, you must process the PDF and create the embeddings. Ensure the file `bst-bme280-ds002.pdf` is in the project root.

Run the creation script:
```bash
python sensorspec_vectors_creation.py
```

### 2. Start the Chat Assistant
Once the database is ready, start the chat interface:

```Bash
python sensorspec_llm.py
```
Type your question when prompted with User:.
Type `exit` or `quit` to end the session.

### ⚙️ Configuration
* ###Adjusting Chunking
In `sensorspec_vectors_creation.py`, you can modify how the text is split:

```Python
chunk_size = 500
chunk_overlap = 50
```

* Changing the LLM
In `sensorspec_llm.py`, you can swap the model (e.g., to a larger Llama model or Mixtral) by changing the model name in the ChatGroq initialization:

```Python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)
```

## 📝 Notes
* The embedding model is configured to run on cpu by default in `sensorspec_vectors_creation.py` and `utils.py`. If you have a CUDA-enabled GPU, you can change device to cuda.

* The system keeps a memory of the last 6 messages (3 interactions) to maintain context without exceeding token limits.
