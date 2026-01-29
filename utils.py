from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_similarity_db():
    """
    Load the existing Chroma vector database from disk.
    Returns:
        Chroma: The loaded Chroma vector database.
    """
    # 1. Initializing the SAME embedding function used during creation
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Load the existing database from disk
    db = Chroma(
        persist_directory="./chroma_db/sensorspec-bme280", 
        embedding_function=embedding_function
    )

    # 3. Search
    query = "device slave address of bme280 sensor"
    print(f"Querying: {query}")
    v1 = embedding_function.embed_query(query)
    print(f"Length of query embedding vector: {len(v1)}")

    # Retrieve top 3 most relevant chunks
    results = db.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Source Header: {doc.metadata}")


def preview_chunks(chunks):
    #preview 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(f"chunk metadata: {chunk.metadata}")
        print(f"chunk content: {chunk.page_content}")
        print()

if __name__ == "__main__":
    load_similarity_db()