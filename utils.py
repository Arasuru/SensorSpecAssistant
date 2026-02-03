import fitz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import pymupdf4llm


def pdf_to_markdown_single(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    #definition of structured text splitters
    headers_to_split_on = [('#', "header 1"), ('##', "header 2"), ('###', "header 3"),]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = markdown_splitter.split_text(md_text)
    return splits
    
def pdf_to_markdown_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    documents = []
    for page_num, page in enumerate(doc):
        page_md = pymupdf4llm.to_markdown(pdf_path, pages=[page_num]) #can pass list of page numbers as well
        #create langchain doc with page number in metadata
        doc_obj = Document(page_content=page_md, metadata={"source": pdf_path, "page": page_num+1})
        documents.append(doc_obj)
    
    headers_to_split_on = [('#', "header 1"), ('##', "header 2"), ('###', "header 3"),]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    final_splits = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        #The splitter returns new list of Documents, we need to add metadata to each split
        for split in splits:
            split.metadata.update(doc.metadata) #adds original metadata(source, page) to each split
            final_splits.append(split)
    
    return final_splits

def preview_chunks(chunks):
    #preview 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(f"chunk metadata: {chunk.metadata}")
        print(f"chunk content: {chunk.page_content}")
        print()

def load_similarity_db(query: str):
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
    print(f"Querying: {query}")
    v1 = embedding_function.embed_query(query)
    print(f"Length of query embedding vector: {len(v1)}")
    # Retrieve top 3 most relevant chunks
    results = db.similarity_search(query, k=3)
    print(f"Top {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Source Header: {doc.metadata}")
