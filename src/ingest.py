from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from . import config, utils

def process_pdf(filepath = config.PDF_PATH):
    print(f"Starting ingestion process for PDF: {filepath}")

    ''' only use this if uploading the file locally and not on streamlit, otherwise the file uploader in streamlit will handle this
    if not config.PDF_PATH.exists():
        print(f"Error: PDF file not found at {config.PDF_PATH}")
        return
    '''
    #Block 1 PDF --> Markdown text
    md_header_splits = utils.pdf_to_markdown(filepath)

    #Block 2 Text Splitting
    #initialize recursive character text splitter for further chunking limiting the chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,    
    )

    final_chunks = text_splitter.split_documents(md_header_splits)
    print(f"Total Chunks Created: {len(final_chunks)}")


    #Block 3 creating vector embeddings
    #Initialize HuggingFace Embeddings
    embeddings = utils.get_embedding_function()

    print(f"Creating vector store using Chroma...")
    #creating vector store using chroma
    vector_store =Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k":3})

if __name__ == "__main__":
    process_pdf()