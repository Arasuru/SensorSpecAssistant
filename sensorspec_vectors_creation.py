import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import utils
import fitz

pdf = 'bst-bme280-ds002.pdf'

#Block 1 PDF --> Markdown text
by_page = False
if by_page:
    '''option 2: Extract PDF as list of pages in Markdown format using fitz (retains page numbers)'''
    md_header_splits = utils.pdf_to_markdown_by_page(pdf)
else:
    '''Option 1: Extract PDF as single Markdown using pymupdf4llm recommended for better formatting(loses page numbers)'''
    md_header_splits = utils.pdf_to_markdown_single(pdf)


#Block 2 Text Splitting
#initialize recursive character text splitter for further chunking limiting the chunk size
chunk_size = 500
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,    
)

final_chunks = text_splitter.split_documents(md_header_splits)
utils.preview_chunks(final_chunks)

print(f"Total Chunks Created: {len(final_chunks)}")


#Block 3 creating vector embeddings
#Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

#creating vector store using chroma
db = Chroma.from_documents(
    documents=final_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db/sensorspec-bme280",
    collection_name="sensorspec-bme280"
)

print("successfully created the vector database")

#Testing the similarity search
utils.load_similarity_db("device slave address of bme280 sensor")