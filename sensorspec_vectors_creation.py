import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import utils

#Block 1 PDF --> Markdown text
#Extracting PDF as a single Markdown String
md_text = pymupdf4llm.to_markdown("bst-bme280-ds002.pdf")

#Block 2 Text Splitting
#definition of structured text splitters
headers_to_split_on = [
    ('#', "header 1"),
    ('##', "header 2"), 
    ('###', "header 3"),
]

#initialize markdownsplitter
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

#create structured header splits
md_header_splits = markdown_splitter.split_text(md_text)

#initialize recursive character text splitter for further chunking limiting the chunk size
chunk_size = 500
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,    
)

final_chunks = text_splitter.split_documents(md_header_splits)
#utils.preview_chunks(final_chunks)

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
)

print("successfully created the vector database")