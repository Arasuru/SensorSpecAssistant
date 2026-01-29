import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

#Extracting PDF as a single Markdown String
md_text = pymupdf4llm.to_markdown("bst-bme280-ds002.pdf")

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

#preview 3 chunks
for i, chunk in enumerate(final_chunks[:3]):
    print(f"--- Chunk {i+1} ---")
    print(f"chunk metadata: {chunk.metadata}")
    print(f"chunk content: {chunk.page_content}")
    print()
