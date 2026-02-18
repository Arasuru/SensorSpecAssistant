from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import pymupdf4llm
import fitz
from . import config


def get_embedding_function():
    """
    Initializes and returns the HuggingFaceEmbeddings function for consistency.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs=config.EMBEDDING_MODEL_KWARGS,
        encode_kwargs={'normalize_embeddings': True}
    )
    
#Formatting each chunk as a paragraph and joining to one context string
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

#formatting chat history from HumanMessage and AIMessage objects to texts
def format_history(history):
    formatted_history = ""
    for message in history:
        if isinstance(message, HumanMessage):
            formatted_history += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"AI: {message.content}\n"
    return formatted_history

def pdf_to_markdown(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    #definition of structured text splitters
    headers_to_split_on = [('#', "header 1"), ('##', "header 2"), ('###', "header 3"),]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = markdown_splitter.split_text(md_text)
    return splits