import os
import sys
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from src import utils, config, ingest

load_dotenv()

st.set_page_config(page_title="SensorSpec Assistant", page_icon="📟", layout="centered")

# Resource Initialization (cache for performance)
@st.cache_resource(show_spinner="Loading Vector Database...")
def load_default_retriever():
    """Loads the pre-processed database, or builds it if it doesn't exist."""
    
    # 1. If the database exists, load it normally
    if config.VECTOR_STORE_DIR.exists():
        embedding_function = utils.get_embedding_function()
        db = Chroma(
            persist_directory=str(config.VECTOR_STORE_DIR), 
            embedding_function=embedding_function
        )
        return db.as_retriever(search_kwargs={"k": 3})
        
    # 2. If the database DOES NOT exist, check if we have the PDF and build it!
    elif config.PDF_PATH.exists():
        print("Building default database on first run...")
        # ingest.process_pdf creates the Chroma db in-memory or on-disk
        db = ingest.process_pdf(config.PDF_PATH)
        return db.as_retriever(search_kwargs={"k": 3})
        
    # 3. If neither the DB nor the PDF exists, gracefully fall back
    else:
        st.warning("Default vector database and PDF not found. Please upload a document to begin.")
        return None

@st.cache_resource(show_spinner="Loading LLM Chains...")
def load_llm_chains():
    #initialize groq chat model
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found in environment variables")
        st.stop()

    #chatgroq automatically looks for GROQ_API_KEY in environment variables
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    #Defining the prompt templates for two stage RAG
    #Prompt 1: to make the question more specific using chat history
    rewrite_system_prompt = """Given a chat history and the latest user question 
                            which might reference context in the chat history, formulate a standalone question 
                            which can be understood without the chat history. 
                            Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", rewrite_system_prompt),
        ("human", "Chat History:\n{chat_history}\n\nLatest Question: {question}")
    ])

    #mini-chain for question rewriting
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    #Prompt 2: to answer the question using retrieved context
    prompt_template = """You are a helpful assistant and embedded sensors expert. Be Concise and precise in your answers. Keep the chat history in mind while answering the question.
        Here is the Conversation history so far:
         {chat_history} 
        Use the following context from the document. Context:
        {context}
        answer the following question. if the answer is not in context , then say I don't know.
        Question: {question}"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = prompt | llm

    return rewrite_chain, rag_chain

@st.cache_resource(show_spinner=False)
def process_uploaded_file(file_bytes):
    #process pdf and caches the vector store, file_bytes act as cache key, so that if the same file is uploaded again, it will use the cached retriever instead of reprocessing
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        temp_file_path = tmp_file.name
    try:
        retriever = ingest.process_pdf(temp_file_path)
        return retriever
    finally:
        os.remove(temp_file_path)

#Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = load_default_retriever()
if "current_file" not in st.session_state:
    st.session_state.current_file = "Default (BME280 Datasheet)"

rewrite_chain, rag_chain = load_llm_chains()
MAX_HISTORY_LENGTH = 6


with st.sidebar:
    st.header("📄 Custom Datasheet")
    st.markdown("Upload your own datasheet to chat with it. If cleared, it defaults back to the BME280 sensor.")
    
    uploaded_file = st.file_uploader("Upload a PDF datasheet", type=["pdf"])
    
    if uploaded_file:
        # Check if it's a new file to avoid resetting chat history 
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner(f"Processing and Embedding {uploaded_file.name}..."):
                
                # Pass the raw bytes to our new cached function
                st.session_state.retriever = process_uploaded_file(uploaded_file.getvalue())
                st.session_state.current_file = uploaded_file.name
                
                # Reset chat history for the new document
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Document ready!")
    else:
        # This prevents Streamlit glitches from wiping chat history!
        if st.session_state.current_file != "Default (BME280)":
            st.warning("To return to the default BME280 datasheet, click below:")
    
    if st.button("Revert to Default"):
                st.session_state.retriever = load_default_retriever()
                st.session_state.current_file = "Default (BME280)"
                st.session_state.messages = []
                st.session_state.chat_history = []

# UI layout
st.title(" SensorSpec Assistant")
st.markdown(f'** chat with the {st.session_state.current_file} datasheet!**')

if st.session_state.retriever is None:
    st.warning("No retriever available. Please upload a document to chat with.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Interaction ---
if user_input := st.chat_input("Ask a question about the sensor..."):
    # 1. Display User Message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Manage Langchain History Length
    if len(st.session_state.chat_history) > MAX_HISTORY_LENGTH:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_LENGTH:]
    
    history_str = utils.format_history(st.session_state.chat_history)

    # 3. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Stage 1: Rewrite query
            if history_str:
                search_query = rewrite_chain.invoke({
                    "chat_history": history_str,
                    "question": user_input
                })
            else:
                search_query = user_input

            # Stage 2: Retrieve
            docs = st.session_state.retriever.invoke(search_query)
            context_str = utils.format_docs(docs)

            # Stage 3: Answer
            response = rag_chain.invoke({
                "chat_history": history_str,
                "context": context_str,
                "question": user_input 
            })
            
            # Display response
            st.markdown(response.content)
            
            with st.expander("View Retrieved Context"):
                st.info(context_str if context_str else "No context retrieved.")

    # 4. Update Histories
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response.content)
    ])