import os
import sys
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

from src import utils, config 

load_dotenv()

st.set_page_config(page_title="SensorSpec Assistant", page_icon="📟", layout="centered")

# Resource Initialization (cache for performance)
@st.cache_resource(show_spinner="Loading Vector Databse...")
def load_rag_component():
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")

    #Initializing Embedding and vector DB
    embedding_function = utils.get_embedding_function()

    if not config.VECTORS_DIR.exists():
        print(f"Error: Vector database not found at {config.VECTORS_DIR}. Please run the ingest script first.")
        return

    #loading the existing vector database
    db = Chroma(
        persist_directory=str(config.VECTORS_DIR), 
        embedding_function=embedding_function
    )
    retriever = db.as_retriever(search_kwargs={"k":3})

    #initialize groq chat model
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

    return retriever, rewrite_chain, rag_chain


#load components
retriever, rewrite_chain, rag_chain = load_rag_component()

#Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


MAX_HISTORY_LENGTH = 6

# UI layout
st.title(" SensorSpec Assistant")
st.markdown(f'** chat with the {config.PDF_FILENAME} datasheet!**')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Interaction ---
if user_input := st.chat_input("Ask a question about the BME280 sensor..."):
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
            docs = retriever.invoke(search_query)
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