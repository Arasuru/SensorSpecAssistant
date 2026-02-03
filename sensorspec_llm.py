import os
from dotenv import load_dotenv
import llm_utils
#load variables from .env file
load_dotenv()
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables")


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

#creating the embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

#loading the existing vector database
db = Chroma(
    persist_directory="./chroma_db/sensorspec-bme280", 
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

#The chat loop with retrieval augmented generation
chat_history = []
MAX_HISTORY_LENGTH = 6 #keep last 6 messages in history(3 human, 3 AI)

print("Starting RAG chat with Groq LLM model...")
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat.")
        break
    
    #checking the length of chat history
    if len(chat_history) > MAX_HISTORY_LENGTH:
        chat_history = chat_history[-MAX_HISTORY_LENGTH:]

    #formatting chat history
    history_str = llm_utils.format_history(chat_history)

    #Stage 1: question rewriting
    if history_str:
        search_query = rewrite_chain.invoke({
            "chat_history": history_str,
            "question": user_input
        })
        print(f"Rewritten Question: {search_query}\n")
    else:
        search_query = user_input

    #retrieve relevant chunks from db
    docs = retriever.invoke(search_query)

    #prepare input
    context_str = llm_utils.format_docs(docs)

    chain = prompt | llm
    response = chain.invoke({
        "chat_history": history_str,
        "context": context_str,
        "question": user_input #no need to rewritten question here as LLm already knows chat history
    })
    print(f"AI: {response.content}")

    #update chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))