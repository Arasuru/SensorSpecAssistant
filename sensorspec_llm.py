import os
from dotenv import load_dotenv

#load variables from .env file
load_dotenv()
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables")


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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

#Defining the prompt template
prompt_template = """You are a helpful assistant and embedded sensors expert. Use the following context to answer the question. if the answer is not in context , then say I don't know.
                Context:
                {context}
                Question: {question}"""

prompt = ChatPromptTemplate.from_template(prompt_template)

#Formatting each chunk as a paragraph and joining to one context string
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

ragchain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

#running the rag chain
query = "What is the device slave address for I2C interface?"
print(f"Querying: {query}")
response = ragchain.invoke(query)
print(response)