from langchain_core.messages import HumanMessage, AIMessage

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