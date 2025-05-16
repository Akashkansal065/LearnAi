from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableMap
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
import traceback
import os
import ollama

# Initialize Ollama client
ollmaclient = ollama.Client(host="http://localhost:11434/")
print("Ollama client initialized.")
print(ollmaclient.list().models[0].model)

# LLM setup
llm_model_name = "qwen3:14b"
llm = ChatOllama(model=llm_model_name, temperature=0.1)

# Vector DB setup (Chroma)
CHROMA_STORE_PATH = "./chroma_store"
EMBED_MODEL_NAME = "mxbai-embed-large"
retriever = Chroma(
    persist_directory=CHROMA_STORE_PATH,
    collection_name="HR_QnA",  # Or whichever collection you're using
    embedding_function=OllamaEmbeddings(model=EMBED_MODEL_NAME)
).as_retriever()

# Prompt with chat history
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

# QA chain using retrieval
qa_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["input"]),
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", [])
    })
    | (lambda x: {
        "input": f"Context: {''.join([doc.page_content for doc in x['context']])}\n\nQuestion: {x['input']}",
        "chat_history": x["chat_history"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
chat_history = ChatMessageHistory()

while True:
    print("-----------------------------------------")
    query = input("Query: ")
    if query.strip().lower() == "exit":
        print("Exiting chat.")
        break

    try:
        response = qa_chain.invoke({
            "input": query,
            "chat_history": chat_history.messages
        })
        chat_history.add_user_message(query)
        chat_history.add_ai_message(response)
        print("AI:", response)
    except Exception as e:
        print("Error during query execution:", str(e))
        traceback.print_exc()
