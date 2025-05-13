from PgVectorChatHistory import PGVectorChatHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
import traceback
import os
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate


DB_URL = "postgresql://username:password@localhost:5432/yourdb"

# 2. Initialize chat history
chat_history = PGVectorChatHistory(
    db_url=DB_URL,
    embedding_model=OllamaEmbeddings(model="mxbai-embed-large")
)

# 3. Initialize Ollama LLM
llm = ChatOllama(model="qwen3:14b", temperature=0.1)

# 4. Initialize vector store retriever
retriever = Chroma(
    persist_directory="./chroma_store",
    collection_name="HR_QnA",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
).as_retriever()

# 5. Chat prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

# 6. QA chain
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

# 7. Chat loop
while True:
    print("-----------------------------------------")
    query = input("Query: ")
    if query.strip().lower() == "exit":
        print("Exiting chat.")
        break

    try:
        response = qa_chain.invoke({
            "input": query,
            "chat_history": chat_history.messages  # from PG vector
        })
        chat_history.add_user_message(query)
        chat_history.add_ai_message(response)
        print("AI:", response)
    except Exception as e:
        print("Error during query execution:", str(e))
        traceback.print_exc()
