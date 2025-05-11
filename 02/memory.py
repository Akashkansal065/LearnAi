from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableMap
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import traceback
import os
import ollama
# Here to learn the memory Management in LLM

ollmaclient = ollama.Client(host="http://localhost:11434/")
print("Ollama client initialized.")
print(ollmaclient.list().models[0].model)
llm_model_name = "mistral"
llm = ChatOllama(model=llm_model_name, temperature=0.1)
# print(llm.invoke("Hello, world!"))

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# wikipedia.run("What is the capital of France?")

tools = [wikipedia]
llm_with_tools = llm.bind_tools(tools)
print("LLM with tools initialized.")
# Create Prompt Tempelate with history

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# from langchain.agents.output_parsers import AgentOutputParser

# Simple output parser


class SimpleAgentOutputParser(ToolsAgentOutputParser):
    def parse(self, text):
        return text.strip()


# Agent pipeline
agent = (
    RunnableMap({
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", []),
        "agent_scratchpad": lambda x: format_to_tool_messages(x.get("intermediate_steps", [])),
    })
    | prompt
    | llm
    | SimpleAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

chat_history = ChatMessageHistory()

while True:
    print("-----------------------------------------")
    query = input("Query: ")
    if query.strip().lower() == "exit":
        print("Exiting chat.")
        break
    # query = "What is the capital of India?"
    # query = "Who won the 2018 Fifa Wordcup?"

    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history.messages})
    # print("Response:", response)
    chat_history.add_user_message(query)
    if isinstance(response, dict):
        response_content = response.get('output', '')
    else:
        response_content = response
    # print("AI:", response_content)
    chat_history.add_ai_message(response_content)
    # print("Updated chat history:", chat_history.messages)

    # query = "when france won the first world cup?"
    # response = agent_executor(
    #     {"input": query, "chat_history": chat_history.messages})
    # print("Response:", response['output'])
