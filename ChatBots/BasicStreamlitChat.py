import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# Streamlit UI setup
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖")
st.title("🤖 Chat with Ollama + LangChain")

# Set up session chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)

# User input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Model & prompt setup
    llm = OllamaLLM(model="qwen3:14b", streaming=True)
    prompt = ChatPromptTemplate.from_template(
        "Continue the conversation:\n{chat_history}\nHuman: {input}\nAI:")

    # History store (new approach)
    history_store = {}

    def get_history(session_id):
        if session_id not in history_store:
            history_store[session_id] = StreamlitChatMessageHistory()
        return history_store[session_id]

    # Runnable Chain
    chain = RunnableWithMessageHistory(
        RunnableLambda(
            lambda inputs: llm.stream(
                prompt.format(input=inputs["input"], chat_history="\n".join(
                    [f"{m.type.capitalize()}: {m.content}" for m in get_history("default").messages]))
            )
        ),
        get_session_history=get_history,
        input_messages_key="input"
    )

    # Generate and display streaming response
    with st.chat_message("assistant"):
        response_md = ""
        response_area = st.empty()
        for chunk in chain.invoke({"input": user_input}, config={"configurable": {"session_id": "default"}}):
            response_md += chunk
            response_area.markdown(response_md)

    st.session_state.messages.append(("assistant", response_md))
    get_history("default").add_user_message(user_input)
    get_history("default").add_ai_message(response_md)
