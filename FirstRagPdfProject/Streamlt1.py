import streamlit as st
from RAG_PDF import upload_markdown, ask_question

st.set_page_config(page_title="Gemini Chat with Upload", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 Gemini-style RAG Chat")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Upload and Chat ---
uploaded_files = st.file_uploader(
    "➕ Upload Files", type=["pdf", "md"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Uploading and processing documents..."):
        result = upload_markdown(uploaded_files)
        st.success(result["message"])

# Chat input
user_input = st.chat_input("Ask anything about your documents...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(user_input)
            answer = result["answer"]
            sources = result.get("sources", [])
            st.markdown(answer)
            if sources:
                st.markdown("📎 **Sources:**")
                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append({"role": "assistant", "content": answer})
