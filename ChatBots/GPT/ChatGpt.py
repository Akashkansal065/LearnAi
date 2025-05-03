import streamlit as st
import time
import requests
import openai
from streamlit_chat import message

# --- CONFIG ---
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

ollama_model = ''


def get_ollama_models_from_api():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [model['name'] for model in data.get('models', [])]
    except Exception as e:
        st.error(f"Error fetching Ollama models from API: {e}")
        return []


# --- Session state for messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio("Choose Model", ("OpenAI", "Ollama"))

    if model_choice == "OpenAI":
        openai_model = st.selectbox("Select OpenAI Model", [
                                    "gpt-3.5-turbo", "gpt-4"])
    else:
        ollama_models = get_ollama_models_from_api()
        selected_models = []
        if len(ollama_models) > 0:
            # st.sidebar.write("Select Ollama Models:")
            ollama_model = st.selectbox("Select Ollama Models:", ollama_models)
            print(ollama_model)
            # for model in ollama_models:
            #     if st.sidebar.checkbox(model, key=f"checkbox_{model}"):
            #         selected_models.append(model)
        else:
            st.sidebar.warning("No Ollama models found.")
            ollama_model = st.sidebar.text_input("Enter Model Manually")
            if ollama_model:
                selected_models.append(ollama_model)

st.title("🤖 Chat with AI (OpenAI + Ollama Switch)")

for idx, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(idx))

prompt = st.chat_input("Type your message...")


def stream_ollama_response(prompt, model):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "messages": [
            {"role": "user", "content": prompt}]},
        stream=True,
    )
    for line in response.iter_lines():
        if line:
            data = line.decode('utf-8')
            if '"done":true' in data:
                break
            # Extract 'message' part manually
            if '"content":"' in data:
                try:
                    content_start = data.index(
                        '"content":"') + len('"content":"')
                    content_end = data.index('"', content_start)
                    content_piece = data[content_start:content_end]
                    yield content_piece.replace('\\n', '\n').replace('\\', '')
                except Exception:
                    continue


# --- If prompt is entered ---
if prompt:
    # Save user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show assistant typing
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        stream_generator = stream_ollama_response(prompt, ollama_model)

        for chunk in stream_generator:
            full_response += chunk
            response_placeholder.markdown(
                full_response + "▌")  # Add blinking cursor
            time.sleep(0.02)  # Adjust typing speed

        response_placeholder.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
