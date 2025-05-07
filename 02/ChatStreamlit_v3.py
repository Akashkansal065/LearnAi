# ChatStreamlit_v3.py
import streamlit as st
import os
import tempfile
import uuid  # For unique chat IDs
import traceback
from Chatapp_v3 import Chat, get_ollama_models_list, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_OCR_MODEL_NAME

# --- Page Configuration ---
st.set_page_config(page_title="Enhanced Chat", layout="wide")
st.title("🚀 Enhanced Chat with Model & Session Management")

# --- Helper function to initialize/get Chat instance ---


@st.cache_resource  # Cache the chat instance based on model selections
def get_chat_instance(llm_model, embedding_model, ocr_model):
    chroma_path_abs = './chroma_store'
    print(
        f"Attempting to initialize Chat with: LLM={llm_model}, Embed={embedding_model}, OCR={ocr_model}")
    try:
        instance = Chat(chroma_path_abs,
                        llm_model_name=llm_model,
                        embedding_model_name=embedding_model,
                        ocr_model_name=ocr_model)
        return instance
    except Exception as e:
        st.error(f"Fatal Error initializing backend Chat class: {e}")
        print(f"Fatal Error initializing backend Chat class: {e}")
        traceback.print_exc()
        return None  # Return None if Chat class fails


# --- Session State Initialization ---
if "chats" not in st.session_state:
    # Stores all chat sessions {chat_id: {history: [], collection: None, doc_name: None}}
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "temp_file_paths" not in st.session_state:
    st.session_state.temp_file_paths = []

# Model selections in session state
if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = DEFAULT_LLM_MODEL
if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = DEFAULT_EMBEDDING_MODEL
if "selected_ocr_model" not in st.session_state:
    st.session_state.selected_ocr_model = DEFAULT_OCR_MODEL_NAME

# --- Fetch Ollama Models for Selectors ---
# @st.cache_data # Cache this list


def fetch_models():
    return get_ollama_models_list()


ollama_models = fetch_models()
if not ollama_models:
    st.sidebar.warning(
        "Could not fetch Ollama models. Using defaults. Ensure Ollama is running and has models.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration & Sessions")

    # --- Model Selection ---
    st.subheader("Model Selection")
    st.session_state.selected_llm_model = st.selectbox(
        "LLM Model:", options=ollama_models,
        index=ollama_models.index(
            st.session_state.selected_llm_model) if st.session_state.selected_llm_model in ollama_models else 0,
        key="llm_select"
    )
    st.session_state.selected_embedding_model = st.selectbox(
        # Ideally, filter for embedding-specific models if possible
        "Embedding Model:", options=ollama_models,
        index=ollama_models.index(
            st.session_state.selected_embedding_model) if st.session_state.selected_embedding_model in ollama_models else 0,
        key="embed_select"
    )
    st.session_state.selected_ocr_model = st.selectbox(
        "OCR/Vision Model:", options=ollama_models,  # Ideally, filter for vision models
        index=ollama_models.index(
            st.session_state.selected_ocr_model) if st.session_state.selected_ocr_model in ollama_models else 0,
        key="ocr_select"
    )
    if st.button("Apply Model Settings", key="apply_models"):
        st.cache_resource.clear()  # Clear cache to re-init Chat instance
        st.success(
            "Model settings applied. Chat instance will be re-initialized.")
        st.rerun()

    # --- Chat Session Management ---
    st.subheader("Chat Sessions")
    if st.button("➕ New Chat", key="new_chat"):
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = {
            "name": f"Chat {len(st.session_state.chats) + 1}",
            "history": [],
            "collection_name": None,
            "doc_name": None
        }
        st.session_state.active_chat_id = chat_id
        st.rerun()

    if st.session_state.chats:
        chat_options = {chat_id: data["name"]
                        for chat_id, data in st.session_state.chats.items()}
        # Ensure active_chat_id is valid, otherwise set to first available
        if st.session_state.active_chat_id not in chat_options and chat_options:
            st.session_state.active_chat_id = list(chat_options.keys())[0]

        selected_chat_id = st.selectbox(
            "Active Chat:",
            options=list(chat_options.keys()),
            format_func=lambda x: chat_options[x],
            index=list(chat_options.keys()).index(
                st.session_state.active_chat_id) if st.session_state.active_chat_id in chat_options else 0,
            key="select_chat_session"
        )
        if selected_chat_id != st.session_state.active_chat_id:
            st.session_state.active_chat_id = selected_chat_id
            st.rerun()

    # Initialize first chat if none exists
    if not st.session_state.active_chat_id and not st.session_state.chats:
        first_chat_id = str(uuid.uuid4())
        st.session_state.chats[first_chat_id] = {
            "name": "Chat 1", "history": [], "collection_name": None, "doc_name": None
        }
        st.session_state.active_chat_id = first_chat_id
        # No rerun needed here, will flow naturally

    # --- Document Management (operates on the active chat) ---
    if st.session_state.active_chat_id:
        active_chat_data = st.session_state.chats[st.session_state.active_chat_id]
        st.header(f"📁 Document Management for '{active_chat_data['name']}'")

        # Get Chat Instance (cached based on current model selections)
        chat_instance = get_chat_instance(
            st.session_state.selected_llm_model,
            st.session_state.selected_embedding_model,
            st.session_state.selected_ocr_model
        )
        if not chat_instance:
            st.error(
                "Chat backend could not be initialized. Please check model settings and Ollama server.")
            st.stop()  # Stop rendering further if backend is not up

        # Load Existing Collection
        st.subheader("Load Existing Collection")
        try:
            existing_collections = chat_instance.list_chroma_collections()
            if not existing_collections:
                st.info("No existing document collections.")
            else:
                options = [""] + [str(col) for col in existing_collections]
                selected_collection_load = st.selectbox(
                    "Select collection:", options=options, index=0, key=f"cl_{st.session_state.active_chat_id}")
                if selected_collection_load and st.button("Load Collection", key=f"lc_{st.session_state.active_chat_id}"):
                    active_chat_data["collection_name"] = selected_collection_load
                    active_chat_data["doc_name"] = f"Collection: {selected_collection_load}"
                    st.success(
                        f"Switched to collection: **{selected_collection_load}** for '{active_chat_data['name']}'")
                    st.rerun()
        except Exception as e:
            st.error(f"Error listing collections: {e}")

        # Switch to General Chat for current session
        if active_chat_data["collection_name"]:
            if st.button("Switch to General Chat (this session)", key=f"sgc_{st.session_state.active_chat_id}"):
                active_chat_data["collection_name"] = None
                active_chat_data["doc_name"] = None
                st.success(
                    f"'{active_chat_data['name']}' switched to General Chat mode.")
                st.rerun()

        # Process New Document(s)
        st.subheader("Process New Document(s)")
        uploaded_files = st.file_uploader("Upload files:", type=[
                                          "pdf", "txt"], key=f"fu_{st.session_state.active_chat_id}", accept_multiple_files=True)
        # ... (The multi-file upload and processing logic from ChatStreamlit02.py, adapted to use active_chat_data)
        # This section needs careful adaptation to ensure it modifies active_chat_data correctly.

        if uploaded_files:
            first_file_name = uploaded_files[0].name
            base_name = os.path.splitext(first_file_name)[0]
            sugg_col_name = base_name.replace(
                " ", "_").replace(".", "_").lower()
            if len(uploaded_files) > 1:
                sugg_col_name += "_batch"

            collection_name_input = st.text_input(
                "Collection name for uploaded document(s):", value=sugg_col_name, key=f"cn_{st.session_state.active_chat_id}")
            # ... (rest of the processing mode selection and button logic from previous version)
            # Ensure st.keys are unique per chat session to avoid conflicts if that's desired, or make them static.
            # For simplicity, let's make keys unique by appending active_chat_id for now.

            contains_pdf = any(os.path.splitext(f.name)[
                               1].lower() == ".pdf" for f in uploaded_files)
            processing_mode_for_pdfs = "text"  # Default
            if contains_pdf:
                pdf_mode_selection = st.radio(
                    "PDF Proc. Mode:", ("Text-Only", "OCR"), index=0, key=f"pmr_{st.session_state.active_chat_id}")
                processing_mode_for_pdfs = "ocr" if "OCR" in pdf_mode_selection else "text"
                if processing_mode_for_pdfs == "ocr":
                    st.warning("OCR requires working setup.", icon="⚙️")

            if st.button("Process Document(s)", key=f"pdb_{st.session_state.active_chat_id}"):
                if not collection_name_input:
                    st.error("Please enter a collection name.")
                else:
                    # Simplified processing loop - refer to ChatStreamlit02.py for full detail
                    st.session_state.temp_file_paths = []
                    files_processed_success_names = []
                    all_success_flag = True
                    with st.spinner(f"Processing {len(uploaded_files)} file(s) for '{active_chat_data['name']}'..."):
                        for i, up_file in enumerate(uploaded_files):
                            file_ext = os.path.splitext(
                                up_file.name)[1].lower()
                            current_file_proc_mode = processing_mode_for_pdfs if file_ext == ".pdf" else "text"
                            # ... (temp file creation)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                tmp_file.write(up_file.getvalue())
                                tmp_file_path = tmp_file.name
                                st.session_state.temp_file_paths.append(
                                    tmp_file_path)

                            print(
                                f"Processing {up_file.name} into {collection_name_input} with mode {current_file_proc_mode}")
                            success = chat_instance.index_pdf(
                                tmp_file_path, current_file_proc_mode, collection_name_input
                            )
                            if success:
                                files_processed_success_names.append(
                                    up_file.name)
                            else:
                                all_success_flag = False
                                st.error(f"Failed: {up_file.name}")
                                break
                    # ... (temp file cleanup)
                    for path in st.session_state.temp_file_paths:
                        if os.path.exists(path):
                            os.remove(path)
                    st.session_state.temp_file_paths = []

                    if all_success_flag and files_processed_success_names:
                        active_chat_data["collection_name"] = collection_name_input
                        active_chat_data[
                            "doc_name"] = f"Coll: {collection_name_input} ({len(files_processed_success_names)} files)"
                        st.success(
                            f"Processed to '{collection_name_input}' for chat '{active_chat_data['name']}'")
                        st.rerun()
                    elif files_processed_success_names:  # Partial
                        # Still set collection
                        active_chat_data["collection_name"] = collection_name_input
                        active_chat_data[
                            "doc_name"] = f"Coll: {collection_name_input} (Partial: {len(files_processed_success_names)} files)"
                        st.warning(
                            f"Partially processed to '{collection_name_input}' for '{active_chat_data['name']}'.")
                        st.rerun()
                    else:
                        st.error("No files processed successfully.")


# --- Main Chat Interface ---
if st.session_state.active_chat_id and st.session_state.chats:
    active_chat_data = st.session_state.chats[st.session_state.active_chat_id]
    st.header(f"💬 Chat: {active_chat_data['name']}")

    # Display current chat mode
    if active_chat_data["collection_name"]:
        st.info(f"Mode: Chatting with **{active_chat_data['doc_name']}**")
    else:
        st.info("Mode: **General Chat**")

    # Display chat history
    for message in active_chat_data["history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_prompt = st.chat_input(
        "Ask a question...", key=f"ci_{st.session_state.active_chat_id}")
    if user_prompt:
        active_chat_data["history"].append(
            {"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.spinner("🧠 Thinking..."):
            chat_instance_for_reply = get_chat_instance(  # Ensure instance is current
                st.session_state.selected_llm_model,
                st.session_state.selected_embedding_model,
                st.session_state.selected_ocr_model
            )
            if not chat_instance_for_reply:
                response = "Error: Backend Chat instance not available."
            elif active_chat_data["collection_name"]:
                response = chat_instance_for_reply.get_answer(
                    user_prompt, active_chat_data["collection_name"])
            else:
                response = chat_instance_for_reply.get_general_answer(
                    user_prompt, active_chat_data["history"][:-1])

            active_chat_data["history"].append(
                {"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            # No st.rerun() here, chat message updates are fine.

    if st.button("Clear This Chat's History", key=f"cch_{st.session_state.active_chat_id}"):
        active_chat_data["history"] = []
        st.rerun()
else:
    st.info("Create a new chat or select an existing one from the sidebar to begin.")
