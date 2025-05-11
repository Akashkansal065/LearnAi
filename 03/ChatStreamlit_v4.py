# ChatStreamlit_v4.py
import streamlit as st
import os
import tempfile
import uuid
import traceback

# Import backend and auth components
from Chatapp_v4 import Chat, get_ollama_models_list, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_OCR_MODEL_NAME
import auth  # Your new authentication module

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Chat UI",
                   layout="wide", initial_sidebar_state="expanded")

# --- Load Custom CSS (Example - create style.css later) ---


def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

# load_css("style.css") # We'll create this file later

# --- Helper function to initialize/get Chat instance ---


@st.cache_resource
# Added trigger for cache invalidation
def get_chat_instance(llm_model, embedding_model, ocr_model, _trigger_reset):
    script_dir = os.path.dirname(__file__)
    # Use a new chroma path to avoid conflicts if schema changes or for fresh start
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
        return None


# --- Session State Initialization (Defaults) ---
# These will be set after successful login if not already present
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "temp_file_paths" not in st.session_state:
    st.session_state.temp_file_paths = []

# Model selections (global)
if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = DEFAULT_LLM_MODEL
if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = DEFAULT_EMBEDDING_MODEL
if "selected_ocr_model" not in st.session_state:
    st.session_state.selected_ocr_model = DEFAULT_OCR_MODEL_NAME
if "model_settings_applied_trigger" not in st.session_state:
    st.session_state.model_settings_applied_trigger = 0


# --- Authentication Check ---
if not st.session_state.get('authenticated', False):
    auth.login_form()
else:
    # --- Main Application UI (Rendered only if authenticated) ---

    # Initialize first chat if none exists and authenticated
    if not st.session_state.active_chat_id and not st.session_state.chats:
        first_chat_id = str(uuid.uuid4())
        st.session_state.chats[first_chat_id] = {
            "name": "Chat 1", "history": [], "collection_name": None, "doc_name": None
        }
        st.session_state.active_chat_id = first_chat_id

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://ollama.com/public/ollama.png",
                 width=100)  # Example logo
        st.markdown(
            f"### Welcome, {st.session_state.get('username', 'User')}!")
        st.divider()

        # --- New Chat Button ---
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_main_btn"):
            chat_id = str(uuid.uuid4())
            st.session_state.chats[chat_id] = {
                "name": f"Chat {len(st.session_state.chats) + 1}",
                "history": [], "collection_name": None, "doc_name": None
            }
            st.session_state.active_chat_id = chat_id
            st.rerun()

        st.markdown("##### Chat Sessions")
        # --- Chat Session List ---
        if st.session_state.chats:
            chat_options = {chat_id: data["name"]
                            for chat_id, data in st.session_state.chats.items()}
            # Ensure active_chat_id is valid
            if st.session_state.active_chat_id not in chat_options and chat_options:
                st.session_state.active_chat_id = list(chat_options.keys())[0]

            # Display chat sessions as a list of buttons or a selectbox
            # For a more OpenWebUI feel, a list of clickable items is better.
            # Streamlit doesn't have a perfect "list of clickable items" widget.
            # We can simulate it using buttons in a loop, or use st.radio/st.selectbox.
            # For simplicity here, let's use st.radio for now, styled later if possible.

            current_chat_index = 0
            if st.session_state.active_chat_id:
                try:
                    current_chat_index = list(chat_options.keys()).index(
                        st.session_state.active_chat_id)
                except ValueError:
                    current_chat_index = 0  # Default if ID not found

            selected_chat_id_radio = st.radio(
                "Select Chat:",
                options=list(chat_options.keys()),
                format_func=lambda x: chat_options[x],
                index=current_chat_index,
                key="select_chat_radio",
                label_visibility="collapsed"
            )
            if selected_chat_id_radio != st.session_state.active_chat_id:
                st.session_state.active_chat_id = selected_chat_id_radio
                st.rerun()
        else:
            st.caption("No chats yet. Click 'New Chat' to start.")

        st.divider()

        # --- Model Selection (Moved to an expander for cleaner sidebar) ---
        with st.expander("⚙️ Model Settings", expanded=False):
            ollama_models = get_ollama_models_list()  # Fetch models once
            if not ollama_models:
                st.warning("Ollama models not found. Using defaults.")
                # Provide default list if Ollama fetch fails, to prevent selectbox errors
                ollama_models = [DEFAULT_LLM_MODEL,
                                 DEFAULT_EMBEDDING_MODEL, DEFAULT_OCR_MODEL_NAME]

            # Helper to find index or default to 0

            def get_model_index(model_list, selected_model_session_key):
                model_name = st.session_state[selected_model_session_key]
                try:
                    return model_list.index(model_name)
                # Model not in list (e.g. pulled model deleted)
                except ValueError:
                    if model_list:  # if ollama_models is not empty
                        # Fallback to first model
                        st.session_state[selected_model_session_key] = model_list[0]
                        return 0
                    else:  # if ollama_models is empty
                        # Fallback to empty string
                        st.session_state[selected_model_session_key] = ""
                        return 0

            selected_llm = st.selectbox("LLM Model:", options=ollama_models, index=get_model_index(
                ollama_models, "selected_llm_model"), key="sb_llm")
            selected_embed = st.selectbox("Embedding Model:", options=ollama_models, index=get_model_index(
                ollama_models, "selected_embedding_model"), key="sb_embed")
            selected_ocr = st.selectbox("OCR/Vision Model:", options=ollama_models,
                                        index=get_model_index(ollama_models, "selected_ocr_model"), key="sb_ocr")

            if st.button("Apply Model Settings", use_container_width=True, key="apply_models_btn"):
                st.session_state.selected_llm_model = selected_llm
                st.session_state.selected_embedding_model = selected_embed
                st.session_state.selected_ocr_model = selected_ocr
                st.session_state.model_settings_applied_trigger += 1  # Invalidate cache
                st.success(
                    "Model settings applied. Backend will re-initialize.")
                st.rerun()

        st.divider()
        # --- Document Management for Active Chat (moved to expander) ---
        if st.session_state.active_chat_id and st.session_state.chats:
            with st.expander("📄 Document Management", expanded=False):
                active_chat_data = st.session_state.chats[st.session_state.active_chat_id]
                # This needs the chat_instance. We get it based on current model selections.
                chat_instance = get_chat_instance(
                    st.session_state.selected_llm_model,
                    st.session_state.selected_embedding_model,
                    st.session_state.selected_ocr_model,
                    st.session_state.model_settings_applied_trigger  # Pass trigger to cache
                )

                if not chat_instance:
                    st.error(
                        "Chat backend could not be initialized. Check Ollama server and model settings.")
                else:
                    # Load Existing Collection
                    st.caption("Load Existing Collection")
                    existing_collections = chat_instance.list_chroma_collections()
                    if not existing_collections:
                        st.info("No collections found.")
                    else:
                        options = [""] + [str(col)
                                          for col in existing_collections]
                        sel_col_load_key = f"sel_col_load_{st.session_state.active_chat_id}"
                        selected_collection_load = st.selectbox(
                            "Collection:", options=options, index=0, key=sel_col_load_key)

                        load_btn_key = f"load_btn_{st.session_state.active_chat_id}"
                        if st.button("Load Collection", use_container_width=True, key=load_btn_key):
                            if selected_collection_load:
                                active_chat_data["collection_name"] = selected_collection_load
                                active_chat_data["doc_name"] = f"Collection: {selected_collection_load}"
                                st.success(
                                    f"Switched to collection: **{selected_collection_load}**")
                                st.rerun()
                            else:
                                st.warning(
                                    "Please select a collection to load.")

                    # Switch to General Chat for current session
                    if active_chat_data["collection_name"]:
                        if st.button("Switch to General Chat Mode", use_container_width=True, key=f"sgc_btn_{st.session_state.active_chat_id}"):
                            active_chat_data["collection_name"] = None
                            active_chat_data["doc_name"] = None
                            st.success(f"Switched to General Chat mode.")
                            st.rerun()

                    st.caption("Process New Document(s)")
                    # --- Process New Document(s) ---
                    # (This section needs the multi-file upload from ChatStreamlit_v3.py, adapted)
                    # For brevity, I'll put a simplified version. Refer to full logic.
                    upload_key = f"upload_{st.session_state.active_chat_id}"
                    uploaded_files = st.file_uploader("Upload files:", type=[
                                                      "pdf", "txt"], key=upload_key, accept_multiple_files=True)

                    if uploaded_files:
                        coll_name_key = f"coll_name_{st.session_state.active_chat_id}"
                        sugg_name = os.path.splitext(uploaded_files[0].name)[0].replace(
                            " ", "_") + ("_batch" if len(uploaded_files) > 1 else "")
                        collection_name_input = st.text_input(
                            "New Collection Name:", value=sugg_name, key=coll_name_key)

                        pdf_mode_key = f"pdf_mode_{st.session_state.active_chat_id}"
                        processing_mode_for_pdfs = "text"
                        if any(f.name.lower().endswith(".pdf") for f in uploaded_files):
                            pdf_mode_selection = st.radio(
                                "PDF Mode:", ("Text-Only", "OCR"), index=0, key=pdf_mode_key, horizontal=True)
                            processing_mode_for_pdfs = "ocr" if "OCR" in pdf_mode_selection else "text"

                        process_btn_key = f"proc_btn_{st.session_state.active_chat_id}"
                        if st.button("Process Uploaded File(s)", use_container_width=True, key=process_btn_key):
                            if not collection_name_input:
                                st.error("Collection name is required.")
                            elif not uploaded_files:
                                st.error("No files uploaded.")
                            else:
                                # Simplified processing loop - for detailed version see previous response
                                with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                                    all_success = True
                                    processed_count = 0
                                    for up_file in uploaded_files:
                                        file_ext = os.path.splitext(
                                            up_file.name)[1].lower()
                                        current_file_mode = processing_mode_for_pdfs if file_ext == ".pdf" else "text"
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_f:
                                            tmp_f.write(up_file.getvalue())
                                            tmp_f_path = tmp_f.name

                                        success = chat_instance.index_pdf(
                                            tmp_f_path, current_file_mode, collection_name_input)
                                        # Clean up temp file
                                        os.remove(tmp_f_path)
                                        if success:
                                            processed_count += 1
                                        else:
                                            all_success = False
                                            st.error(f"Failed: {up_file.name}")
                                            break

                                if all_success and processed_count > 0:
                                    active_chat_data["collection_name"] = collection_name_input
                                    active_chat_data[
                                        "doc_name"] = f"Coll: {collection_name_input} ({processed_count} files)"
                                    st.success(
                                        f"Processed {processed_count} file(s) to '{collection_name_input}'.")
                                    st.rerun()
                                elif processed_count > 0:  # Partial success
                                    # Still set collection
                                    active_chat_data["collection_name"] = collection_name_input
                                    active_chat_data[
                                        "doc_name"] = f"Coll: {collection_name_input} (Partial: {processed_count} files)"
                                    st.warning(
                                        f"Partially processed to '{collection_name_input}'.")
                                    st.rerun()
                                else:
                                    st.error(
                                        "No files were processed successfully.")

        st.divider()
        auth.logout_button()  # Logout button from auth module

    # --- Main Chat Area ---
    if st.session_state.active_chat_id and st.session_state.chats:
        active_chat_data = st.session_state.chats[st.session_state.active_chat_id]

        # Retrieve the chat instance again, ensures it uses current model selections
        chat_instance_main = get_chat_instance(
            st.session_state.selected_llm_model,
            st.session_state.selected_embedding_model,
            st.session_state.selected_ocr_model,
            st.session_state.model_settings_applied_trigger
        )

        if not chat_instance_main:
            st.header(
                f"⚠️ Chat Backend Error for '{active_chat_data['name']}'")
            st.error(
                "Could not initialize the chat backend. Please check model settings and ensure Ollama is running.")
        else:
            st.header(f"💬 Chat: {active_chat_data['name']}")
            if active_chat_data["collection_name"]:
                st.caption(
                    f"Mode: Chatting with **{active_chat_data['doc_name']}** | LLM: `{st.session_state.selected_llm_model}`")
            else:
                st.caption(
                    f"Mode: **General Chat** | LLM: `{st.session_state.selected_llm_model}`")

            # Chat messages display
            for message in active_chat_data["history"]:
                # avatar=USER_AVATAR if message["role"] == "user" else BOT_AVATAR):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User input
            prompt_key = f"prompt_{st.session_state.active_chat_id}"
            if user_prompt := st.chat_input("Your message...", key=prompt_key):
                active_chat_data["history"].append(
                    {"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.spinner("Thinking..."):
                    if active_chat_data["collection_name"]:
                        response = chat_instance_main.get_answer(
                            user_prompt, active_chat_data["collection_name"])
                    else:
                        response = chat_instance_main.get_general_answer(
                            user_prompt, active_chat_data["history"][:-1])

                    active_chat_data["history"].append(
                        {"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    # No st.rerun() here, new messages appear automatically

            if st.button("Clear This Chat's History", key=f"clear_hist_btn_{st.session_state.active_chat_id}"):
                active_chat_data["history"] = []
                st.rerun()
    else:
        st.info(
            "Create a new chat or select an existing one from the sidebar to begin.")
