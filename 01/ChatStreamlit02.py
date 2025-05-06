# ChatStreamlit.py (equivalent to app.py)
import streamlit as st
import os
import tempfile
from Chatapp import Chat  # Assuming Chatapp.py contains the Chat class

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Chat with Docs / General AI", layout="wide")
st.title("📄 Chat with Documents or General AI")

# --- Initialize Chat Backend ---
# Ensure CHROMA_PATH used here matches the one in Chatapp.py if not using env vars
try:
    chat_instance = Chat('./chroma_store')
except Exception as e:
    st.error(f"Failed to initialize backend Chat class: {e}")
    st.stop()  # Stop the app if backend fails to initialize

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    # Stores [{"role": "user", "content": "..."}, ...]
    st.session_state.chat_history = []
if "current_collection_name" not in st.session_state:
    st.session_state.current_collection_name = None
if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = None
# Remove processing_mode from here if managed within the sidebar logic
# if "processing_mode" not in st.session_state:
#     st.session_state.processing_mode = "text"

# --- Sidebar for Document Management ---
with st.sidebar:
    st.header("📁 Document Management")

    # --- Load Existing Collection ---
    st.subheader("Load Existing Document")
    try:
        existing_collections = chat_instance.list_chroma_collections()
        if not existing_collections:
            st.info("No existing document collections found.")
            selected_collection_load = None
        else:
            selected_collection_load = st.selectbox(
                "Select a document collection to chat with:",
                options=[""] + existing_collections,  # Add blank option
                index=0,  # Default to blank
                key="collection_loader"
            )
            if selected_collection_load and st.button("Load Selected", key="load_button"):
                st.session_state.current_collection_name = selected_collection_load
                # Try to infer doc name, otherwise use collection name
                st.session_state.current_doc_name = f"{selected_collection_load} (loaded)"
                # st.session_state.chat_history = []  # Reset chat history
                st.success(
                    f"Switched to collection: **{selected_collection_load}**")
                st.rerun()  # Rerun to update main page status

    except Exception as e:
        st.error(f"Error listing collections: {e}")
        existing_collections = []  # Ensure it's a list even on error

    # Option to switch back to General Chat
    if st.session_state.current_collection_name:
        if st.button("Switch to General Chat", key="general_chat_button"):
            st.session_state.current_collection_name = None
            st.session_state.current_doc_name = None
            # st.session_state.chat_history = []  # Reset chat history
            st.success("Switched to General Chat mode.")
            st.rerun()

    # --- Process New Document ---
    st.subheader("Process New Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT:",
        type=["pdf", "txt"],
        key="doc_uploader"
    )

    processing_mode = "text"  # Default
    processing_mode_selection = None

    if uploaded_file is not None:
        base_name = os.path.splitext(uploaded_file.name)[0]
        default_collection_name = base_name.replace(
            " ", "_").replace(".", "_").lower()
        collection_name = st.text_input(
            "Enter collection name for this document:",
            value=default_collection_name,
            key="collection_name_input"
        )

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            processing_mode_selection = st.radio(
                "PDF Processing Mode:",
                ("Text-Only (Faster)", "Contains Images/OCR (Slower)"),
                index=0, key="processing_mode_radio"
            )
            processing_mode = "ocr" if "OCR" in processing_mode_selection else "text"
            if processing_mode == "ocr":
                st.warning(
                    "OCR requires Tesseract installed & in PATH.", icon="⚙️")
        elif file_extension == ".txt":
            processing_mode = "text"
            st.info("Processing TXT file in text mode.")
            # To enable button
            processing_mode_selection = "Text-Only (Faster)"

        # Enable button
        process_button_disabled = not (
            collection_name and processing_mode_selection)
        if st.button("Process Document", key="process_button", disabled=process_button_disabled):
            # --- (Processing Logic - largely unchanged from previous version) ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                st.session_state.temp_file_path = tmp_file_path  # Store for cleanup

            progress_bar = st.progress(0, text="Starting processing...")
            try:
                with st.spinner(f"Processing '{uploaded_file.name}' (Mode: {processing_mode})..."):
                    progress_bar.progress(
                        10, text=f"Using {processing_mode} mode...")
                    # Call the backend index function
                    success = chat_instance.index_pdf(
                        file_path=tmp_file_path,
                        collection_name=collection_name,
                        processing_mode=processing_mode
                    )
                    progress_bar.progress(75, text="Embedding & storing...")

                    if success:
                        st.session_state.current_collection_name = collection_name
                        st.session_state.current_doc_name = uploaded_file.name
                        # st.session_state.chat_history = []  # Reset chat
                        st.success(
                            f"✅ Processed '{uploaded_file.name}'. Switched to this document.")
                        st.rerun()  # Rerun to reflect the change immediately
                    else:
                        st.error(
                            f"❌ Failed to process '{uploaded_file.name}'. Check console logs.")
                        # Don't change current collection if processing fails
                    progress_bar.progress(100, text="Processing complete.")
                    progress_bar.empty()
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                if progress_bar:
                    progress_bar.empty()
            finally:
                # Clean up temp file
                if 'temp_file_path' in st.session_state and st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
                    try:
                        os.remove(st.session_state.temp_file_path)
                        del st.session_state.temp_file_path
                    except OSError as oe:
                        st.warning(f"Could not delete temp file: {oe}")
            # --- (End of Processing Logic) ---


# --- Main Chat Interface ---
st.header("💬 Chat Window")

# Display current chat mode
if st.session_state.current_collection_name:
    st.info(
        f"Mode: Chatting with Document **'{st.session_state.current_doc_name}'** (Collection: `{st.session_state.current_collection_name}`)")
else:
    st.info("Mode: **General Chat** (Not using documents)")

# Display chat history (works for both modes)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_prompt = st.chat_input("Ask a question...")

if user_prompt:
    # Add user message to history immediately
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Determine which backend method to call
    with st.spinner("🧠 Thinking..."):
        try:
            if st.session_state.current_collection_name:
                # --- RAG Mode ---
                response = chat_instance.get_answer(
                    question=user_prompt,
                    collection_name=st.session_state.current_collection_name
                )
            else:
                # --- General Chat Mode ---
                # Pass the current chat history (excluding the latest user prompt for context)
                response = chat_instance.get_general_answer(
                    question=user_prompt,
                    # Pass history before current question
                    chat_history_list=st.session_state.chat_history[:-1]
                )

            # Display assistant response and add to history
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Optionally add error to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
            # Ensure spinner stops by rerunning or just letting it finish
            # st.rerun() # Use cautiously, might clear spinner too early

# --- Optional: Clear Chat History ---
# Placed outside sidebar for general access
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
