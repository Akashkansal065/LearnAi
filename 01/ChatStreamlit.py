# app.py
import streamlit as st
import os
import tempfile
from Chatapp import Chat

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Chat with PDF (Local RAG)", layout="wide")
st.title("📄 Chat with your PDF using Ollama & ChromaDB")
chat_instance = Chat('./chroma_store')


# --- Session State Initialization ---
# Stores chat history and the currently active document collection name
existing_collections = chat_instance.list_chroma_collections()

if existing_collections:
    selected_collection = st.selectbox(
        "Or select an existing collection:", existing_collections)
    if st.button("💬 Load Selected Collection"):
        st.session_state.current_collection_name = selected_collection
        st.session_state.current_pdf_name = f"{selected_collection}.pdf"
        st.session_state.chat_history = []
        st.success(f"Loaded collection: {selected_collection}")
else:
    st.info("No existing collections found in ChromaDB.")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_collection_name" not in st.session_state:
    st.session_state.current_collection_name = None
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = "pdf"  # Default mode
# --- Sidebar for PDF Upload and Processing ---
if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = "text"  # Default mode

# --- Sidebar for Document Upload and Processing ---
with st.sidebar:
    st.header("📁 Document Setup")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        key="doc_uploader"
    )

    if uploaded_file is not None:
        st.subheader("🗂️ Collection Setup")

        collection_mode = st.radio(
            "Choose collection handling:",
            ["Use Existing Collection", "Create New Collection"],
            index=1,  # Default to create new
            key="collection_mode"
        )

        collection_name = None

        if collection_mode == "Use Existing Collection":
            if existing_collections:
                collection_name = st.selectbox(
                    "Select an existing collection:", existing_collections)
            else:
                st.warning(
                    "No collections available. Please create a new one.")
        else:
            base_name = os.path.splitext(uploaded_file.name)[0]
            default_name = base_name.replace(
                " ", "_").replace(".", "_").lower()
            collection_name = st.text_input(
                "Enter new collection name:", value=default_name)

        if collection_name:
            st.session_state.proposed_collection_name = collection_name

            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension == ".pdf":
                processing_mode_selection = st.radio(
                    "Select processing mode for this PDF:",
                    ("Text-Only (Faster)",
                     "Contains Images/OCR (Slower, requires Tesseract)"),
                    index=0,
                    key="processing_mode_radio"
                )
                st.session_state.processing_mode = "ocr" if "OCR" in processing_mode_selection else "text"

                if st.session_state.processing_mode == "ocr":
                    st.warning(
                        "⚠️ OCR mode selected. Ensure Tesseract OCR is installed and accessible in your system's PATH.",
                        icon="⚙️"
                    )

            elif file_extension == ".txt":
                st.session_state.processing_mode = "text"
                st.info("Processing TXT file in text mode.")
                processing_mode_selection = "Text-Only (Faster)"

            # Enable button only when everything is set
            process_button_disabled = not (
                uploaded_file and processing_mode_selection and collection_name)

            if st.button("Process Document", key="process_button", disabled=process_button_disabled):
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                    st.session_state.temp_file_path = tmp_file_path

                progress_bar = st.progress(0, text="Starting processing...")

                try:
                    processing_mode_to_use = st.session_state.processing_mode
                    with st.spinner(f"Processing '{uploaded_file.name}' (Mode: {processing_mode_to_use})..."):
                        progress_bar.progress(
                            10, text=f"Using {processing_mode_to_use} mode...")

                        success = chat_instance.index_pdf(
                            file_path=tmp_file_path,
                            collection_name=collection_name,
                            processing_mode=processing_mode_to_use
                        )
                        progress_bar.progress(
                            75, text="Embedding & storing...")

                        if success:
                            st.session_state.current_collection_name = collection_name
                            st.session_state.current_pdf_name = uploaded_file.name
                            st.session_state.chat_history = []
                            st.success(
                                f"✅ Successfully processed '{uploaded_file.name}'. Ready to chat!")
                        else:
                            st.error(
                                f"❌ Failed to process '{uploaded_file.name}'. Check backend logs.")
                            st.session_state.current_collection_name = None
                            st.session_state.current_pdf_name = None

                        progress_bar.progress(100, text="Processing complete.")
                        progress_bar.empty()

                except Exception as e:
                    st.error(
                        f"An unexpected error occurred during processing: {e}")
                    progress_bar.empty()
                finally:
                    if 'temp_file_path' in st.session_state and os.path.exists(st.session_state.temp_file_path):
                        os.remove(st.session_state.temp_file_path)
                        del st.session_state.temp_file_path

    # Display the currently active document
    if st.session_state.current_pdf_name:
        st.sidebar.success(
            f"Chatting with: **{st.session_state.current_pdf_name}**"
        )
        st.sidebar.caption(
            f"(Collection: `{st.session_state.current_collection_name}`)"
        )
    else:
        st.sidebar.warning("Upload and process a PDF to begin.")


# --- Main Chat Interface ---
st.header("💬 Chat Window")

# Display chat history
if st.session_state.current_collection_name:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("Upload and process a document using the sidebar to start chatting.")

# Get user input
user_prompt = st.chat_input("Ask a question about the processed document...")

if user_prompt:
    if st.session_state.current_collection_name:
        # Add user message to chat history and display it
        st.session_state.chat_history.append(
            {"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Get response from backend
        with st.spinner("🧠 Thinking..."):
            try:
                # Call the backend function to get the answer

                response = chat_instance.get_answer(
                    question=user_prompt, collection_name=st.session_state.current_collection_name)

                # Display assistant response and add to history
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred while getting the answer: {e}")
                # Optionally add error to chat history
                # st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

    else:
        st.warning(
            "⚠️ Please upload and process a PDF document first before asking questions.")

# --- Optional: Clear Chat History ---
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     st.rerun()
