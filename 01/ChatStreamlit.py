# ChatStreamlit02.py
import traceback
import streamlit as st
import os
import tempfile
from Chatapp import Chat  # Assuming Chatapp.py contains the Chat class

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Chat with Docs / General AI", layout="wide")
st.title("📄 Chat with Documents or General AI")

# --- Initialize Chat Backend ---
try:
    # Use an absolute path or a path relative to the script location if needed
    chat_instance = Chat('./chroma_store')
    # Ensure CHROMA_PATH in Chatapp.py also points correctly if not using the instance variable directly
except Exception as e:
    st.error(
        f"Failed to initialize backend Chat class from ./chroma_store: {e}")
    st.stop()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    # Stores [{"role": "user", "content": "..."}, ...]
    st.session_state.chat_history = []
if "current_collection_name" not in st.session_state:
    st.session_state.current_collection_name = None
if "current_doc_name" not in st.session_state:
    # Represents the document(s) or collection currently being chatted with
    st.session_state.current_doc_name = None
if "temp_file_paths" not in st.session_state:
    st.session_state.temp_file_paths = []  # Store multiple temp file paths

# --- Sidebar for Document Management ---
with st.sidebar:
    st.header("📁 Document Management")

    # --- Load Existing Collection ---
    st.subheader("Load Existing Collection")
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
        existing_collections = []

    # Option to switch back to General Chat
    if st.session_state.current_collection_name:
        if st.button("Switch to General Chat", key="general_chat_button"):
            st.session_state.current_collection_name = None
            st.session_state.current_doc_name = None
            # st.session_state.chat_history = []  # Reset chat history
            st.success("Switched to General Chat mode.")
            st.rerun()

    # --- Process New Document(s) ---
    st.subheader("Process New Document(s)")
    # --- Allow Multiple Files ---
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files:",
        type=["pdf", "txt"],
        key="doc_uploader",
        accept_multiple_files=True  # <-- Key change
    )

    processing_mode = "text"  # Default
    processing_mode_selection = None
    collection_name = None  # Initialize collection name

    # --- Logic to handle multiple files ---
    if uploaded_files:  # Check if list is not empty
        # Suggest collection name based on the first file, or a generic name
        first_file_name = uploaded_files[0].name
        base_name = os.path.splitext(first_file_name)[0]
        suggested_collection_name = base_name.replace(
            " ", "_").replace(".", "_").lower()
        # if len(uploaded_files) > 1:
        # suggested_collection_name += "_batch"  # Indicate multiple files
        collection_name = st.text_input(
            "Enter collection name for ALL uploaded document(s):",
            value=suggested_collection_name,
            key="collection_name_input"
        )

        # Check if any PDFs are present to offer OCR option
        contains_pdf = any(os.path.splitext(f.name)[
                           1].lower() == ".pdf" for f in uploaded_files)
        only_txt = all(os.path.splitext(f.name)[
                       1].lower() == ".txt" for f in uploaded_files)

        if contains_pdf:
            processing_mode_selection = st.radio(
                "PDF Processing Mode (applies to all PDFs in batch):",
                ("Text-Only (Faster)", "Contains Images/OCR (Slower)"),
                index=0, key="processing_mode_radio"
            )
            processing_mode = "ocr" if "OCR" in processing_mode_selection else "text"
            if processing_mode == "ocr":
                st.warning(
                    "OCR requires a functioning setup (e.g., OCRProcessor or Tesseract in PATH).", icon="⚙️")
        elif only_txt:
            processing_mode = "text"
            st.info("Processing TXT file(s) in text mode.")
            # Set this to enable button
            processing_mode_selection = "Text-Only (Faster)"
        else:  # Should not happen if type=["pdf", "txt"]
            # Default to enable button
            processing_mode_selection = "Text-Only (Faster)"

        # Enable process button
        process_button_disabled = not (
            collection_name and processing_mode_selection and uploaded_files)
        if st.button("Process Document(s)", key="process_button", disabled=process_button_disabled):
            st.session_state.temp_file_paths = []  # Clear previous temp paths
            files_processed_success = []
            files_processed_error = []
            total_files = len(uploaded_files)
            progress_bar = st.progress(0, text="Starting processing...")

            all_success = True  # Flag to track if all files processed ok
            for i, uploaded_file in enumerate(uploaded_files):
                file_extension = os.path.splitext(
                    uploaded_file.name)[1].lower()
                # Determine mode for THIS file (TXT always text, PDF uses selection)
                current_file_mode = processing_mode if file_extension == ".pdf" else "text"
                print("📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄📄")
                print(uploaded_file.name)
                progress_text = f"Processing file {i+1}/{total_files}: '{uploaded_file.name}' (Mode: {current_file_mode})..."
                st.spinner(progress_text)
                progress_bar.progress((i / total_files), text=progress_text)

                # Create temp file for each uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                    st.session_state.temp_file_paths.append(
                        tmp_file_path)  # Store for cleanup

                try:
                    # Call the backend index function FOR EACH FILE
                    # All files go into the SAME collection_name

                    success = chat_instance.index_pdf(
                        file_path=tmp_file_path,
                        collection_name=collection_name,  # Use the single collection name
                        processing_mode=current_file_mode  # Use mode specific to this file type
                    )

                    if success:
                        files_processed_success.append(uploaded_file.name)
                    else:
                        all_success = False
                        files_processed_error.append(uploaded_file.name)
                        st.error(
                            f"❌ Failed to process '{uploaded_file.name}'. Check console logs. Stopping batch.")
                        break  # Stop processing batch on first error

                except Exception as e:
                    all_success = False
                    files_processed_error.append(uploaded_file.name)
                    st.error(
                        f"An error occurred during processing of '{uploaded_file.name}': {e}")
                    traceback.print_exc()  # Print traceback for debugging
                    break  # Stop processing batch on error

            # Update progress bar final state
            progress_bar.progress(
                1.0, text=f"Processing finished. {len(files_processed_success)} succeeded, {len(files_processed_error)} failed.")

            # Final status update based on overall success
            if all_success and files_processed_success:
                st.session_state.current_collection_name = collection_name
                # Update doc name to reflect the collection/batch
                st.session_state.current_doc_name = f"Collection: {collection_name} ({len(files_processed_success)} files)"
                # !!! REMOVED chat history reset: st.session_state.chat_history = []
                st.success(
                    f"✅ Processed {len(files_processed_success)} file(s) into collection '{collection_name}'. Ready to chat.")
                st.rerun()  # Rerun to reflect the change
            elif files_processed_success:  # Partial success
                st.warning(
                    f"Processed {len(files_processed_success)} file(s) successfully, but failed on: {', '.join(files_processed_error)}. Check logs.")
                # Decide if you want to switch to the partially populated collection
                st.session_state.current_collection_name = collection_name
                st.session_state.current_doc_name = f"Collection: {collection_name} ({len(files_processed_success)} files, partial)"
                st.rerun()
            else:  # Complete failure
                st.error(
                    f"❌ Failed to process any documents. Failed file(s): {', '.join(files_processed_error)}")
                # Don't switch collection on complete failure

            # Clean up ALL temp files attempted in this batch
            if 'temp_file_paths' in st.session_state:
                for path in st.session_state.temp_file_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError as oe:
                            st.warning(
                                f"Could not delete temp file: {path} - {oe}")
                st.session_state.temp_file_paths = []  # Clear list after attempting cleanup

# --- Main Chat Interface ---
st.header("💬 Chat Window")

# Display current chat mode
if st.session_state.current_collection_name:
    # Updated doc name display
    st.info(f"Mode: Chatting with **{st.session_state.current_doc_name}**")
else:
    st.info("Mode: **General Chat** (Not using documents)")

# Display chat history (works for both modes, now persistent)
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
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

# --- Optional: Clear Chat History ---
if st.button("Clear Chat History", key="clear_chat_button"):
    st.session_state.chat_history = []
    st.rerun()
