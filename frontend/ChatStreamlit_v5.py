import httpx
import re
import time
from streamlit_js_eval import streamlit_js_eval
import streamlit as st
import os
import uuid
import traceback
import httpx  # For making API calls to FastAPI
import json  # For parsing JSON from cookies if needed
import asyncio  # Added for running async functions


# --- Page Configuration ---
st.set_page_config(page_title="Advanced Chat UI - FastAPI Backend",
                   layout="wide", initial_sidebar_state="expanded")

# --- Environment & API Base URL ---
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8080")
STREAMLIT_SERVER_ADDRESS = os.getenv(
    "STREAMLIT_SERVER_ADDRESS", "http://localhost:8501")


# --- Load Custom CSS ---
def load_css(file_name):
    try:
        css_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(
            f"CSS file '{file_name}' not found at {css_path}. Using default styles.")


load_css("style.css")


# --- Session State Initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
# No need to store auth_token or cookies_for_api if relying solely on HttpOnly cookies
# and FastAPI handles them correctly with browser.

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

DEFAULT_LLM_MODEL_ST = "llama3:8b"
DEFAULT_EMBEDDING_MODEL_ST = "nomic-embed-text"
DEFAULT_OCR_MODEL_ST = "llava:latest"

if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = DEFAULT_LLM_MODEL_ST
if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = DEFAULT_EMBEDDING_MODEL_ST
if "selected_ocr_model" not in st.session_state:
    st.session_state.selected_ocr_model = DEFAULT_OCR_MODEL_ST

# --- Asynchronous API Call Helper Functions ---


def clear_session_token_cookie():
    """Delete session_token cookie via JS."""
    streamlit_js_eval(
        js_expressions="""
            document.cookie = 'session_token=; Max-Age=0; path=/';
        """,
        key="clear_cookie"
    )


def ensure_token_ready():
    """Wait for session_token cookie via JS and store in session_state."""
    if not st.session_state.auth_token:
        session_token = streamlit_js_eval(
            js_expressions="document.cookie", key="get_cookie")
        if session_token:
            match = re.search(r'session_token=([^;]+)', session_token)
            if match:
                token = match.group(1)
                print("Session token cookie:", token)  # Debug log
                st.session_state.auth_token = token
                # st.success("Session token retrieved.")
            else:
                st.warning("Session token cookie missing or malformed.")
        else:
            st.info("Waiting for cookies to be received...")


def token():
    # print(st.session_state.get("auth_token"))
    return st.session_state.get("auth_token")


# --- Ensure token before any API interaction ---
ensure_token_ready()


async def fetch_user_me_api(token=None):
    """Fetches user data if authenticated."""
    try:
        head = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL, follow_redirects=True) as client:
            response = await client.get("/users/me", headers=head)
            if response.status_code == 200:
                return response.json()
            else:
                if response.status_code == 401:
                    st.session_state.auth_token = None
                    clear_session_token_cookie()
                    st.warning("Session expired. Please log in again.")
                print(
                    f"/users/me call failed: {response.status_code} {response.text}")
                return None
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error fetching user data: {e}")
        return None
    except httpx.RequestError as e:
        st.error(f"Connection error fetching user: {e}")
        return None


async def fetch_ollama_models_api():
    print("Fetching available models...")  # Debug log
    try:
        # head = {"Authorization": f"Bearer {token}"}
        token_value = token()
        print("Token for fetching models:", token_value, flush=True)
        async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL) as client:
            response = await client.get("/ollama-models", headers={"Authorization": f"Bearer {token_value}"})
            response.raise_for_status()
            return response.json().get("models", [])
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error fetching user data: {e}")
        return None
    except httpx.RequestError as e:
        st.error(f"Connection error fetching user: {e}")
        return None


async def call_index_file_api(payload, files_data):
    token_value = token()  # Assuming you have a function to get the token
    # Fine-tuned timeouts
    timeout = httpx.Timeout(
        connect=30.0,  # Time to establish connection
        read=2300.0,    # Time to wait for server response
        write=2300.0,   # Time to upload file
        pool=2310.0     # Total timeout for the request
    )
    async with httpx.AsyncClient(timeout=300.0, base_url=FASTAPI_BASE_URL) as client:
        # Separate query parameters (collection_name) and form data
        query_params = {
            "collection_name": payload["collection_name"],
            "processing_mode": payload["processing_mode"],
            "llm_model_name": payload["llm_model_name"],
            "embedding_model_name": payload["embedding_model_name"],
            "ocr_model_name": payload["ocr_model_name"],
        }

        form_data = {
            "collection_name": payload["collection_name"],
            "processing_mode": payload["processing_mode"],
            "llm_model_name": payload["llm_model_name"],
            "embedding_model_name": payload["embedding_model_name"],
            "ocr_model_name": payload["ocr_model_name"],
        }

        # Send the request with query parameters and form data separately
        response = await client.post(
            "/index-file/",
            # params=query_params,
            data=form_data,
            files=files_data,
            headers={"Authorization": f"Bearer {token_value}"}
        )

        return response


async def call_chat_api(endpoint_url_suffix, payload):
    async with httpx.AsyncClient(timeout=300.0, base_url=FASTAPI_BASE_URL) as client:
        # Browser should automatically send the HttpOnly session_token cookie
        token_value = token()
        api_response = await client.post(endpoint_url_suffix, json=payload,
                                         headers={"Authorization": f"Bearer {token_value}"})
        return api_response


async def call_logout_api():
    async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL) as client:
        # Browser should automatically send the HttpOnly session_token cookie
        response = await client.post("/logout")
        return response


async def fetch_chroma_collections_api():
    async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL) as client:
        token_value = token()
        response = await client.get("/chroma-collections", headers={"Authorization": f"Bearer {token_value}"})
        response.raise_for_status()
        return response.json().get("collections", [])


# --- Authentication Check ---
def attempt_auth_verification():
    """Verifies authentication by calling /users/me."""
    print("attempt_auth_verification: Starting")  # Debug log
    print("Fetching user data...")  # Debug log
    # token_value = asyncio.run(token())
    token_value = token()
    time.sleep(0.5)  # Add a 0.5 second delay (adjust as needed)
    if not st.session_state.get("authenticated", False):
        print("attempt_auth_verification: Not currently authenticated, calling /users/me")
        user_data = asyncio.run(fetch_user_me_api(token_value))
        if user_data:
            st.session_state.authenticated = True
            st.session_state.user_info = user_data
            print(
                f"attempt_auth_verification: Auth successful: {user_data.get('email')}")
            st.rerun()  # Force a rerun to update the UI immediately
        else:
            st.session_state.authenticated = False
            st.session_state.user_info = None
            print("attempt_auth_verification: Auth failed")
    else:
        # Debug log
        print("attempt_auth_verification: Already authenticated, skipping check")


# Call this once when the app loads if user is not marked authenticated
# This helps catch sessions where cookie exists but Streamlit state was lost (e.g. server restart)
# --- Initial Auth Check ---
if "initial_auth_check_done" not in st.session_state:
    print("Initial auth check: Starting")  # Debug log
    attempt_auth_verification()
    st.session_state.initial_auth_check_done = True
    print("Initial auth check: Done")  # Debug log


# --- Authentication Flow / UI ---
if not st.session_state.get('authenticated', False):
    st.title("Welcome - Please Log In")
    login_url = f"{FASTAPI_BASE_URL}/login"
    st.link_button("Login with Google", login_url,
                   use_container_width=True)

    query_params = st.query_params
    print(f"Current query params: {query_params}")  # Debug log
    if "login_attempt" in query_params:
        st.info("Login attempt detected, verifying session...")
        st.experimental_set_query_params()  # Clear params
        attempt_auth_verification()
        # Add an extra rerun here, just to be sure
        st.rerun()  # Force another rerun after verification
else:  # Authenticated
    # --- Main Application UI ---
    if not st.session_state.active_chat_id and not st.session_state.chats:
        first_chat_id = str(uuid.uuid4())
        st.session_state.chats[first_chat_id] = {
            "name": "Chat 1", "history": [], "collection_name": None, "doc_name": None
        }
        st.session_state.active_chat_id = first_chat_id

    with st.sidebar:
        user_name_display = "User"
        if st.session_state.user_info:
            user_name_display = st.session_state.user_info.get(
                "name", st.session_state.user_info.get("email", "User"))
            if st.session_state.user_info.get("picture"):
                st.image(st.session_state.user_info.get("picture"), width=70)
        st.markdown(f"### Welcome, {user_name_display}!")
        st.divider()

        if st.button("➕ New Chat", use_container_width=True, key="new_chat_main_btn"):
            chat_id = str(uuid.uuid4())
            st.session_state.chats[chat_id] = {
                "name": f"Chat {len(st.session_state.chats) + 1}",
                "history": [], "collection_name": None, "doc_name": None
            }
            st.session_state.active_chat_id = chat_id
            st.rerun()

        st.markdown("##### Chat Sessions")
        if st.session_state.chats:
            chat_options = {chat_id: data["name"]
                            for chat_id, data in st.session_state.chats.items()}
            if st.session_state.active_chat_id not in chat_options and chat_options:
                st.session_state.active_chat_id = list(chat_options.keys())[0]

            current_chat_index = 0
            # Check if chat_options is not empty
            if st.session_state.active_chat_id and list(chat_options.keys()):
                try:
                    current_chat_index = list(chat_options.keys()).index(
                        st.session_state.active_chat_id)
                except ValueError:
                    current_chat_index = 0

            selected_chat_id_radio = st.radio(
                "Select Chat:", options=list(chat_options.keys()),
                format_func=lambda x: chat_options[x], index=current_chat_index,
                key="select_chat_radio", label_visibility="collapsed"
            )
            if selected_chat_id_radio != st.session_state.active_chat_id:
                st.session_state.active_chat_id = selected_chat_id_radio
                st.rerun()
        else:
            st.caption("No chats yet. Click 'New Chat' to start.")
        st.divider()

        with st.expander("⚙️ Model Settings", expanded=False):
            if "ollama_models_available" not in st.session_state:
                st.session_state.ollama_models_available = []

            if st.button("Refresh Available Models"):
                # Add this line
                print("Button 'Refresh Available Models' clicked!")
                try:
                    st.session_state.ollama_models_available = asyncio.run(
                        fetch_ollama_models_api())
                    if not st.session_state.ollama_models_available:
                        st.session_state.ollama_models_available = [
                            DEFAULT_LLM_MODEL_ST, DEFAULT_EMBEDDING_MODEL_ST, DEFAULT_OCR_MODEL_ST]  # Fallback
                    st.success("Model list updated.")
                except Exception as e:
                    st.warning(f"Could not fetch Ollama models: {e}")
                    st.session_state.ollama_models_available = [
                        st.session_state.selected_llm_model,
                        st.session_state.selected_embedding_model,
                        st.session_state.selected_ocr_model]  # Fallback to current

            # if not st.session_state.ollama_models_available:  # Initial load or if refresh failed
            #     st.session_state.ollama_models_available = [
            #         st.session_state.selected_llm_model, st.session_state.selected_embedding_model, st.session_state.selected_ocr_model, "other-model1-demo", "other-model2-demo"]

            def get_model_index(model_list, selected_model_session_key):
                model_name = st.session_state[selected_model_session_key]
                try:
                    return model_list.index(model_name)
                except ValueError:
                    if model_list:  # if model_list is not empty
                        # default to first
                        st.session_state[selected_model_session_key] = model_list[0]
                        return 0
                    return 0

            current_models = st.session_state.ollama_models_available
            selected_llm = st.selectbox("LLM Model:", options=current_models, index=get_model_index(
                current_models, "selected_llm_model"), key="sb_llm")
            selected_embed = st.selectbox("Embedding Model:", options=current_models, index=get_model_index(
                current_models, "selected_embedding_model"), key="sb_embed")
            selected_ocr = st.selectbox("OCR/Vision Model:", options=current_models,
                                        index=get_model_index(current_models, "selected_ocr_model"), key="sb_ocr")

            if st.button("Apply Model Settings", use_container_width=True, key="apply_models_btn"):
                st.session_state.selected_llm_model = selected_llm
                st.session_state.selected_embedding_model = selected_embed
                st.session_state.selected_ocr_model = selected_ocr
                st.success(
                    f"Model settings updated. They will be used for new API calls.{st.session_state.selected_embedding_model}")
                print(st.session_state.selected_embedding_model)
                # No rerun needed unless you want to force something, settings are used on next API call.
        st.divider()

        if st.session_state.active_chat_id and st.session_state.chats:
            with st.expander("📄 Document Management", expanded=True):
                active_chat_data = st.session_state.chats[st.session_state.active_chat_id]

                if "chroma_collections_available" not in st.session_state:
                    st.session_state.chroma_collections_available = []

                if st.button("Refresh Collections List"):
                    try:
                        st.session_state.chroma_collections_available = asyncio.run(
                            fetch_chroma_collections_api())
                        st.success("Collections list updated.")
                    except Exception as e:
                        st.warning(f"Could not fetch collections: {e}")

                existing_collections = st.session_state.chroma_collections_available
                # Add current if not in list
                if not existing_collections and active_chat_data.get("collection_name"):
                    existing_collections = [
                        active_chat_data["collection_name"]]
                elif active_chat_data.get("collection_name") and active_chat_data["collection_name"] not in existing_collections:
                    existing_collections.append(
                        active_chat_data["collection_name"])

                if not existing_collections:
                    st.info(
                        "No collections found via API. Refresh or process a document.")
                else:
                    options = [""] + [str(col) for col in existing_collections]
                    sel_col_load_key = f"sel_col_load_{st.session_state.active_chat_id}"
                    idx = options.index(active_chat_data["collection_name"]) if active_chat_data.get(
                        "collection_name") in options else 0
                    selected_collection_load = st.selectbox(
                        "Collection:", options=options, index=idx, key=sel_col_load_key)

                    if st.button("Load Collection", use_container_width=True, key=f"load_btn_{st.session_state.active_chat_id}"):
                        if selected_collection_load:
                            active_chat_data["collection_name"] = selected_collection_load
                            active_chat_data["doc_name"] = f"Collection: {selected_collection_load}"
                            st.success(
                                f"Switched to collection: **{selected_collection_load}**")
                            st.rerun()
                        else:
                            st.warning("Please select a collection to load.")

                if active_chat_data.get("collection_name"):
                    if st.button("Switch to General Chat Mode", use_container_width=True, key=f"sgc_btn_{st.session_state.active_chat_id}"):
                        active_chat_data["collection_name"] = None
                        active_chat_data["doc_name"] = None
                        st.success("Switched to General Chat mode.")
                        st.rerun()
                allowed_types = ["pdf", "txt", "png",
                                 "jpg", "jpeg", "bmp", "gif", "webp"]
                st.caption("Process New Document(s)")
                uploaded_files = st.file_uploader(
                    "Upload files:", type=allowed_types,
                    key=f"upload_{st.session_state.active_chat_id}", accept_multiple_files=True)
                if uploaded_files:
                    sugg_name = os.path.splitext(uploaded_files[0].name)[0].replace(
                        " ", "_") + ("_batch" if len(uploaded_files) > 1 else "")
                    collection_name_input = st.text_input(
                        "New Collection Name:", value=sugg_name, key=f"coll_name_{st.session_state.active_chat_id}")

                    processing_mode_for_pdfs = "text"
                    if any((f.name.lower().endswith(".pdf") or f.name.lower().endswith(".png") or f.name.lower().endswith(".jpg") or f.name.lower().endswith(".jpeg")) for f in uploaded_files):
                        pdf_mode_sel = st.radio("PDF Mode:", ("Text-Only", "OCR"), index=0,
                                                key=f"pdf_mode_{st.session_state.active_chat_id}",
                                                horizontal=True)
                        processing_mode_for_pdfs = "ocr" if "OCR" in pdf_mode_sel else "text"

                    if st.button("Process Uploaded File(s)", use_container_width=True, key=f"proc_btn_{st.session_state.active_chat_id}"):
                        if not collection_name_input:
                            st.error("Collection name is required.")
                        elif not uploaded_files:
                            st.error("No files uploaded.")
                        else:
                            with st.spinner(f"Processing {len(uploaded_files)} file(s) via API..."):
                                all_success = True
                                processed_count = 0
                                for up_file in uploaded_files:
                                    file_ext = os.path.splitext(
                                        up_file.name)[1].lower()
                                    current_file_mode = processing_mode_for_pdfs if file_ext in [
                                        ".pdf", ".png", ".jpg", ".jpeg"] else "text"

                                    payload = {
                                        "collection_name": collection_name_input,
                                        "processing_mode": current_file_mode,
                                        "llm_model_name": st.session_state.selected_llm_model,
                                        "embedding_model_name": st.session_state.selected_embedding_model,
                                        "ocr_model_name": st.session_state.selected_ocr_model,
                                    }
                                    files_data = {
                                        'file': (up_file.name, up_file.getvalue(), up_file.type)}

                                    try:
                                        # Corrected: Use asyncio.run() to call the async helper
                                        print("Payload for file upload:",
                                              payload)  # Debug log
                                        response = asyncio.run(
                                            call_index_file_api(payload, files_data))

                                        if response.status_code == 200:
                                            processed_count += 1
                                            st.info(
                                                f"API Response for {up_file.name}: {response.json().get('message')}")
                                        else:
                                            all_success = False
                                            st.error(
                                                f"Failed API call for {up_file.name}: {response.status_code} - {response.text}")
                                            break
                                    except Exception as e_api:
                                        all_success = False
                                        st.error(
                                            f"API request failed for {up_file.name}: {e_api}")
                                        traceback.print_exc()
                                        break

                                if all_success and processed_count > 0:
                                    active_chat_data["collection_name"] = collection_name_input
                                    active_chat_data[
                                        "doc_name"] = f"Coll: {collection_name_input} ({processed_count} files)"
                                    st.success(
                                        f"Successfully processed {processed_count} file(s) to '{collection_name_input}'.")
                                    if collection_name_input not in st.session_state.chroma_collections_available:
                                        st.session_state.chroma_collections_available.append(
                                            collection_name_input)
                                    st.rerun()
                                elif processed_count > 0:  # Partial success
                                    # Still set collection
                                    active_chat_data["collection_name"] = collection_name_input
                                    active_chat_data[
                                        "doc_name"] = f"Coll: {collection_name_input} (Partial: {processed_count} files)"
                                    st.warning(
                                        f"Partially processed files to '{collection_name_input}'.")
                                    if collection_name_input not in st.session_state.chroma_collections_available:
                                        st.session_state.chroma_collections_available.append(
                                            collection_name_input)
                                    st.rerun()
                                else:
                                    st.error(
                                        "No files were processed successfully by the API.")
        st.divider()
        if st.sidebar.button("Logout", key="logout_btn_api", use_container_width=True):
            try:
                response = asyncio.run(call_logout_api())
                if response.status_code == 200:
                    st.session_state.authenticated = False
                    st.session_state.user_info = None
                    # Clear other session data if needed, but keep some like model selections
                    # Be careful about what to clear to avoid losing all user preferences
                    st.success("Logged out successfully!")
                    st.rerun()
                else:
                    st.error(f"Logout failed on server: {response.text}")
            except Exception as e:
                st.error(f"Logout request failed: {e}")

    # --- Main Chat Area ---
    if st.session_state.active_chat_id and st.session_state.chats:
        active_chat_data = st.session_state.chats[st.session_state.active_chat_id]
        st.header(f"💬 Chat: {active_chat_data['name']}")

        if active_chat_data.get("collection_name"):
            st.caption(
                f"Mode: Chatting with **{active_chat_data['doc_name']}** | LLM: `{st.session_state.selected_llm_model}`")
        else:
            st.caption(
                f"Mode: **General Chat** | LLM: `{st.session_state.selected_llm_model}`")

        for message in active_chat_data["history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_prompt := st.chat_input("Your message...", key=f"prompt_{st.session_state.active_chat_id}"):
            active_chat_data["history"].append(
                {"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.spinner("Thinking via API..."):
                response_text = ""
                try:
                    payload = {
                        "question": user_prompt,
                        "llm_model_name": st.session_state.selected_llm_model,
                    }
                    endpoint_suffix = ""
                    if active_chat_data.get("collection_name"):
                        payload["collection_name"] = active_chat_data["collection_name"]
                        payload["embedding_model_name"] = st.session_state.selected_embedding_model
                        endpoint_suffix = "/get-rag-answer/"
                    else:
                        # Exclude current user prompt
                        payload["chat_history"] = active_chat_data["history"][:-1]
                        endpoint_suffix = "/get-general-answer/"

                    api_response = asyncio.run(
                        call_chat_api(endpoint_suffix, payload))

                    if api_response.status_code == 200:
                        response_text = api_response.json().get("answer", "Error: No answer from API.")
                    else:
                        response_text = f"Error from API: {api_response.status_code} - {api_response.text}"
                except Exception as e_api_chat:
                    response_text = f"API request failed: {e_api_chat}"
                    traceback.print_exc()

                active_chat_data["history"].append(
                    {"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)

        if st.button("Clear This Chat's History", key=f"clear_hist_btn_{st.session_state.active_chat_id}"):
            active_chat_data["history"] = []
            st.rerun()
    # Authenticated but no active chat
    elif st.session_state.get('authenticated', False):
        st.info(
            "Create a new chat or select an existing one from the sidebar to begin.")
