# auth.py
import streamlit as st
import hashlib  # For a slightly more secure password check example

# --- !!! IMPORTANT SECURITY NOTE !!! ---
# Hardcoding credentials like this is NOT secure for production.
# In a real application, use a database with hashed passwords and proper user management.
# This is for demonstration purposes only.
# Consider using libraries like streamlit-authenticator for more robust solutions.

# Example: Store hashed passwords. To generate, run something like:
# import hashlib
# print(hashlib.sha256("testpassword".encode()).hexdigest())
# test_user: 0a041b9462caa4a31bac3567e0b6e6fd9100787db2ab433d96f6d178cabfce90

# This would ideally come from a config file or environment variables in a real app
USER_CREDENTIALS = {
}


def hash_password(password: str) -> str:
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(username, provided_password):
    """Verifies a provided password against the stored hashed password."""
    if username in USER_CREDENTIALS:
        hashed_provided_password = hash_password(provided_password)
        print(hashed_provided_password)
        return hashed_provided_password == USER_CREDENTIALS[username]
    return False


def login_form():
    """Displays a login form and handles login logic."""
    st.title("Welcome - Please Log In")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input(
            "Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if verify_password(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success(f"Welcome, {username}!")
                st.rerun()  # Rerun to hide form and show app
            else:
                st.error("Invalid username or password.")
                st.session_state['authenticated'] = False
    st.info("Demo credentials: testuser / testpassword")


def logout_button():
    """Displays a logout button and handles logout logic."""
    if st.sidebar.button("Logout", key="logout_btn"):
        for key in list(st.session_state.keys()):
            # Persist model selections?
            if key not in ['selected_llm_model', 'selected_embedding_model', 'selected_ocr_model']:
                del st.session_state[key]
        st.session_state['authenticated'] = False
        st.rerun()


