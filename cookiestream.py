import streamlit as st
from streamlit_js_eval import streamlit_js_eval

session_token = streamlit_js_eval(
    js_expressions="document.cookie", key="get_cookie")

if session_token:
    # Extract session_token value from cookie string
    import re
    match = re.search(r'session_token=([^;]+)', session_token)
    if match:
        token = match.group(1)
        st.write("Token found:", token)
    else:
        st.warning("session_token not found in cookies")
else:
    st.info("No cookies received yet.")
