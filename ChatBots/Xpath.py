import streamlit as st
from bs4 import BeautifulSoup
import requests
from lxml import etree
from io import BytesIO
import base64
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Initialize LLM
llm = OllamaLLM(model="qwen3:14b", streaming=True)

# HTML content placeholder
html_content = ""

st.set_page_config(page_title="XPath Auto Generator", layout="wide")
st.title("🔍 XPath Auto Generator with LLM & Visualizer")

input_mode = st.radio("Choose Input Method:",
                      ("Upload HTML File", "Paste Raw HTML", "Load from URL"))

if input_mode == "Upload HTML File":
    uploaded_file = st.file_uploader("Upload HTML file", type="html")
    if uploaded_file:
        html_content = uploaded_file.read().decode("utf-8")
elif input_mode == "Paste Raw HTML":
    html_content = st.text_area("Paste HTML here")
elif input_mode == "Load from URL":
    url = st.text_input("Enter URL")
    if url:
        try:
            html_content = requests.get(url).text
        except Exception as e:
            st.error(f"Failed to fetch URL: {e}")

# Process HTML if available
if html_content:
    soup = BeautifulSoup(html_content, "html.parser")
    tree = etree.HTML(str(soup))

    query = st.text_input(
        "🔎 Ask a Natural Language Question (e.g., 'Find the login button')")
    prom = """ """
    if query:
        prompt = PromptTemplate.from_template("""
You are an expert web scraper. Your task is to find the XPath of an element based on a user's request.

Only return the XPath string.

Request:
"{query}"

HTML:
```
                                              {html_content}
```
            """
                                              )
        final_prompt = prompt.format(query=query, html_content=html_content)
        xpath_result = llm.invoke(final_prompt).strip()
        st.code(xpath_result, language="text")

    if st.checkbox("Show Parsed HTML Preview"):
        st.write(soup.prettify())

    # Optional: Screenshot + overlay tool would go here using Selenium or Playwright
    st.markdown("---")
    st.subheader("📸 Visual Overlay (Prototype Coming Soon)")
    st.info("This will let you click on the visual UI and highlight XPath elements.")
else:
    st.warning("Please input HTML using one of the methods above.")
