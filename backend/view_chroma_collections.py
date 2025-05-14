import streamlit as st
import chromadb

from chromadb import PersistentClient
# Set Chroma directory (change if needed)
CHROMA_STORE_PATH = './chroma_store'
client = chromadb.PersistentClient(path=CHROMA_STORE_PATH)


# Get list of collections
collections = client.list_collections()
print(f"Collections: {collections}")
# st.title(collections)

if not collections:
    st.warning("No collections found in the Chroma DB.")
else:
    collection_names = [col for col in collections]

    # Collection selector
    selected_name = st.selectbox(
        "Select a Chroma Collection", collection_names)

    # Load selected collection
    collection = client.get_collection(name=selected_name)

    # Fetch data
    limit = st.slider("How many items to display?",
                      min_value=1, max_value=100, value=10)
    results = collection.get(include=["documents", "metadatas"], limit=limit)

    st.subheader(f"Documents in '{selected_name}'")
    for i in range(len(results['ids'])):
        st.markdown(f"### ID: {results['ids'][i]}")
        st.markdown(f"**Document:** {results['documents'][i]}")
        st.markdown(f"**Metadata:** {results['metadatas'][i]}")
        st.markdown("---")
