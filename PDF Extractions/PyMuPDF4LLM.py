import json
import pathlib
import pymupdf4llm

# Extract PDF content as Markdown
md_text = pymupdf4llm.to_markdown(path)
# print(md_text[:500])


# Use Case 2: Extracting Specific Pages

# Extract only pages 10 and 11
md_text = pymupdf4llm.to_markdown(path, pages=[0, 0])
# print(md_text[:500])  # Print first 500 characters

# Use Case 3: Saving Markdown to a File

md_text = pymupdf4llm.to_markdown(path)
pathlib.Path("output.md").write_bytes(md_text.encode())
print("Not working completely good only reading text not images")


# Use Case 4: Extracting Data as LlamaIndex Documents

llama_reader = pymupdf4llm.LlamaMarkdownReader()
llama_docs = llama_reader.load_data(path)
print(f"Number of LlamaIndex documents: {len(llama_docs)}")
# print(f"Content of first document: {llama_docs[0].text[:500]}")
# for i in llama_docs:
#     print(i)


# Use Case 5: Image Extraction
md_text_images = pymupdf4llm.to_markdown(doc=path,
                                         #  pages=[0, 11],
                                         page_chunks=True,
                                         write_images=True,
                                         image_path="images",
                                         image_format="jpg",
                                         dpi=100)
# Print image information from the first chunk
# print(md_text_images[0]['images'])

# Use Case 6: Chunking with Metadata
md_text_chunks = pymupdf4llm.to_markdown(doc=path,
                                         pages=[0, 1, 2],
                                         page_chunks=True)
# print(md_text_chunks[0])  # Print the first chunk

# Use Case 7: Word-by-Word Extraction
md_text_words = pymupdf4llm.to_markdown(doc=path,
                                        pages=[1, 2],
                                        page_chunks=True,
                                        write_images=True,
                                        image_path="images",
                                        image_format="jpg",
                                        dpi=200,
                                        extract_words=True)
# print(md_text_words[0]['words'])

# Use Case 8: Table Extraction

md_text_tables = pymupdf4llm.to_markdown(doc=path,
                                         # Specify pages containing tables
                                         pages=[2],
                                         )
print(md_text_tables)
# Not working great
