from ollama_ocr import OCRProcessor

# model_name = 'llama3.2-vision'
model_name = 'gemma3:12b'
# Initialize OCR processor
# You can use any vision model available on Ollama
ocr = OCRProcessor(model_name,
                   base_url="http://localhost:11434/api/generate")

# Process an image
result = ocr.process_image(
    # path to your pdf files "path/to/your/file.pdf"
    image_path=path,
    format_type="markdown",  # Options: markdown, text, json, structured, key_value
    # Optional custom prompt
    custom_prompt="Extract all data without any modification",
    language="English"  # Specify the language of the text (New! 🆕)
)
print(result)
