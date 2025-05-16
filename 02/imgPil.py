from PIL import Image
import pytesseract
import pandas as pd
from io import StringIO

# Load the image
image = Image.open('/Users/akash.kansal/Downloads/image (1).png')

# Extract text using Tesseract OCR
raw_text = pytesseract.image_to_string(image)

# Print the raw OCR output for inspection
print("Raw OCR Output:\n", raw_text)

# Optional: Save the raw OCR to a text file for manual checking
with open("ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)
