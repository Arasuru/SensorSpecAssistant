import fitz

# Open a PDF document
doc = fitz.open("bst-bme280-ds002.pdf")
# Print basic info
print(f"Number of pages: {doc.page_count}")
print(f"Metadata: {doc.metadata}")

import pydantic

# Simple version string (e.g., '2.12.5')
print(pydantic.__version__)

# Detailed version info (v2 only)
from pydantic.version import version_info
print(version_info())