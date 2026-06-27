import PyPDF2
import os

# Use raw string (r"...") to handle spaces in filename
pdf_path = r"D:\pakistan_constitution_llm\consitution rights.pdf"
output_path = r"D:\pakistan_constitution_llm\data\raw\constitution.txt"

print(f"Reading: {pdf_path}")
print(f"File exists? {os.path.exists(pdf_path)}")

if not os.path.exists(pdf_path):
    print("ERROR: PDF not found!")
    print("Please check the filename and path.")
    exit(1)

with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for i, page in enumerate(reader.pages):
        print(f"  Page {i+1}/{len(reader.pages)}")
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\n✓ Saved to: {output_path}")
print(f"Total characters: {len(text):,}")