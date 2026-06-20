import os
from pathlib import Path
from pypdf import PdfReader

DOCS_DIR = Path("./docs")

def main():
    pdf_path = DOCS_DIR / "RegsNavyIII.pdf"
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return
        
    reader = PdfReader(pdf_path)
    print(f"Reading {pdf_path.name} (Total pages: {len(reader.pages)})")
    
    # Find the page where Chapter I starts (avoiding table of contents)
    chapter_1_page_idx = -1
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        # Look for "CHAPTER I" and make sure it's not the Contents page
        if "CHAPTER I" in text and "CONTENTS" not in text and idx > 8:
            chapter_1_page_idx = idx
            break
            
    if chapter_1_page_idx == -1:
        # Fallback to page 10
        chapter_1_page_idx = 10
        
    print(f"Chapter I starts at page index: {chapter_1_page_idx + 1}")
    
    for i in range(3):
        page_num = chapter_1_page_idx + i
        if page_num < len(reader.pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            print(f"\n--- PAGE {page_num + 1} (Length: {len(text)} chars) ---")
            print(text[:1500])
            print("...")

if __name__ == "__main__":
    main()
