import os

import fitz  # PyMuPDF

def load_pdf(file_path):
    doc = fitz.open(file_path)

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]

        pages.append(
            {
                "source": os.path.basename(file_path),
                "page": page_num + 1,
                "text": page.get_text(),  # type: ignore
            }
        )
    return pages

def load_all_pdfs(folder_path):
    all_pages = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            pages = load_pdf(file_path)
            all_pages.extend(pages)
    return all_pages