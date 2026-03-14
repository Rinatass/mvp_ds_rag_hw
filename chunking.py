import fitz


def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def _recursive_split(text: str, separators: list[str], max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            result = []
            for part in parts:
                piece = part + sep
                if len(piece) > max_len and sep != separators[-1]:
                    result.extend(
                        _recursive_split(piece, separators[separators.index(sep) + 1 :], max_len)
                    )
                else:
                    result.append(piece)
            return result

    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    chunk_id = 0

    for page_data in pages:
        page = page_data["page"]
        raw_parts = _recursive_split(page_data["text"], separators, chunk_size)

        buffer = ""
        for part in raw_parts:
            if len(buffer) + len(part) <= chunk_size:
                buffer += part
            else:
                if buffer.strip():
                    chunks.append({"text": buffer.strip(), "page": page, "chunk_id": chunk_id})
                    chunk_id += 1
                overlap_text = buffer[-overlap:] if overlap else ""
                buffer = overlap_text + part

        if buffer.strip():
            chunks.append({"text": buffer.strip(), "page": page, "chunk_id": chunk_id})
            chunk_id += 1

    return chunks


def chunk_pdf(pdf_path: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, chunk_size, overlap)