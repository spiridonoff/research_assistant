from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import re
import config as config


def extract_text_from_pdf(pdf_path):
    """
    Extract clean text from a research PDF using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")

    pages = []

    for page in doc:
        text = page.get_text()
        pages.append(text)

    raw_text = "\n".join(pages)
    cleaned = clean_text(raw_text)

    return cleaned


def clean_text(text):
    """
    Basic cleanup for line breaks, multiple spaces, and hyphenation.
    """

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove hyphenated line-breaks ("transformer-\nbased" → "transformer based")
    text = re.sub(r"-\s+", "", text)

    # Fix weird unicode dashes
    text = text.replace("–", "-").replace("—", "-")

    # Remove multiple spaces again after fixes
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text, max_tokens=300, overlap=50, tokenize=False):
    """
    Chunk text into overlapping segments for embedding & retrieval.

    max_tokens: size of each chunk (~ a few sentences)
    overlap: how many tokens overlap between chunks
    """
    if tokenize:
        # Load embedding model
        model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        tokenizer = model.tokenizer
        tokens = tokenizer.encode(text)
    else:
        # Basic tokenization by splitting on whitespace
        tokens = text.split()

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]

        if tokenize:
            chunk_text = tokenizer.decode(chunk_tokens)
        else:
            chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)

        # move window forward w/ overlap
        start = end - overlap
        if start < 0:
            break

    return chunks


# if __name__ == "__main__":
#     # Quick test:
#     test_pdf = "pdfs/16ea8af3dd97d9fb29f8bb2efd132bb590d63df4.pdf"
#     text = extract_text_from_pdf(test_pdf)
#     print(text[:1500])  # show first part
