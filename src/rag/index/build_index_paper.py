from sentence_transformers import SentenceTransformer
from rag.io.text_utils import chunk_text, extract_text_from_pdf
import numpy as np
import json
import os
import faiss
import config

def chunk_papers(papers: list, pdf_dir=config.PDF_DIR, **kwargs):
    """Chunk full texts from papers into overlapping segments."""
    paper_chunks = []
    for paper in papers:
        paper_id = paper["paperId"]

        pdf_name = f"{paper_id}.pdf"
        pdf_path = os.path.join(pdf_dir, pdf_name)
        print(f"\nProcessing PDF: {pdf_path}")

        # Extract full text
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 500:
            print("Warning: PDF text too short, skipping.")
            continue

        # Chunk text
        chunks = chunk_text(text, **kwargs)

        # Store metadata
        for i, chunk in enumerate(chunks):
            paper_chunks.append(
                {
                    "paperId": paper_id,
                    "chunk_id": i,
                    "text": chunk,
                }
            )

        print(f"Added {len(chunks)} chunks.\n")
    return paper_chunks


def build_chunk_index(papers: list, **kwargs):
    paper_chunks = chunk_papers(papers, config.PDF_DIR, **kwargs)
    with open("chunks_full.json", "w") as f:
        json.dump(paper_chunks, f, indent=2)

    # embed all chunks
    chunk_texts = [c["text"] for c in paper_chunks]
    model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    all_embeddings = model.encode(
        chunk_texts, convert_to_numpy=True, normalize_embeddings=True
    )
    all_embeddings = np.vstack(all_embeddings)
    print("Paper chunks embeddings shape:", all_embeddings.shape)

    # Build FAISS index
    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)

    print("FAISS paper index size:", index.ntotal)

    # Save index
    index_path = os.path.join(config.INDEX_DIR, "papers.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")


# if __name__ == "__main__":
#     test
#     with open(config.PAPER_FILE) as f:
#         papers = json.load(f)
#     build_chunk_index(papers[:5])
