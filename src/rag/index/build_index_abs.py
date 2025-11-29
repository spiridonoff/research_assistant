from sentence_transformers import SentenceTransformer
import numpy as np
import json
from rag.io.text_utils import chunk_text
import faiss
import os
import config as config


def chunk_abstracts(papers, **kwargs):
    """Chunk abstracts from papers into overlapping segments."""
    abs_chunks = []
    for paper in papers:
        paper_id = paper["paperId"]

        chunks = chunk_text(paper["abstract"], **kwargs)

        for i, chunk in enumerate(chunks):
            abs_chunks.append(
                {
                    "paperId": paper_id,
                    "chunk_id": i,
                    "text": chunk,
                }
            )

    return abs_chunks


def build_abstract_index(chunked=False, **kwargs):
    """Build FAISS index for abstracts or chunked abstracts."""
    # Load embedding model
    model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    # Load papers
    with open(config.PAPER_FILE) as f:
        papers = json.load(f)

    # embed abstracts as a whole
    if not chunked:
        abstracts = [p["abstract"] for p in papers]

        embeddings_abs = model.encode(
            abstracts, convert_to_numpy=True, normalize_embeddings=True
        )
        embeddings_abs = np.vstack(embeddings_abs)
        print("Abstract embeddings shape:", embeddings_abs.shape)
        # np.save("embeddings_abs.npy", embeddings_abs)

        # Create FAISS index (Inner Product sim = cosine sim after normalization)
        index_abs = faiss.IndexFlatIP(embeddings_abs.shape[1])

        # Add vectors to index
        index_abs.add(embeddings_abs)
        print("FAISS abstract index size:", index_abs.ntotal)
        # Save index to disk
        index_path = os.path.join(config.INDEX_DIR, "abs.index")
        faiss.write_index(index_abs, index_path)

    # embed chunked abstracts
    else:
        abs_chunks = chunk_abstracts(papers, **kwargs)
        with open(config.CHUNKS_ABS_FILE, "w") as f:
            json.dump(abs_chunks, f, indent=2)
        print(f"Saved Total of {len(abs_chunks)} chunks.")

        abs_chunk_texts = [c["text"] for c in abs_chunks]

        embeddings_abs_chunk = model.encode(
            abs_chunk_texts, convert_to_numpy=True, normalize_embeddings=True
        )
        embeddings_abs_chunk = np.vstack(embeddings_abs_chunk)
        print("Abstract chunks embeddings shape:", embeddings_abs_chunk.shape)
        # np.save("embeddings_abs_chunk.npy", embeddings_abs_chunk)

        # Create FAISS index (Inner Product sim = cosine sim after normalization)
        index_abs_chunk = faiss.IndexFlatIP(embeddings_abs_chunk.shape[1])

        # Add vectors to index
        index_abs_chunk.add(embeddings_abs_chunk)
        print("FAISS abstract chunk index size:", index_abs_chunk.ntotal)
        # Save index to disk
        index_path = os.path.join(config.INDEX_DIR, "abs_chunk.index")
        faiss.write_index(index_abs_chunk, index_path)

    print("Saved FAISS index to abs.index and abs_chunk.index")


if __name__ == "__main__":
    build_abstract_index(chunked=True)
