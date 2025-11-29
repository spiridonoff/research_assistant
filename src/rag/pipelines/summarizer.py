from openai import OpenAI
import config as config

client = OpenAI()  # Your key must be in OPENAI_API_KEY env variable


def summarize_papers(results, query):
    paper_blocks = []
    for entry in results:
        title = entry.get("title", "Unknown Title")
        # paper_id = entry.get("paperId", "Unknown")
        score = entry.get("score", 0.0)
        chunks = entry.get("chunk_texts", [])

        block_lines = [
            f"paper title: {title}",
            # f"Paper ID: {paper_id}",
            f"Relevance score: {score:.4f}\n",
        ]

        for i, chunk in enumerate(chunks, 1):
            block_lines.append(f"Chunk {i}:\n{chunk}\n")

        paper_blocks.append("\n".join(block_lines))

    paper_blocks_text = "\n---\n\n".join(paper_blocks)

    prompt = f"""
    You are an expert scientific research assistant.

    Your task is to read the retrieved passages from multiple research papers and
    produce a concise, accurate, and well-organized summary that answers the user’s query.

    User query:
    "{query}"

    You are given the most relevant passages from several papers. 
    These passages may be incomplete, may overlap, or may not appear in the original order.
    Do NOT invent information. Base all statements strictly on the given text.

    For each paper:
    - First, infer the paper’s topic based only on the provided passages.
    - Then summarize the key findings relevant to the query.
    - If the passages do not provide enough information to infer a finding, say so.

    After analyzing all papers, provide:
    1. **Cross-paper synthesis**: What themes, methods, or results appear across papers?
    2. **Contradictions or differences** between papers.
    3. **Answer to the original query** using only supported claims.
    4. **Open questions or limitations**, if detectable from the passages.

    When referring to specific evidence, cite as: (paper_id, chunk #).  
    Example: “The model significantly improved accuracy (P12345, chunk 2).”

    Here are the retrieved passages:
    {paper_blocks_text}
"""
    response = client.responses.create(model=config.OPENAI_MODEL_NAME, input=prompt)

    return response.output_text