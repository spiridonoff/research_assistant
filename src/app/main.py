import streamlit as st
from rag.index.search_abs import search_abstracts
from rag.index.build_index_paper import build_chunk_index
from rag.index.search_paper import search_fulltext
from rag.io.fetch_abs import fetch_papers
from rag.io.fetch_paper import lookup_paper_by_id, download_papers
from rag.pipelines.summarizer import summarize_papers
import config

st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("📄 AI Research Assistant")
st.write("Search papers, retrieve full text, rank chunks, and summarize with an LLM")

# Initialize session state
if "topic_submitted" not in st.session_state:
    st.session_state.topic_submitted = False

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if not st.session_state.topic_submitted:
    topic = st.text_input("Enter a topic to search papers:", key="topic_input")

    if st.button("Search Papers"):
        if topic.strip() == "":
            st.warning("Please enter a topic first.")
        else:
            # Call your fetch_papers pipeline
            papers = fetch_papers(topic)
            st.session_state.search_results = papers
            st.session_state.topic_submitted = True

            st.success(f"Found {len(papers)} papers for topic: {topic}")

            # Preview the first few results
            for i, paper in enumerate(papers[:5]):
                st.markdown(f"**{i+1}. {paper['title']}**")
                st.markdown(f"*Authors:* {', '.join(paper.get('authors', []))}")
                st.markdown(f"*Abstract:* {paper.get('abstract', '')[:300]}...")  # truncated
                st.markdown("---")

if st.session_state.topic_submitted:
    query = st.text_area("Enter your research question:", height=150)

if st.button("Search (Abstracts Only)"):
    with st.spinner("Searching abstracts..."):
        abs_results = search_abstracts(
            query, top_k_raw=config.TOP_K_RAW, top_k_final=config.TOP_K_FINAL
        )

    st.subheader("Top Papers (Abstract-level)")
    for r in abs_results:
        paper_id = r["paperId"]
        paper = lookup_paper_by_id(paper_id)
        st.markdown(
            f"""**Title:** {paper.get("title", "Unknown Title")} —  Score: `{r["score"]:.4f}`
                    \n **Paper ID:** {r["paperId"]} 
        """
        )
        # st.markdown(f" \n ")
        st.markdown(f"> {paper.get('abstract', 'No abstract available.')[:1000]}...")

    st.session_state["abs_results"] = abs_results

# Only show the second button if abstracts already retrieved
if "abs_results" in st.session_state:
    if st.button("Run Full-Text Retrieval"):
        with st.spinner("Downloading PDFs, chunking and ranking..."):
            top_paper_ids = [
                p["paperId"] for p in st.session_state["abs_results"]
            ]  # Test with first 5 papers
            top_papers = [lookup_paper_by_id(pid) for pid in top_paper_ids]

            downloaded_papers = download_papers(top_papers, save_dir=config.PDF_DIR)
        with st.spinner("Scanning and chunking texts..."):
            build_chunk_index(downloaded_papers)
        with st.spinner("ranking chunks..."):
            search_results = search_fulltext(
                query, top_k_raw=config.TOP_K_RAW, top_k_final=config.TOP_K_FINAL
            )

        st.session_state["search_results"] = search_results
        st.subheader("Top Relevant Chunks (Full-Text)")
        for r in search_results:
            paper_id = r["paperId"]
            paper = lookup_paper_by_id(paper_id)
            st.markdown(
                f"""**Title:** {paper.get("title", "Unknown Title")} —  Score: `{r["score"]:.4f}`
                    \n **Paper ID:** {r["paperId"]} 
                    """
            )
            st.markdown(f"> {' ... '.join(r['chunk_texts'])}...")

# Summarize
if "search_results" in st.session_state:
    search_results = st.session_state["search_results"]
    for result in search_results:
        paper_id = result["paperId"]
        paper = lookup_paper_by_id(paper_id)
        result["title"] = paper.get("title", "Unknown Title")
    if st.button("Summarize with LLM"):
        with st.spinner("Generating summary..."):
            summary = summarize_papers(search_results, query)

        st.subheader("🧠 Final Summary")
        st.write(summary)
        st.button(
            "Copy to Clipboard",
            on_click=lambda: st.session_state.update(
        {"_copy_js": f"navigator.clipboard.writeText(`{summary}`)"}
    ),
        )
