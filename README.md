# 📚 Mini Research Assistant (RAG-based)

A **research assistant prototype** that helps you quickly find, rank, and summarize scientific papers on a given topic using **retrieval-augmented generation (RAG)** with **OpenAI GPT models**.

The system fetches paper abstracts, creates embeddings, performs similarity search (FAISS), reranks results with a cross-encoder, and summarizes relevant papers for your query. Includes a simple **Streamlit UI** for interactive use.

---

## ⚡ Features

* Search scientific papers by **topic** and preview abstracts.
* Semantic ranking using **FAISS embeddings**.
* **cross-encoder reranking** for improved relevance.
* Chunking support for both abstracts and full-text papers (PDFs).
* Summarization of top relevant papers with an **LLM** (OpenAI GPT).
* Streamlit-based **interactive UI**:

  * Enter a topic
  * Search abstracts
  * View summaries
  * Copy or download summaries

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/spiridonoff/mini-research-assistant.git
cd mini-research-assistant
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"  # Linux/Mac
setx OPENAI_API_KEY "your_api_key_here"    # Windows
```

---

## 🚀 Usage

Run the Streamlit app:

```bash
./run.sh
```

or manually:

```bash
export PYTHONPATH=src
streamlit run src/app/main.py
```

**Workflow:**

1. Enter a **topic** to fetch related papers.
2. Preview the first few abstracts.
3. Enter a **research query** to search across abstracts.
4. View **ranked results** and summaries.
5. Copy or download the summaries for further use.

---

## 📁 Project Structure

```
src/
├─ app/
│  └─ main.py        # Streamlit UI
├─ rag/
│  ├─ io/
│  |  ├─ fetch_abs.py
│  │  ├─ fetch_papers.py
│  │  └─ text_utils.py
│  ├─ index/
│  │  ├─ build_index_abs.py
│  │  ├─ build_index_paper.py
│  │  ├─ search_abs.py
│  │  └─ search_paper.py
│  ├─ pipelines/
│  │  └─ summarizer.py
├─ config.py          # API keys and configuration
run.sh                # Launcher script with PYTHONPATH
requirements.txt
```

---

## ⚡ Next Steps / Future Improvements

* Add **selection & download of specific papers**.
* Integrate **OLMo** or other open source LLMs for research summaries.
* Improve **prompt design** for better summaries.
* Extend UI for **follow-up questions** using conversational LLM.

---

## 💡 Notes

* This project is intended as a **mini prototype** / learning project.
* Designed to be **modular**: abstracts search, embedding, FAISS indexing, reranking, summarization, and UI can be extended independently.
* OpenAI API usage may incur costs depending on your queries.

---

## 📝 License

MIT License – feel free to reuse and modify.
