import os
import json
import config


def lookup_paper_by_id(paper_id: str) -> dict:
    """
    Lookup a paper by its paperId in a list of papers.
    """
    papers = json.load(open(config.PAPER_FILE))
    for paper in papers:
        if paper["paperId"] == paper_id:
            return paper
    return None


def download_pdf(url: str, save_path: str) -> bool:
    """
    Download a PDF from a given URL and save it to the specified path.
    Returns True if successful, False otherwise.
    """
    import requests

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")
        return False


def download_papers(papers: list, save_dir) -> None:
    """
    Given a list of papers, download their PDFs if available.
    """
    os.makedirs(save_dir, exist_ok=True)
    downloaded_papers = []
    for paper in papers:
        pdf_url = paper.get("pdf_url")
        if not pdf_url or len(pdf_url) == 0:
            print(f"No PDF URL for paper {paper['paperId']} , trying arXiv...")
            # TODO: fall-back to arXiv fetch if arXiv ID is available
            if paper.get("arxiv_id"):
                print(f"Paper has arXiv ID {paper['arxiv_id']}, fetching from arXiv.")
                # TODO: implement arXiv fetch
                arxiv_id = paper["arxiv_id"]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            else:
                print(
                    f"No arXiv ID available for paper {paper['paperId']}, skipping.\n"
                )
                continue

        save_path = os.path.join(save_dir, f"{paper['paperId']}.pdf")
        if os.path.exists(save_path):
            print(
                f"PDF for paper {paper['paperId']} already exists, skipping download.\n"
            )
            downloaded_papers.append(paper)
            continue

        print(f"Downloading PDF for paper {paper['paperId']}...")
        success = download_pdf(pdf_url, save_path)
        if success:
            print(f"Saved PDF to {save_path}\n")
            downloaded_papers.append(paper)
        else:
            print(f"Failed to download PDF for paper {paper['paperId']}\n")
    return downloaded_papers


# if __name__ == "__main__":
#  test
# with open(config.PAPER_FILE) as f:
#     papers = json.load(f)

# paper_ids_to_download = [p["paperId"] for p in papers[:5]]  # Test with first 5 papers
# papers_to_download = [lookup_paper_by_id(pid) for pid in paper_ids_to_download]

# download_papers(papers_to_download, save_dir=config.PDF_DIR)
