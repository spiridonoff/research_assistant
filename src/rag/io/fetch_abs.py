import requests
import json
import config

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_papers(query, limit=20):
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,url,authors,year,citationCount,externalIds,openAccessPdf,isOpenAccess",
    }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    papers = []
    for item in data.get("data", []):
        # skip papers without abstracts
        if not item.get("abstract"):
            continue

        papers.append(
            {
                "paperId": item["paperId"],
                "source": "s2",
                "title": item["title"],
                "abstract": item["abstract"],
                "url": item.get("url", ""),
                "authors": [a["name"] for a in item.get("authors", [])],
                "year": item.get("year", None),
                "citationCount": item.get("citationCount", 0),
                "arxiv_id": item.get("externalIds", {}).get("ArXiv", None),
                "isOpenAccess": item.get("isOpenAccess", False),
                "pdf_url": item.get("openAccessPdf", {}).get("url", None),
            }
        )
    # Save results to JSON
    with open(config.PAPER_FILE, "w") as f:
        json.dump(papers, f, indent=2)

    print(f"Saved to {config.PAPER_FILE}")

    return papers


# if __name__ == "__main__":
#     query = input("Enter a research topic to fetch papers: ")
#     papers = fetch_papers(query, limit=100)

#     print(f"Fetched {len(papers)} papers.")