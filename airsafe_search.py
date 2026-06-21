import os
import math
import re
from collections import defaultdict, Counter

# -------- CONFIG --------
DOCS_DIR = "docs"      # folder where your accident reports are stored
TOP_K = 5              # how many results to show


# -------- TEXT PREPROCESSING --------
STOPWORDS = {
    "the", "is", "am", "are", "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "with", "this", "that", "by", "at", "as", "from", "be", "was",
    "were", "it", "its", "into", "during", "after", "before"
}


def tokenize(text: str):
    """Lowercase, keep only alphanumeric tokens, remove stopwords."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS]


# -------- LOAD DOCUMENTS --------
def load_documents():
    """Load all .txt files from DOCS_DIR into a dict: id -> {name, text}."""
    documents = {}
    doc_id = 0

    if not os.path.exists(DOCS_DIR):
        print(f"Docs folder '{DOCS_DIR}' does not exist.")
        return documents

    for fname in os.listdir(DOCS_DIR):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(DOCS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            documents[doc_id] = {
                "id": doc_id,
                "name": fname,
                "text": text
            }
            doc_id += 1
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    return documents


# -------- INDEX CONSTRUCTION --------
def build_index(documents):
    """
    Build inverted index, df, idf and document vector lengths.
    Returns:
        inverted_index: term -> list of (doc_id, tf)
        idf: term -> idf value
        doc_lengths: doc_id -> vector length (for cosine)
    """
    inverted_index = defaultdict(list)
    N = len(documents)

    # 1. term frequencies for each doc
    for doc_id, doc in documents.items():
        tokens = tokenize(doc["text"])
        counts = Counter(tokens)
        for term, tf in counts.items():
            inverted_index[term].append((doc_id, tf))

    # 2. document frequency and idf
    df = {term: len(postings) for term, postings in inverted_index.items()}
    idf = {term: math.log((N + 1) / (df_t + 1)) + 1 for term, df_t in df.items()}

    # 3. document vector lengths
    doc_lengths = {}
    for doc_id, doc in documents.items():
        tokens = tokenize(doc["text"])
        counts = Counter(tokens)
        length_sq = 0.0
        for term, tf in counts.items():
            if term in idf:
                weight = tf * idf[term]
                length_sq += weight * weight
        doc_lengths[doc_id] = math.sqrt(length_sq)

    return inverted_index, idf, doc_lengths


# -------- SNIPPET GENERATOR --------
def make_snippet(text, query_tokens, max_len=250):
    """
    Very simple snippet generator:
    - looks for first occurrence of any query term
    - returns some context around it
    """
    lowered = text.lower()
    pos = -1

    for term in query_tokens:
        p = lowered.find(term)
        if p != -1 and (pos == -1 or p < pos):
            pos = p

    if pos == -1:
        # fallback: just first max_len chars
        return text[:max_len].replace("\n", " ")

    start = max(0, pos - 60)
    end = min(len(text), pos + 60)
    snippet = text[start:end].replace("\n", " ")
    return f"...{snippet}..."


# -------- SEARCH FUNCTION --------
def search(query, documents, inverted_index, idf, doc_lengths, k=TOP_K):
    """Return top-k (doc_name, score, snippet) for the given query."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    q_counts = Counter(query_tokens)

    # Build query vector
    q_vec = {}
    for term, tf in q_counts.items():
        if term in idf:
            q_vec[term] = tf * idf[term]

    # Accumulate scores
    scores = defaultdict(float)
    for term, q_wt in q_vec.items():
        for doc_id, tf in inverted_index.get(term, []):
            d_wt = tf * idf[term]
            scores[doc_id] += q_wt * d_wt

    # Cosine normalization
    for doc_id in list(scores.keys()):
        if doc_lengths[doc_id] > 0:
            scores[doc_id] /= doc_lengths[doc_id]

    # Sort and take top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for doc_id, score in ranked:
        snippet = make_snippet(documents[doc_id]["text"], query_tokens)
        results.append((documents[doc_id]["name"], score, snippet))

    return results


# -------- MAIN CLI LOOP --------
def main():
    print("Loading documents from folder:", DOCS_DIR)
    documents = load_documents()
    if not documents:
        print("No documents found in 'docs' folder. Please add .txt files.")
        return

    print(f"Loaded {len(documents)} documents.")
    print("Building index...")
    inverted_index, idf, doc_lengths = build_index(documents)
    print("Index built. Ready to search.\n")

    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Exiting AirSafeSearch Engine.")
            break

        results = search(query, documents, inverted_index, idf, doc_lengths)
        if not results:
            print("No matching accident reports found.\n")
            continue

        print(f"\nTop {len(results)} results:")
        for i, (name, score, snippet) in enumerate(results, start=1):
            print(f"\n{i}. {name}  (score = {score:.4f})")
            print(f"   Snippet: {snippet}")
        print()


if __name__ == "__main__":
    main()
