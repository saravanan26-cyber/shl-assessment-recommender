"""
Build a retrieval index over the SHL catalog.
Uses TF-IDF + cosine similarity (sklearn) — no GPU, no heavy deps.
Saves index artifacts to data/index.pkl
"""

import json
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_document(assessment: dict) -> str:
    """
    Combine all assessment fields into a single searchable text document.
    Repeat important fields to boost their weight in TF-IDF.
    """
    parts = [
        assessment.get("name", "") * 3,  # name is most important
        assessment.get("description", "") * 2,
        " ".join(assessment.get("test_type_labels", [])) * 2,
        " ".join(assessment.get("job_levels", [])),
        assessment.get("duration", ""),
        " ".join(assessment.get("languages", [])),
    ]

    # Add semantic expansions based on test types
    expansions = []
    for code in assessment.get("test_types", []):
        if code == "A":
            expansions.extend(["cognitive ability", "reasoning", "aptitude", "intelligence", "problem solving", "analytical"])
        elif code == "P":
            expansions.extend(["personality", "behaviour", "character", "traits", "culture fit", "interpersonal"])
        elif code == "K":
            expansions.extend(["knowledge", "skills", "technical", "expertise", "proficiency"])
        elif code == "M":
            expansions.extend(["motivation", "drive", "engagement", "values", "what motivates"])
        elif code == "B":
            expansions.extend(["situational judgement", "SJT", "judgment", "scenarios", "decision making"])
        elif code == "S":
            expansions.extend(["simulation", "realistic", "work sample", "hands-on"])
        elif code == "E":
            expansions.extend(["exercise", "group", "roleplay", "assessment centre"])
        elif code == "C":
            expansions.extend(["competency", "competencies", "behavioral", "structured interview"])
        elif code == "D":
            expansions.extend(["development", "360", "feedback", "leadership development"])

    parts.append(" ".join(expansions))

    # Tech-specific expansions
    name = assessment.get("name", "").lower()
    if any(lang in name for lang in ["java", "python", "javascript", "c++", "scala", "rust", "node"]):
        parts.append("developer programming coding software engineer technical")
    if "sql" in name:
        parts.append("database query data analyst backend")
    if "machine learning" in name or "data science" in name:
        parts.append("AI ML data scientist analytics statistics")
    if "aws" in name or "devops" in name:
        parts.append("cloud infrastructure platform engineer SRE")
    if any(w in name for w in ["manager", "leadership", "360"]):
        parts.append("management leadership senior executive director")
    if any(w in name for w in ["sales", "service", "customer"]):
        parts.append("client-facing commercial revenue business development")
    if "verbal" in name:
        parts.append("communication writing language comprehension")
    if "numerical" in name:
        parts.append("math quantitative finance accounting data")

    return " ".join(filter(None, parts))


def build_index(catalog_path: str = "data/catalog.json", output_path: str = "data/index.pkl"):
    print("Loading catalog...")
    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    print(f"Building index over {len(catalog)} assessments...")

    documents = [build_document(a) for a in catalog]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    index_data = {
        "catalog": catalog,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }

    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)

    print(f"✅ Index saved to {output_path}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Matrix shape: {tfidf_matrix.shape}")
    return index_data


def retrieve(query: str, index_data: dict, top_k: int = 15) -> list[dict]:
    """
    Retrieve top_k assessments most relevant to the query.
    Returns list of (assessment, score) sorted by relevance.
    """
    vectorizer = index_data["vectorizer"]
    tfidf_matrix = index_data["tfidf_matrix"]
    catalog = index_data["catalog"]

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0.001:
            results.append({
                **catalog[idx],
                "_score": float(scores[idx]),
            })

    return results


if __name__ == "__main__":
    build_index()

    # Quick test
    with open("data/index.pkl", "rb") as f:
        idx = pickle.load(f)

    print("\n--- Test Retrieval ---")
    tests = [
        "Java developer mid-level stakeholder communication",
        "personality assessment for manager",
        "data scientist machine learning Python",
        "entry level customer service",
    ]
    for q in tests:
        results = retrieve(q, idx, top_k=3)
        print(f"\nQuery: '{q}'")
        for r in results:
            print(f"  [{r['_score']:.3f}] {r['name']} ({r['test_types']})")
