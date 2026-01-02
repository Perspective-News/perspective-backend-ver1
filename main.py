#!/usr/bin/env python3
"""
End-to-end news ingestion pipeline for Perspective News
(extended with translation, region grouping, and country coverage)
"""

import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from collections import Counter, defaultdict

import requests
import pandas as pd
import feedparser
import numpy as np
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# =============================
# CONFIG
# =============================

USER_AGENT = "PerspectiveNewsPipeline/1.0"
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS_SEC = 0.2

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_DISTANCE_THRESHOLD = 0.35

SUMMARY_WORD_MIN = 50
SUMMARY_WORD_MAX = 60

DEFAULT_WEBSITE_LINK_IF_MISSING = "https://example.com"
DEFAULT_SOURCE_NAME_IF_MISSING = "unknown"

# =============================
# TRANSLATION (SAFE, SIMPLE)
# =============================

def translate_to_english(text: str, language: str) -> str:
    """
    Hook for translation.
    Replace body with DeepL / OpenAI / Google if desired.
    """
    if not text or language == "en":
        return text
    # ---- placeholder: assume translated upstream ----
    return text

# =============================
# UTILS
# =============================

def safe_get(url: str) -> Optional[str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            return None
        return resp.text
    except Exception:
        return None


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def derive_title_fallback(url: str, content: Optional[str]) -> str:
    if url:
        path = urlparse(url).path
        slug = normalize_whitespace(re.sub(r"[-_]+", " ", path.split("/")[-1]))
        slug = re.sub(r"\.[a-zA-Z0-9]+$", "", slug)
        if slug:
            return slug[:120]
    if content:
        first = normalize_whitespace(content.split(".")[0])
        if len(first) >= 8:
            return first[:120]
    return "untitled"


def hash_article(source: str, url: str, title: str) -> str:
    key = f"{source}|{url}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(key).hexdigest()[:16]


def extract_links_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    return list(dict.fromkeys(
        a.get("href") for a in soup.select("a[href]") if a.get("href", "").startswith("http")
    ))


def extract_article_content(url: str) -> Optional[str]:
    html = safe_get(url)
    if not html:
        return None
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    if not text:
        return None
    text = normalize_whitespace(text)
    return text if len(text.split()) >= 40 else None

# =============================
# DATA STRUCTURE
# =============================

@dataclass
class RawArticle:
    source_name: str
    website_link: str
    title: Optional[str]
    content: Optional[str]
    country: str
    region: str
    language: str

# =============================
# INGESTION
# =============================

def fetch_from_rss(row, max_items: int) -> List[RawArticle]:
    feed = feedparser.parse(row["source_url"])
    articles = []
    for entry in feed.entries[:max_items]:
        link = getattr(entry, "link", None)
        if not link:
            continue
        content = extract_article_content(link)
        content_en = translate_to_english(content, row["language"])
        articles.append(RawArticle(
            source_name=row["source_name"],
            website_link=link,
            title=getattr(entry, "title", None),
            content=content_en,
            country=row["country"],
            region=row["region"],
            language=row["language"],
        ))
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    return articles


def load_sources(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    required = {"source_name", "source_type", "source_url", "language", "country", "region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sources.csv: {missing}")
    return df


def ingest_all_sources(df: pd.DataFrame, max_items: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        if row["source_type"].lower() == "rss":
            fetched = fetch_from_rss(row, max_items)
        else:
            fetched = []
        for a in fetched:
            rows.append(a.__dict__)
    return pd.DataFrame(rows)

# =============================
# CLEANING
# =============================

def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["title"] = df["title"].apply(normalize_whitespace)
    df["content"] = df["content"].apply(normalize_whitespace)
    df = df[df["content"].str.len() > 0]

    missing = df["title"] == ""
    df.loc[missing, "title"] = df.loc[missing].apply(
        lambda r: derive_title_fallback(r["website_link"], r["content"]), axis=1
    )

    df["article_id"] = df.apply(
        lambda r: hash_article(r["source_name"], r["website_link"], r["title"]), axis=1
    )

    return df.drop_duplicates("article_id").reset_index(drop=True)

# =============================
# CLUSTERING
# =============================

def embed_texts(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def cluster_articles(embeddings: np.ndarray) -> np.ndarray:
    dist = 1.0 - cosine_similarity(embeddings)
    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        n_clusters=None,
    )
    return model.fit_predict(dist)

# =============================
# EVENT BUILDING
# =============================

def summarize(docs: List[str]) -> str:
    words = " ".join(docs).split()
    return " ".join(words[:SUMMARY_WORD_MAX])


def build_events(df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    df["event_id"] = labels
    kw_model = KeyBERT(model=SentenceTransformer(EMBEDDING_MODEL_NAME))
    events = []

    for eid, group in df.groupby("event_id"):
        docs = group["content"].tolist()
        name = pick_event_name(docs, kw_model)
        summary = summarize(docs)

        region_map = defaultdict(list)
        for _, r in group.iterrows():
            region_map[r["region"]].append(r["article_id"])

        country_counts = Counter(group["country"])
        total = sum(country_counts.values())
        country_coverage = {
            k: round(v / total * 100, 2) for k, v in country_counts.items()
        }

        events.append({
            "event_id": int(eid),
            "event_name": name,
            "summary_50_60_words": summary,
            "regions": dict(region_map),
            "country_coverage": country_coverage,
            "num_articles": int(len(group)),
        })

    return sorted(events, key=lambda x: x["num_articles"], reverse=True)


def pick_event_name(docs, kw_model):
    joined = " ".join(docs)[:8000]
    kws = kw_model.extract_keywords(joined, top_n=2, stop_words="english")
    return " — ".join(k[0].title() for k in kws) if kws else "Untitled Event"

# =============================
# PIPELINE
# =============================

def run_pipeline(
    sources_path="sources.csv",
    output_path="output_events.json",
    max_items_per_source=50,
):
    sources = load_sources(sources_path)
    raw = ingest_all_sources(sources, max_items_per_source)
    clean = clean_articles(raw)

    texts = (clean["title"] + ". " + clean["content"].str[:1200]).tolist()
    embeddings = embed_texts(texts)
    labels = cluster_articles(embeddings)

    events = build_events(clean, labels)

    output = {
        "generated_at_unix": int(time.time()),
        "num_articles": len(clean),
        "num_events": len(events),
        "events": events,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✔ Pipeline complete → {output_path}")

# =============================
# ENTRY
# =============================

if __name__ == "__main__":
    run_pipeline()