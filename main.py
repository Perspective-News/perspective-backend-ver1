#!/usr/bin/env python3
"""
End-to-end news ingestion pipeline for Perspective News
Outputs WEBSITE-COMPATIBLE /data/stories.json

What this script does:
1) Reads sources.csv (rss sources supported here)
2) Fetches + extracts article text
3) Embeds + clusters articles into events
4) Converts each cluster into ONE "story" object for the website
5) Writes stories.json in the schema your frontend expects:
   {
     "generated_at_unix": ...,
     "num_articles": ...,
     "num_stories": ...,
     "stories": [ { id, regionIds, languageIds, countries, headline, summary, image, sources } ... ]
   }
"""

import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from collections import defaultdict

import requests
import pandas as pd
import feedparser
import numpy as np
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


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

DEFAULT_IMAGE_PLACEHOLDER = "assets/placeholder.svg"


# =============================
# TRANSLATION (SAFE, SIMPLE)
# =============================

def translate_to_english(text: str, language: str) -> str:
    """
    Hook for translation.
    Replace body with DeepL / OpenAI / Google if desired.
    """
    if not text or (language or "").strip().lower() == "en":
        return text
    # placeholder: assume translated upstream
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
    return re.sub(r"\s+", " ", (s or "")).strip()


def derive_title_fallback(url: str, content: Optional[str]) -> str:
    if url:
        path = urlparse(url).path
        slug = normalize_whitespace(re.sub(r"[-_]+", " ", path.split("/")[-1]))
        slug = re.sub(r"\.[a-zA-Z0-9]+$", "", slug)
        if slug:
            return slug[:120]
    if content:
        first = normalize_whitespace((content or "").split(".")[0])
        if len(first) >= 8:
            return first[:120]
    return "untitled"


def hash_article(source: str, url: str, title: str) -> str:
    key = f"{source}|{url}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(key).hexdigest()[:16]


def extract_links_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    return list(
        dict.fromkeys(
            a.get("href")
            for a in soup.select("a[href]")
            if a.get("href", "").startswith("http")
        )
    )


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


def normalize_region_id(region: str) -> str:
    return normalize_whitespace(region).lower().replace(" ", "-")


def normalize_language_id(language: str) -> str:
    return normalize_whitespace(language).lower()


def normalize_country_code(country: str) -> str:
    return normalize_whitespace(country).upper()


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

def fetch_from_rss(row: pd.Series, max_items: int) -> List[RawArticle]:
    feed = feedparser.parse(row["source_url"])
    articles: List[RawArticle] = []

    for entry in feed.entries[:max_items]:
        link = getattr(entry, "link", None)
        if not link:
            continue

        content = extract_article_content(link)
        if not content:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
            continue

        content_en = translate_to_english(content, row["language"])

        articles.append(
            RawArticle(
                source_name=str(row.get("source_name", DEFAULT_SOURCE_NAME_IF_MISSING)),
                website_link=link,
                title=getattr(entry, "title", None),
                content=content_en,
                country=str(row.get("country", "") or ""),
                region=str(row.get("region", "") or ""),
                language=str(row.get("language", "") or ""),
            )
        )

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
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        source_type = str(row["source_type"]).lower().strip()
        if source_type == "rss":
            fetched = fetch_from_rss(row, max_items)
        else:
            # only rss implemented in this version
            fetched = []

        for a in fetched:
            rows.append(a.__dict__)

    return pd.DataFrame(rows)


# =============================
# CLEANING
# =============================

def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["source_name"] = df["source_name"].fillna(DEFAULT_SOURCE_NAME_IF_MISSING).astype(str)
    df["website_link"] = df["website_link"].fillna(DEFAULT_WEBSITE_LINK_IF_MISSING).astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["region"] = df["region"].fillna("").astype(str)
    df["language"] = df["language"].fillna("en").astype(str)

    df["title"] = df["title"].apply(normalize_whitespace)
    df["content"] = df["content"].apply(normalize_whitespace)

    # keep only articles with content
    df = df[df["content"].str.len() > 0].copy()

    # ---- FIXED TITLE FALLBACK LOGIC ----
    missing_mask = df["title"] == ""

    if missing_mask.any():
        fallback_titles = (
            df.loc[missing_mask]
            .apply(
                lambda r: derive_title_fallback(
                    r["website_link"],
                    r["content"]
                ),
                axis=1
            )
        )
        # assign using aligned index
        df.loc[missing_mask, "title"] = fallback_titles

    # deterministic article id
    df["article_id"] = df.apply(
        lambda r: hash_article(
            r["source_name"],
            r["website_link"],
            r["title"]
        ),
        axis=1
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
# STORY BUILDING (WEBSITE FORMAT)
# =============================

def summarize(docs: List[str]) -> str:
    """
    Simple extractive summary (first N words across docs).
    Keeps it short for the card UI.
    """
    words = " ".join([d for d in docs if d]).split()
    if not words:
        return ""
    # aim for 50-60 words if possible, but cap at SUMMARY_WORD_MAX
    return " ".join(words[:SUMMARY_WORD_MAX])


def stable_story_id(headline: str, extra: str = "") -> str:
    base = (headline or "") + "|" + (extra or "")
    return f"story-{hashlib.sha1(base.encode('utf-8', errors='ignore')).hexdigest()[:10]}"


def build_stories(df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    """
    Convert each cluster into a website story:
    - id
    - regionIds
    - languageIds
    - countries
    - headline
    - summary
    - image (placeholder)
    - sources: [{name,url}, ...]
    """
    if df.empty:
        return []

    df = df.copy()
    df["cluster_id"] = labels

    stories: List[Dict[str, Any]] = []

    for cid, group in df.groupby("cluster_id"):
        # headline: choose the most common non-empty title (or first)
        titles = [t for t in group["title"].tolist() if t.strip()]
        headline = titles[0] if titles else "Untitled Story"

        # summary from contents
        summary = summarize(group["content"].tolist())

        # regionIds / languageIds / countries (unique, normalized)
        region_ids = sorted({normalize_region_id(r) for r in group["region"].dropna().tolist() if r.strip()})
        language_ids = sorted({normalize_language_id(l) for l in group["language"].dropna().tolist() if l.strip()})
        countries = sorted({normalize_country_code(c) for c in group["country"].dropna().tolist() if c.strip()})

        # sources for modal (one per article)
        sources = []
        seen = set()
        for _, r in group.iterrows():
            name = (r.get("source_name") or DEFAULT_SOURCE_NAME_IF_MISSING).strip()
            url = (r.get("website_link") or DEFAULT_WEBSITE_LINK_IF_MISSING).strip()
            key = f"{name}|{url}"
            if key in seen:
                continue
            seen.add(key)
            sources.append({"name": name, "url": url})

        # ensure non-empty arrays for frontend robustness
        if not region_ids:
            region_ids = ["unknown-region"]
        if not language_ids:
            language_ids = ["en"]
        if not countries:
            countries = ["XX"]

        story = {
            "id": stable_story_id(headline, extra=str(cid)),
            "regionIds": region_ids,
            "languageIds": language_ids,
            "countries": countries,
            "headline": headline,
            "summary": summary,
            "image": DEFAULT_IMAGE_PLACEHOLDER,
            "sources": sources,
        }
        stories.append(story)

    # sort by number of sources descending (website uses "most sources" first)
    stories.sort(key=lambda x: len(x.get("sources", [])), reverse=True)
    return stories


# =============================
# PIPELINE
# =============================

def run_pipeline(
    sources_path: str = "sources.csv",
    output_path: str = "stories.json",
    max_items_per_source: int = 50,
):
    sources = load_sources(sources_path)

    raw = ingest_all_sources(sources, max_items_per_source)
    if raw.empty:
        output = {
            "generated_at_unix": int(time.time()),
            "num_articles": 0,
            "num_stories": 0,
            "stories": [],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"✔ Pipeline complete (no articles) → {output_path}")
        return

    clean = clean_articles(raw)
    if clean.empty:
        output = {
            "generated_at_unix": int(time.time()),
            "num_articles": 0,
            "num_stories": 0,
            "stories": [],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"✔ Pipeline complete (no clean articles) → {output_path}")
        return

    texts = (clean["title"] + ". " + clean["content"].str[:1200]).tolist()
    embeddings = embed_texts(texts)
    labels = cluster_articles(embeddings)

    stories = build_stories(clean, labels)

    output = {
        "generated_at_unix": int(time.time()),
        "num_articles": int(len(clean)),
        "num_stories": int(len(stories)),
        "stories": stories,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✔ Pipeline complete → {output_path}")
    print(f"  Articles: {output['num_articles']}, Stories: {output['num_stories']}")


# =============================
# ENTRY
# =============================

if __name__ == "__main__":
    run_pipeline()
