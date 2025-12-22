#!/usr/bin/env python3
"""
End-to-end news ingestion pipeline for Perspective News.

This script reads a list of news sources from a CSV file, fetches
recent articles from each source (via RSS feeds or simple URL lists),
cleans and normalizes the article metadata, clusters the articles into
events based on semantic similarity, generates a short summary for
each event, and writes the results to a JSON file.

See README.md for usage instructions.
"""

import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests
import pandas as pd
import feedparser
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import trafilatura
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT


# Configuration constants
DEFAULT_WEBSITE_LINK_IF_MISSING = "https://youtube.com"
DEFAULT_SOURCE_NAME_IF_MISSING = "unknown"

USER_AGENT = "PerspectiveNewsPipeline/1.0 (+https://example.com)"
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS_SEC = 0.2

# Clustering controls
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_DISTANCE_THRESHOLD = 0.35

# Summary controls
SUMMARY_WORD_MIN = 50
SUMMARY_WORD_MAX = 60


def safe_get(url: str) -> Optional[str]:
    """Fetch URL content safely; return text or None on error."""
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
    """Collapse multiple whitespace characters and strip leading/trailing space."""
    return re.sub(r"\s+", " ", s).strip()


def derive_title_fallback(url: str, content: Optional[str]) -> str:
    """
    Derive a fallback title when a feed entry is missing one.
    Try to extract a slug from the URL or the first sentence of the content.
    """
    if url:
        path = urlparse(url).path
        slug = path.strip("/").split("/")[-1]
        slug = re.sub(r"[-_]+", " ", slug)
        slug = re.sub(r"\.[a-zA-Z0-9]+$", "", slug)
        slug = normalize_whitespace(slug)
        if slug:
            return slug[:120]

    if content:
        first = content.split(".")[0]
        first = normalize_whitespace(first)
        if len(first) >= 8:
            return first[:120]

    return "untitled"


def hash_article(source: str, url: str, title: str) -> str:
    """Create a short deterministic hash for an article."""
    key = f"{source}|{url}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(key).hexdigest()[:16]


def extract_links_from_html(html: str) -> List[str]:
    """Extract all http links from an HTML page, deduplicated."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if href.startswith("http"):
            links.append(href)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_article_content(url: str) -> Optional[str]:
    """
    Extract the main article text from a given URL using trafilatura.
    Returns None if extraction fails or yields too little content.
    """
    html = safe_get(url)
    if not html:
        return None
    downloaded = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_recall=True,
    )
    if not downloaded:
        return None
    downloaded = normalize_whitespace(downloaded)
    if len(downloaded.split()) < 40:
        return None
    return downloaded


@dataclass
class RawArticle:
    source_name: str
    website_link: str
    title: Optional[str]
    content: Optional[str]


def fetch_from_rss(source_name: str, feed_url: str, max_items: int = 50) -> List[RawArticle]:
    """Fetch articles from an RSS feed."""
    feed = feedparser.parse(feed_url)
    articles: List[RawArticle] = []
    for entry in feed.entries[:max_items]:
        link = getattr(entry, "link", None)
        title = getattr(entry, "title", None)
        if not link:
            continue
        content = extract_article_content(link)
        articles.append(RawArticle(
            source_name=source_name,
            website_link=link,
            title=title,
            content=content,
        ))
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    return articles


def fetch_from_url_list(source_name: str, list_url: str, max_items: int = 50) -> List[RawArticle]:
    """Fetch articles from a plain URL list or HTML page of links."""
    text = safe_get(list_url)
    if not text:
        return []
    urls: List[str] = []
    if "<html" in text.lower():
        urls = extract_links_from_html(text)
    else:
        urls = [line.strip() for line in text.splitlines() if line.strip().startswith("http")]
    urls = urls[:max_items]
    articles: List[RawArticle] = []
    for link in urls:
        content = extract_article_content(link)
        articles.append(RawArticle(
            source_name=source_name,
            website_link=link,
            title=None,
            content=content,
        ))
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    return articles


def load_sources(sources_path: str) -> pd.DataFrame:
    """Load the sources CSV file."""
    df = pd.read_csv(sources_path)
    required = {"source_name", "source_type", "source_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sources file missing columns: {missing}")
    return df


def ingest_all_sources(sources_df: pd.DataFrame, max_items_per_source: int = 50) -> pd.DataFrame:
    """Ingest articles from all sources in the given DataFrame."""
    rows: List[Dict[str, Any]] = []
    for _, row in sources_df.iterrows():
        source_name = row.get("source_name", None)
        source_type = str(row.get("source_type", "")).strip().lower()
        source_url = row.get("source_url", None)
        if not source_url:
            continue
        if not source_name or str(source_name).strip() == "" or str(source_name).lower() == "nan":
            source_name = DEFAULT_SOURCE_NAME_IF_MISSING
        if source_type == "rss":
            fetched = fetch_from_rss(source_name, source_url, max_items=max_items_per_source)
        elif source_type == "url_list":
            fetched = fetch_from_url_list(source_name, source_url, max_items=max_items_per_source)
        else:
            fetched = []
        for art in fetched:
            rows.append({
                "source_name": art.source_name,
                "website_link": art.website_link,
                "title": art.title,
                "content": art.content,
            })
    df = pd.DataFrame(rows)
    return df


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the raw article data according to the specification.

    - Keep only selected columns
    - Fill missing source_name and website_link values
    - Drop rows with missing content
    - Derive missing titles
    - Deduplicate identical articles
    """
    df = df[["source_name", "website_link", "title", "content"]].copy()
    df["source_name"] = df["source_name"].fillna("").astype(str).str.strip()
    df.loc[df["source_name"] == "", "source_name"] = DEFAULT_SOURCE_NAME_IF_MISSING
    df["website_link"] = df["website_link"].fillna("").astype(str).str.strip()
    df.loc[df["website_link"] == "", "website_link"] = DEFAULT_WEBSITE_LINK_IF_MISSING
    df["content"] = df["content"].fillna("").astype(str)
    df["content"] = df["content"].apply(normalize_whitespace)
    df = df[df["content"].str.len() > 0].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["title"] = df["title"].apply(lambda s: normalize_whitespace(s) if s else "")
    missing_title = df["title"].eq("")
    df.loc[missing_title, "title"] = df.loc[missing_title].apply(
        lambda r: derive_title_fallback(r["website_link"], r["content"]), axis=1
    )
    df["article_id"] = df.apply(lambda r: hash_article(r["source_name"], r["website_link"], r["title"]), axis=1)
    df = df.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    return df


def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Embed a list of texts using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return emb


def cluster_articles(embeddings: np.ndarray, distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD) -> np.ndarray:
    """Cluster articles using agglomerative clustering on cosine distance."""
    distances = 1.0 - cosine_similarity(embeddings)
    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = model.fit_predict(distances)
    return labels


def pick_event_name(docs: List[str], kw_model: KeyBERT, top_n: int = 5) -> str:
    """Generate a human-readable name for an event using KeyBERT."""
    joined = " ".join(docs)[:10000]
    keywords = kw_model.extract_keywords(
        joined,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
    )
    if not keywords:
        return "Untitled event"
    phrases = [k[0] for k in keywords[:2]]
    name = " — ".join([p.title() for p in phrases])
    return name if name else "Untitled event"


def summarize_50_60_words(docs: List[str]) -> str:
    """
    Summarize a list of documents into 50–60 words using a simple
    extractive method that picks sentences most similar to the centroid
    of all sentence embeddings.
    """
    text = " ".join(docs)
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [normalize_whitespace(s) for s in sents if len(s.split()) >= 8]
    if not sents:
        words = text.split()
        return " ".join(words[:SUMMARY_WORD_MAX])
    sent_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    sent_emb = sent_model.encode(sents, normalize_embeddings=True)
    centroid = np.mean(sent_emb, axis=0, keepdims=True)
    sims = cosine_similarity(sent_emb, centroid).reshape(-1)
    ranked_idx = np.argsort(-sims)
    chosen = []
    word_count = 0
    used = set()
    for idx in ranked_idx:
        if idx in used:
            continue
        sent = sents[idx]
        sent_words = sent.split()
        if word_count < SUMMARY_WORD_MIN:
            chosen.append(sent)
            word_count += len(sent_words)
            used.add(idx)
        if word_count >= SUMMARY_WORD_MIN:
            break
    summary = " ".join(chosen)
    summary_words = summary.split()
    if len(summary_words) > SUMMARY_WORD_MAX:
        summary = " ".join(summary_words[:SUMMARY_WORD_MAX])
    if len(summary.split()) < SUMMARY_WORD_MIN:
        for idx in ranked_idx:
            if idx in used:
                continue
            sent = sents[idx]
            summary_words = (summary + " " + sent).split()
            if len(summary_words) >= SUMMARY_WORD_MIN:
                summary = " ".join(summary_words[:SUMMARY_WORD_MAX])
                break
            summary = " ".join(summary_words)
            used.add(idx)
    return normalize_whitespace(summary)


def build_event_outputs(df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Build a structured output for each event cluster."""
    df = df.copy()
    df["event_id"] = labels
    kw_model = KeyBERT(model=SentenceTransformer(EMBEDDING_MODEL_NAME))
    events = []
    for event_id, group in df.groupby("event_id"):
        docs = group["content"].tolist()
        event_name = pick_event_name(docs, kw_model)
        summary = summarize_50_60_words(docs)
        sources = []
        for _, r in group.iterrows():
            sources.append({
                "source_name": r["source_name"],
                "website_link": r["website_link"],
                "title": r["title"],
                "article_id": r["article_id"],
            })
        events.append({
            "event_id": int(event_id),
            "event_name": event_name,
            "summary_50_60_words": summary,
            "sources": sources,
            "num_articles": int(len(group)),
        })
    events.sort(key=lambda x: x["num_articles"], reverse=True)
    return events


def run_pipeline(
    sources_path: str = "sources.csv",
    output_path: str = "output_events.json",
    max_items_per_source: int = 50,
) -> None:
    """
    Main entry point: load sources, fetch and process articles, cluster them,
    generate summaries, and write the output.
    """
    print(f"Loading sources from: {sources_path}")
    sources_df = load_sources(sources_path)
    print("Ingesting articles from all sources...")
    raw_df = ingest_all_sources(sources_df, max_items_per_source=max_items_per_source)
    print(f"Fetched rows: {len(raw_df)}")
    print("Cleaning articles...")
    clean_df = clean_articles(raw_df)
    print(f"Rows after cleaning: {len(clean_df)}")
    if len(clean_df) == 0:
        print("No articles left after cleaning. Exiting.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"events": []}, f, ensure_ascii=False, indent=2)
        return
    print("Embedding articles for clustering...")
    cluster_texts = (clean_df["title"] + ". " + clean_df["content"].str.slice(0, 1200)).tolist()
    embeddings = embed_texts(cluster_texts)
    print("Clustering into events...")
    labels = cluster_articles(embeddings, distance_threshold=CLUSTER_DISTANCE_THRESHOLD)
    print("Building event outputs (naming + summaries)...")
    events = build_event_outputs(clean_df, labels)
    output = {
        "generated_at_unix": int(time.time()),
        "num_articles": int(len(clean_df)),
        "num_events": int(len(events)),
        "events": events,
    }
    print(f"Writing output to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    run_pipeline()
