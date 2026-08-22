#!/usr/bin/env python3
"""Pull a static snapshot of Hugging Face model data into data/*.json.

Rate limit (unauthenticated): 500 req / 300s. This script throttles to stay
well under that and takes single-digit minutes to run.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models"
DATA_DIR = Path(__file__).parent.parent / "data"
THROTTLE_SECONDS = 0.4

LIST_FIELDS = ["author", "downloadsAllTime", "likes",
               "trendingScore", "pipeline_tag", "library_name", "tags",
               "lastModified"]

PIPELINE_TAGS_FOR_COVERAGE = [
    "text-generation", "text-to-image", "image-text-to-text",
    "text-to-speech", "automatic-speech-recognition", "feature-extraction",
    "sentence-similarity", "text-classification", "image-classification",
    "text-to-video",
]


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "oss-model-station/0.1"})
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
        link = resp.headers.get("Link", "")
    time.sleep(THROTTLE_SECONDS)
    return body, link


def _next_cursor_url(link_header):
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header or "")
    return m.group(1) if m else None


def fetch_list(params, max_items):
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{API}?{qs}"
    items = []
    while url and len(items) < max_items:
        body, link = _get(url)
        if not body:
            break
        items.extend(body)
        url = _next_cursor_url(link)
    return items[:max_items]


def fetch_detail(model_id):
    url = f"{API}/{urllib.parse.quote(model_id)}"
    try:
        body, _ = _get(url)
    except Exception:
        return None
    card = body.get("cardData") or {}
    license_tag = next((t.split(":", 1)[1] for t in body.get("tags", [])
                         if t.startswith("license:")), None)
    return {
        "license": card.get("license") or license_tag,
        "params": (body.get("safetensors") or {}).get("total"),
    }


def dedupe(models):
    seen = {}
    for m in models:
        seen[m["id"]] = m
    return list(seen.values())


def normalize(raw):
    return {
        "id": raw["id"],
        "author": raw.get("author") or raw["id"].split("/")[0],
        "downloads": raw.get("downloads", 0),
        "downloadsAllTime": raw.get("downloadsAllTime", raw.get("downloads", 0)),
        "likes": raw.get("likes", 0),
        "trendingScore": raw.get("trendingScore", 0),
        "pipeline_tag": raw.get("pipeline_tag"),
        "library_name": raw.get("library_name"),
        "tags": raw.get("tags", []),
        "lastModified": raw.get("lastModified"),
        "license": None,
        "params": None,
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("fetching top 5000 by downloads...")
    top_downloads = fetch_list(
        {"sort": "downloads", "direction": -1, "limit": 100,
         "expand[]": LIST_FIELDS},
        max_items=5000,
    )

    print("fetching top 100 per pipeline_tag...")
    per_tag = []
    for tag in PIPELINE_TAGS_FOR_COVERAGE:
        per_tag.extend(fetch_list(
            {"pipeline_tag": tag, "sort": "downloads", "direction": -1,
             "limit": 100, "expand[]": LIST_FIELDS},
            max_items=100,
        ))
        print(f"  {tag}: done")

    print("fetching trending text-generation for TOP10 homepage pool...")
    trending_textgen = fetch_list(
        {"pipeline_tag": "text-generation", "sort": "trendingScore",
         "direction": -1, "limit": 100, "expand[]": LIST_FIELDS},
        max_items=100,
    )

    all_models = dedupe([normalize(m) for m in
                          top_downloads + per_tag + trending_textgen])
    print(f"total unique models: {len(all_models)}")

    print("fetching detail (license + params) for top 200 text-generation...")
    textgen_sorted = sorted(
        [m for m in all_models if m["pipeline_tag"] == "text-generation"],
        key=lambda m: m["downloads"], reverse=True,
    )[:200]
    for i, m in enumerate(textgen_sorted):
        detail = fetch_detail(m["id"])
        if detail:
            m["license"] = detail["license"]
            m["params"] = detail["params"]
        if (i + 1) % 25 == 0:
            print(f"  detail {i + 1}/{len(textgen_sorted)}")

    out_path = DATA_DIR / "models.json"
    out_path.write_text(json.dumps(all_models, ensure_ascii=False))
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"wrote {out_path} ({size_mb:.2f} MB, {len(all_models)} models)")


if __name__ == "__main__":
    main()
