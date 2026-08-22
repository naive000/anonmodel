#!/usr/bin/env python3
"""Pull a static snapshot of Hugging Face model data into data/*.json.

Rate limit (unauthenticated): 500 req / 300s (confirmed live via the API's
own `ratelimit` / `ratelimit-policy` response headers: a fixed 300s window,
quota 500). THROTTLE_SECONDS is set to stay well under that even at the
expanded ~1,300-request detail-fetch volume this script now does; a run
takes roughly 12-20 minutes and is meant to run unattended (CI/cron), not
interactively.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models"
DATA_DIR = Path(__file__).parent.parent / "data"
THROTTLE_SECONDS = 0.65

# NOTE on the expand[] gotcha (re-verified live 2026-08-23): passing ANY
# expand[] param to /api/models REPLACES the entire default field set
# rather than adding to it. Every field this script needs back (including
# `downloads` and `gated`, see below) must live in this one list and be
# passed together, never as a separate/standalone expand[] param.
#
# `downloads` bug fix: earlier versions of this script omitted `downloads`
# from LIST_FIELDS and relied on an undocumented quirk where the API
# auto-includes `downloads` in the response ONLY when the query itself
# uses `sort=downloads` (verified live). Any fetch pass sorted a different
# way (trendingScore, or a `search=` pass) silently got downloads=0, and
# since dedupe() is last-write-wins, those zeroed records could clobber a
# real downloads value already collected from an earlier pass. This
# affects task #5's "top-N by downloads" detail-fetch selection and now
# matters more because of the new search passes below, so `downloads` is
# listed explicitly here (confirmed: explicit expand[]=downloads returns
# the real count regardless of sort order).
#
# `gated` (verified live): present directly on /api/models list responses
# when explicitly requested via expand[] — no per-model detail-fetch
# needed for it. Observed values: `false` (JSON bool) for ungated models,
# or the strings "manual" / "auto" for gated ones. Bare `true` was never
# observed in sampling; treat any non-`false` value as gated, and the
# string as the gate mode.
LIST_FIELDS = ["author", "downloads", "downloadsAllTime", "likes",
               "trendingScore", "pipeline_tag", "library_name", "tags",
               "lastModified", "gated"]

PIPELINE_TAGS_FOR_COVERAGE = [
    "text-generation", "text-to-image", "image-text-to-text",
    "text-to-speech", "automatic-speech-recognition", "feature-extraction",
    "sentence-similarity", "text-classification", "image-classification",
    "text-to-video",
]

# Modalities that get a detail-fetch pass (license/params) beyond
# text-generation, selected from the pool already gathered above (no new
# list calls needed — these pipeline_tags are already in
# PIPELINE_TAGS_FOR_COVERAGE).
FLAGSHIP_PIPELINE_TAGS = [
    "image-text-to-text", "text-to-image", "text-to-speech",
    "automatic-speech-recognition", "text-to-video",
]

# search= queries used to surface niche/low-download uncensored models
# that wouldn't otherwise rank into the popularity-sorted pools above.
# Deliberately excludes "dpo" (a training method used by fully-aligned
# models too, not a signal of uncensored-ness).
UNCENSORED_SEARCH_TERMS = ["uncensored", "abliterated", "heretic"]

# Seed signal for the uncensored auto-tagger: substring match (case
# insensitive) against a record's id or any of its tags.
UNCENSORED_PATTERN = re.compile(r"uncensor|abliterat|heretic", re.IGNORECASE)

# base_model:<relation>:<parent-id> relation types actually observed in
# this dataset (verified by grepping tags[] for "base_model:" across all
# 5,156 existing records: quantized=1330, finetune=652, adapter=66,
# merge=37 occurrences; no other relation keyword occurs). A `quantized`
# child of a flagged parent is the same weights just requantized, so it
# inherits the parent's exact reason string. finetune/merge/adapter can
# re-add alignment, so they only get a weaker "lineage:<parent-id>" claim.
UNCENSORED_SAME_REASON_RELATIONS = {"quantized"}
UNCENSORED_LINEAGE_RELATIONS = {"finetune", "merge", "adapter"}


def _get(url, max_retries=4):
    req = urllib.request.Request(url, headers={"User-Agent": "oss-model-station/0.1"})
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
                link = resp.headers.get("Link", "")
            time.sleep(THROTTLE_SECONDS)
            return body, link
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                wait = 30 * (attempt + 1)
                print(f"  429 rate-limited, backing off {wait}s...")
                time.sleep(wait)
                continue
            raise


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


def _license_from_tags(tags):
    """HF encodes license as a `license:xxx` tag for ~85% of records even
    when the structured cardData/detail license field is empty. Parse it
    as a fallback wherever the authoritative source didn't populate it."""
    return next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)


def fetch_detail(model_id):
    url = f"{API}/{urllib.parse.quote(model_id)}"
    try:
        body, _ = _get(url)
    except Exception:
        return None
    card = body.get("cardData") or {}
    return {
        "license": card.get("license") or _license_from_tags(body.get("tags", [])),
        "params": (body.get("safetensors") or {}).get("total"),
    }


def dedupe(models):
    seen = {}
    for m in models:
        seen[m["id"]] = m
    return list(seen.values())


def normalize(raw):
    tags = raw.get("tags", [])
    return {
        "id": raw["id"],
        "author": raw.get("author") or raw["id"].split("/")[0],
        "downloads": raw.get("downloads", 0),
        "downloadsAllTime": raw.get("downloadsAllTime", raw.get("downloads", 0)),
        "likes": raw.get("likes", 0),
        "trendingScore": raw.get("trendingScore", 0),
        "pipeline_tag": raw.get("pipeline_tag"),
        "library_name": raw.get("library_name"),
        "tags": tags,
        "lastModified": raw.get("lastModified"),
        # Cheap fallback populated for ~85% of records at zero extra API
        # cost; may be overwritten with a more authoritative cardData
        # license by the detail-fetch pass later (never overwritten with
        # a worse/empty value — see the guard in main()).
        "license": _license_from_tags(tags),
        "params": None,
        # `false` (ungated), or "manual" / "auto" (gated) — see LIST_FIELDS
        # comment above for the verified value shapes.
        "gated": raw.get("gated", False),
        # Filled in by compute_uncensored() after the full pool (including
        # the search=uncensored/abliterated/heretic passes) is merged and
        # deduped. These two fields are ANONMODEL-COMPUTED inferences, not
        # data Hugging Face itself publishes or labels — see
        # compute_uncensored()'s docstring.
        "uncensored": False,
        "uncensored_reason": None,
    }


def compute_uncensored(all_models):
    """Populate `uncensored` / `uncensored_reason` on every record in place.

    IMPORTANT: these are fields AnonModel computes by inference, not data
    Hugging Face published or labels itself. The UI must render them as an
    inference (e.g. "推定"/inferred, with the reason visible) — never as
    if HF itself tagged the model this way.

    uncensored_reason is one of:
      - null            no signal found
      - "self-tagged"   id or any tag matches /uncensor|abliterat|heretic/i
      - "nfaa-tag"      carries the `not-for-all-audiences` tag (weaker,
                         supplementary signal; only applied when the
                         self-tagged check above didn't already match)
      - "lineage:<parent-id>"
                         a finetune/merge/adapter of a flagged record (a
                         claim, not a certainty — finetuning can re-add
                         alignment)
    A `base_model:quantized:<parent-id>` child of a flagged record is the
    same weights just requantized, so it inherits the parent's exact
    reason string verbatim (including an inherited "lineage:X") rather
    than getting its own "lineage:" wrapper.

    Known caveat (seed regex is intentionally broad, per spec: "id or ANY
    tag matches"): a plain `base_model:<parent-id>` tag (no relation
    keyword) or a `base_model:finetune:<parent-id>` tag whose parent id
    itself contains "uncensored"/"abliterated"/"heretic" also matches the
    seed pattern, so some records get classified "self-tagged" that are
    really lineage derivatives whose parent's *name* leaked the substring
    into their own tags. This inflates self-tagged counts and deflates
    lineage counts somewhat — report this caveat alongside the
    distribution numbers rather than treating self-tagged as 100% "this
    specific repo announced uncensored-ness about itself".

    Returns the number of BFS passes it took to reach a fixed point (the
    max lineage hop-depth reached), for reporting.
    """
    reason = {}

    for m in all_models:
        if UNCENSORED_PATTERN.search(m["id"]) or any(
                UNCENSORED_PATTERN.search(t) for t in m["tags"]):
            reason[m["id"]] = "self-tagged"
    for m in all_models:
        if m["id"] not in reason and "not-for-all-audiences" in m["tags"]:
            reason[m["id"]] = "nfaa-tag"

    # child_id -> [(relation, parent_id), ...] parsed once up front.
    child_relations = {}
    for m in all_models:
        rels = []
        for t in m["tags"]:
            if t.startswith("base_model:"):
                parts = t.split(":", 2)
                if len(parts) == 3 and (
                        parts[1] in UNCENSORED_SAME_REASON_RELATIONS
                        or parts[1] in UNCENSORED_LINEAGE_RELATIONS):
                    rels.append((parts[1], parts[2]))
        if rels:
            child_relations[m["id"]] = rels

    depth = 0
    changed = True
    while changed:
        changed = False
        depth += 1
        for child_id, rels in child_relations.items():
            if child_id in reason:
                continue
            for relation, parent_id in rels:
                parent_reason = reason.get(parent_id)
                if parent_reason is None:
                    continue
                if relation in UNCENSORED_SAME_REASON_RELATIONS:
                    reason[child_id] = parent_reason
                else:
                    reason[child_id] = f"lineage:{parent_id}"
                changed = True
                break

    for m in all_models:
        r = reason.get(m["id"])
        m["uncensored_reason"] = r
        m["uncensored"] = r is not None

    return depth - 1  # last pass made no changes; depth-1 = real hop count


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

    print("fetching uncensored-signal search passes...")
    uncensored_search = []
    for term in UNCENSORED_SEARCH_TERMS:
        results = fetch_list(
            {"search": term, "sort": "downloads", "direction": -1,
             "limit": 100, "expand[]": LIST_FIELDS},
            max_items=100,
        )
        uncensored_search.extend(results)
        print(f"  search={term}: {len(results)} results")

    all_models = dedupe([normalize(m) for m in
                          top_downloads + per_tag + trending_textgen +
                          uncensored_search])
    print(f"total unique models: {len(all_models)}")

    print("computing uncensored seed + lineage propagation...")
    bfs_depth = compute_uncensored(all_models)
    uncensored_models = [m for m in all_models if m["uncensored"]]
    self_tagged_n = sum(1 for m in all_models if m["uncensored_reason"] == "self-tagged")
    nfaa_n = sum(1 for m in all_models if m["uncensored_reason"] == "nfaa-tag")
    lineage_n = len(uncensored_models) - self_tagged_n - nfaa_n
    print(f"  uncensored total: {len(uncensored_models)} "
          f"(self-tagged: {self_tagged_n}, nfaa-tag: {nfaa_n}, "
          f"lineage-propagated: {lineage_n}, BFS depth: {bfs_depth})")

    print("selecting detail-fetch targets "
          "(text-gen top500 + all uncensored + flagship top100s)...")
    detail_targets = {}
    textgen_top500 = sorted(
        [m for m in all_models if m["pipeline_tag"] == "text-generation"],
        key=lambda m: m["downloads"], reverse=True,
    )[:500]
    for m in textgen_top500:
        detail_targets[m["id"]] = m
    for m in uncensored_models:
        detail_targets[m["id"]] = m
    for tag in FLAGSHIP_PIPELINE_TAGS:
        top100 = sorted(
            [m for m in all_models if m["pipeline_tag"] == tag],
            key=lambda m: m["downloads"], reverse=True,
        )[:100]
        for m in top100:
            detail_targets[m["id"]] = m

    targets = list(detail_targets.values())
    print(f"fetching detail (license + params) for {len(targets)} models...")
    for i, m in enumerate(targets):
        detail = fetch_detail(m["id"])
        if detail:
            # Only override when the detail-fetch actually found something
            # better — never clobber the tag-fallback license/params set
            # in normalize() with an empty detail-fetch result.
            if detail["license"]:
                m["license"] = detail["license"]
            if detail["params"] is not None:
                m["params"] = detail["params"]
        if (i + 1) % 50 == 0:
            print(f"  detail {i + 1}/{len(targets)}")

    license_n = sum(1 for m in all_models if m["license"])
    gated_n = sum(1 for m in all_models if m["gated"] not in (False, None))
    print(f"license coverage: {license_n}/{len(all_models)} "
          f"({100 * license_n / len(all_models):.1f}%)")
    print(f"gated (non-false): {gated_n}/{len(all_models)}")

    out_path = DATA_DIR / "models.json"
    out_path.write_text(json.dumps(all_models, ensure_ascii=False))
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"wrote {out_path} ({size_mb:.2f} MB, {len(all_models)} models)")


if __name__ == "__main__":
    main()
