# AnonModel

AnonModel is an open-source AI model data station: a static dashboard that
snapshots the [Hugging Face](https://huggingface.co) public model API and
turns it into something browsable and comparable.

It surfaces:

- **TOP10** — the ten currently trending `text-generation` models, ranked by
  Hugging Face's `trendingScore` (not raw downloads, which favors long-tail
  embedding/classification models over what people are actually excited
  about right now).
- **Model list** — the full snapshot, searchable by id/author/tags and
  sortable by downloads, all-time downloads, likes, or trending score, with
  a pipeline-tag filter.
- **Provider list** — models aggregated by author/organization (model count,
  total downloads), so you can see who's shipping.
- **GPU compatibility tables** — for RTX 5090 (32GB), RTX 4090 (24GB), and
  RTX 3090 (24GB), showing which models with a known parameter count fit at
  which quantization level.

## Data pipeline

All data comes from the public, unauthenticated Hugging Face models API
(`https://huggingface.co/api/models`) — no scraping, no private endpoints.

- **`scripts/fetch_snapshot.py`** — a stdlib-only Python 3 script (no
  third-party dependencies) that pulls a fresh snapshot of model metadata
  (author, downloads, likes, trending score, pipeline tag, library, tags,
  license, parameter count where available) and writes it to
  `data/models.json`. It throttles requests to stay well under Hugging
  Face's unauthenticated rate limit (500 requests / 300s) and takes a few
  minutes to run.
- **`.github/workflows/refresh-snapshot.yml`** — a scheduled GitHub Actions
  workflow that re-runs the fetch script once a day (04:17 UTC, off-peak)
  and can also be triggered manually via `workflow_dispatch`. If
  `data/models.json` changed, it commits and pushes the update using the
  default `GITHUB_TOKEN` (no PAT required) under the `github-actions[bot]`
  identity. If nothing changed, it skips the commit — no empty commits.
- **`data/models.json`** — the committed snapshot the frontend reads
  directly via `fetch()`. This means the site has no backend and no
  database: every deploy just serves the latest committed JSON.

### GPU compatibility formula

For the 190-or-so models that expose a parameter count, estimated VRAM is:

```
VRAM ≈ params × bytes/param + 1.5GB
```

with `bytes/param` roughly `2.0` for fp16, `1.07` for Q8, and `0.6` for Q4
quantization. RTX 4090 and RTX 3090 share the same 24GB capacity but are
shown with different recommended quantization levels (3090: conservative
Q4; 4090: Q5/Q6 workable), while the RTX 5090's 32GB headroom fits larger
models at higher precision.

## Running locally

The page fetches `data/models.json` at runtime, so it **must be served over
HTTP** — opening `index.html` directly as a `file://` URL will fail (the
fetch is blocked by the browser's CORS policy for local files).

```bash
python -m http.server 22225
```

Then open [http://localhost:22225](http://localhost:22225).

To pull a fresh snapshot before serving:

```bash
python scripts/fetch_snapshot.py
```

## Deployment

AnonModel is deployed as a static site on **Cloudflare Pages**, connected
directly to this GitHub repository:

- **Framework preset:** `None`
- **Build command:** *(none)*
- **Build output directory:** `/`

Every push to `main` — including the automated daily snapshot refresh —
triggers a redeploy, so the live site stays in sync with the latest data
without any manual step.

## Disclaimer

AnonModel is an **unofficial**, community-built mirror and dashboard built
on top of the public Hugging Face API. It is **not affiliated with, endorsed
by, or sponsored by Hugging Face**. All model data belongs to its
respective authors and is subject to Hugging Face's own terms; this project
only aggregates and visualizes what is already publicly available through
their API.

## License

MIT — see [LICENSE](./LICENSE).
