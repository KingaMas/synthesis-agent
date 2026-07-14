#!/usr/bin/env python
"""
Fetch publication years for all recipe DOIs from Crossref (polite pool).

Uses the batch filter endpoint (50 DOIs per request) with exponential
backoff on 429/5xx. Writes results/doi_years.json incrementally and is
fully resumable. Transient failures are NOT cached; DOIs Crossref does
not know are cached as null after the batch pass confirms them missing.

Usage
-----
    PYTHONPATH=. python scripts/fetch_doi_years.py [--batch-size 50]

The output file is committed: the temporal split must be reproducible
without network access.
"""

import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("results/doi_years.json")
MAILTO = "kinga.oliwia.mastej@gmail.com"  # Crossref polite-pool contact
BASE_DELAY_S = 1.0                        # ~1 request/s steady state


def _request(url: str) -> dict:
    """GET with exponential backoff on 429/5xx; honors Retry-After."""
    delay = 5.0
    for attempt in range(8):
        req = urllib.request.Request(
            url, headers={"User-Agent": f"sky-benchmark (mailto:{MAILTO})"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"  HTTP {e.code}, backing off {wait:.0f}s", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 300)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError(f"giving up after 8 attempts: {url}")


def extract_year(msg: dict) -> int | None:
    years = []
    for field in ("published-print", "published-online", "issued"):
        parts = (msg.get(field) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            years.append(int(parts[0][0]))
    return min(years) if years else None


def fetch_batch(dois: list[str]) -> dict[str, int | None]:
    """Batch lookup via the works filter endpoint. Missing DOIs -> null."""
    flt = ",".join("doi:" + d for d in dois)
    url = (
        "https://api.crossref.org/works?filter="
        + urllib.parse.quote(flt, safe=",:")
        + f"&rows={len(dois)}&select=DOI,published-print,published-online,issued"
        + f"&mailto={MAILTO}"
    )
    msg = _request(url)["message"]
    found = {
        item["DOI"].lower(): extract_year(item) for item in msg.get("items", [])
    }
    # DOIs absent from the response do not exist in Crossref -> cache null
    return {d: found.get(d.lower()) for d in dois}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    with gzip.open("assets/mp_synthesis_recipes.json.gz") as f:
        recipes = json.load(f)
    dois = sorted({r.get("doi") for r in recipes if r.get("doi")})

    years: dict = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [d for d in dois if d not in years]
    print(f"{len(dois)} unique DOIs, {len(years)} cached, {len(todo)} to fetch")

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        years.update(fetch_batch(batch))
        OUT.write_text(json.dumps(years, indent=0, sort_keys=True))
        found = sum(1 for v in years.values() if v)
        print(
            f"  {start + len(batch)}/{len(todo)} fetched "
            f"({found}/{len(years)} with year)",
            flush=True,
        )
        time.sleep(BASE_DELAY_S)

    found = sum(1 for v in years.values() if v)
    print(f"done: {found}/{len(years)} DOIs resolved to a year -> {OUT}")


if __name__ == "__main__":
    main()
