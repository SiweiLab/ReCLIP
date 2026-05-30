#!/usr/bin/env python3
"""Download AlphaMissense PDB files for Target_UPID list.

URL template:
  https://alphamissense.hegelab.org/pdb/AF-{UNIPROT_ACC}-F1-AM_v4.pdb

Reads Target_UPID from the TSV and downloads unique IDs in parallel.
Skips missing (404) and already-downloaded files.
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Set, Tuple

URL_TEMPLATE = "https://alphamissense.hegelab.org/pdb/AF-{upid}-F1-AM_v4.pdb"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_target_upids(tsv_path: str) -> List[str]:
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        try:
            idx = header.index("Target_UPID")
        except ValueError as e:
            raise ValueError("Target_UPID column not found in TSV header") from e

        upids: Set[str] = set()
        for row in reader:
            if not row:
                continue
            if idx >= len(row):
                continue
            upid = row[idx].strip()
            if upid:
                upids.add(upid)
    return sorted(upids)


def download_one(upid: str, out_dir: str, timeout: int) -> Tuple[str, str]:
    """Return (status, upid). status in {downloaded, exists, not_found, error}."""
    out_path = os.path.join(out_dir, f"AF-{upid}-F1-AM_v4.pdb")
    if os.path.exists(out_path):
        return ("exists", upid)

    url = URL_TEMPLATE.format(upid=upid)
    req = urllib.request.Request(url, method="GET")
    tmp_path = out_path + ".tmp"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return ("error", upid)
            with open(tmp_path, "wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
        os.replace(tmp_path, out_path)
        return ("downloaded", upid)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ("not_found", upid)
        return ("error", upid)
    except Exception:
        return ("error", upid)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def chunks(it: Iterable[str], n: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in it:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AlphaMissense PDBs for Target_UPID list.")
    parser.add_argument(
        "--tsv",
        default=str(PROJECT_ROOT / "data" / "four_classes_mutation" / "Mutation_IMEx_IntAct_clean.tsv"),
        help="Path to Mutation_IMEx_IntAct_clean.tsv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "four_classes_mutation" / "alpha_missense_data"),
        help="Output directory for PDB files",
    )
    parser.add_argument("--workers", type=int, default=16, help="Parallel download workers")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of IDs (0 = no limit)")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    upids = read_target_upids(args.tsv)
    if args.limit and args.limit > 0:
        upids = upids[: args.limit]

    total = len(upids)
    if total == 0:
        print("No Target_UPID entries found.")
        return 1

    print(f"Total unique Target_UPID: {total}")

    counts = {"downloaded": 0, "exists": 0, "not_found": 0, "error": 0}
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, upid, args.out_dir, args.timeout): upid for upid in upids}
        done = 0
        for fut in as_completed(futures):
            status, upid = fut.result()
            counts[status] += 1
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start
                print(
                    f"Progress {done}/{total} | downloaded={counts['downloaded']} "
                    f"exists={counts['exists']} not_found={counts['not_found']} error={counts['error']} "
                    f"elapsed={elapsed:.1f}s"
                )

    print("Done.")
    print(
        f"downloaded={counts['downloaded']} exists={counts['exists']} "
        f"not_found={counts['not_found']} error={counts['error']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
