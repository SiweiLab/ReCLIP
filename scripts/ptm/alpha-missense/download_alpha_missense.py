#!/usr/bin/env python3
"""Download AlphaMissense PDB files for PTM/IntAct mutation tables."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Sequence, Set, Tuple


URL_TEMPLATE = "https://alphamissense.hegelab.org/pdb/AF-{upid}-F1-AM_v4.pdb"
UPID_CANDIDATES = ("Target_UPID", "Uniprot")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))


def _collect_input_files(path_or_glob: str) -> List[str]:
    if os.path.isdir(path_or_glob):
        # Prefer PTM fold test files when present.
        test_files = sorted(glob.glob(os.path.join(path_or_glob, "fold_*_test.csv")))
        if test_files:
            return test_files
        files = sorted(glob.glob(os.path.join(path_or_glob, "*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_glob, "*.tsv")))
        if not files:
            raise FileNotFoundError(f"No CSV/TSV files found in directory: {path_or_glob}")
        return files

    glob_hits = sorted(glob.glob(path_or_glob))
    if glob_hits:
        return glob_hits

    if os.path.exists(path_or_glob):
        return [path_or_glob]
    raise FileNotFoundError(f"Input path not found: {path_or_glob}")


def _detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    if first_line.count("\t") > first_line.count(","):
        return "\t"
    return ","


def _pick_upid_col(fieldnames: Sequence[str], forced_col: str | None) -> str:
    if forced_col:
        if forced_col not in fieldnames:
            raise ValueError(f"Specified column '{forced_col}' not found in header: {fieldnames}")
        return forced_col
    for c in UPID_CANDIDATES:
        if c in fieldnames:
            return c
    raise ValueError(f"No UniProt column found, expected one of {UPID_CANDIDATES}, got: {fieldnames}")


def read_upids(paths: Iterable[str], upid_col: str | None) -> List[str]:
    upids: Set[str] = set()
    for p in paths:
        delimiter = _detect_delimiter(p)
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if not reader.fieldnames:
                continue
            col = _pick_upid_col(reader.fieldnames, upid_col)
            count = 0
            for row in reader:
                val = str(row.get(col, "")).strip()
                if not val:
                    continue
                upids.add(val)
                count += 1
            print(f"Loaded {count} rows from {p} (column={col})")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AlphaMissense PDBs for mutation rows.")
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "ten_folds"),
        help="Input file/dir/glob for mutation table(s). Default: PTM ten_folds directory.",
    )
    parser.add_argument(
        "--upid-col",
        default=None,
        help="Optional explicit UniProt column name (e.g. Target_UPID or Uniprot). Auto-detected by default.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "alpha_missense_data"),
        help="Output directory for PDB files.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Parallel download workers.")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of IDs (0 = no limit).")
    args = parser.parse_args()

    input_files = _collect_input_files(args.input)
    print(f"Input files: {len(input_files)}")
    for p in input_files[:10]:
        print(f"  - {p}")
    if len(input_files) > 10:
        print(f"  ... ({len(input_files) - 10} more)")

    os.makedirs(args.out_dir, exist_ok=True)
    upids = read_upids(input_files, args.upid_col)
    if args.limit and args.limit > 0:
        upids = upids[: args.limit]

    total = len(upids)
    if total == 0:
        print("No UniProt IDs found.")
        return 1

    print(f"Total unique UniProt IDs: {total}")
    counts = {"downloaded": 0, "exists": 0, "not_found": 0, "error": 0}
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, upid, args.out_dir, args.timeout): upid for upid in upids}
        done = 0
        for fut in as_completed(futures):
            status, _ = fut.result()
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
