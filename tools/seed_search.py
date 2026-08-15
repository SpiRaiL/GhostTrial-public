#!/usr/bin/env python3
"""Search the BONES-SEED annotations for motions worth retargeting.

Searches every free-text annotation column (4 natural descriptions, technical
description, 2 short descriptions, content/move name) plus the temporal event
labels, case-insensitively.

    python tools/seed_search.py uppercut
    python tools/seed_search.py "spell|magic|cast" --category Magic
    python tools/seed_search.py punch --mirrors --limit 50
"""

import argparse
import json
import pathlib

import pandas as pd

SEED = pathlib.Path(__file__).resolve().parent.parent / "retargeted" / "bones-seed"
META = SEED / "metadata" / "seed_metadata_v004.parquet"
TEMPORAL = SEED / "metadata" / "seed_metadata_v002_temporal_labels.jsonl"

TEXT_COLS = [
    "content_name",
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
    "move_name",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="regex, case-insensitive")
    ap.add_argument("--category", help="restrict to one category (e.g. Magic, Sports)")
    ap.add_argument("--mirrors", action="store_true", help="include mirrored duplicates")
    ap.add_argument("--limit", type=int, default=25, help="rows to print (default 25)")
    ap.add_argument("--temporal", action="store_true", help="also search the temporal event labels")
    ap.add_argument("--paths", action="store_true", help="print g1 csv paths instead of a summary table")
    args = ap.parse_args()

    df = pd.read_parquet(META)
    blob = df[TEXT_COLS].fillna("").agg(" | ".join, axis=1)

    hits = df[blob.str.contains(args.pattern, case=False, regex=True)]
    if not args.mirrors:
        hits = hits[~hits.is_mirror]
    if args.category:
        hits = hits[hits.category == args.category]

    print(f"{len(hits)} motions match {args.pattern!r}"
          f"{'' if args.mirrors else ' (mirrors excluded)'}\n")

    if args.paths:
        for p in hits.move_g1_path:
            print(p)
    elif len(hits):
        grouped = hits.groupby("content_name").agg(
            takes=("move_name", "size"),
            package=("package", "first"),
            category=("category", "first"),
            frames=("move_duration_frames", "median"),
            description=("content_short_description", "first"),
        ).sort_values("takes", ascending=False)
        with pd.option_context("display.width", 220, "display.max_colwidth", 90):
            print(grouped.head(args.limit).to_string())

    if args.temporal:
        print("\n--- temporal event labels ---")
        shown = 0
        with open(TEMPORAL) as fh:
            for line in fh:
                if shown >= args.limit:
                    break
                rec = json.loads(line)
                for ev in rec["events"]:
                    if pd.Series([ev["description"]]).str.contains(args.pattern, case=False, regex=True).item():
                        print(f"{rec['filename']}  [{ev['start_time']:.2f}-{ev['end_time']:.2f}s]  {ev['description']}")
                        shown += 1
                        break


if __name__ == "__main__":
    main()
