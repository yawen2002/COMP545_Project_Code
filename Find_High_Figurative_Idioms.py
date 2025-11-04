#!/usr/bin/env python3
import json
import os
import csv
import sys

def word_count(idiom: str) -> int:
    # split on whitespace, ignore empty parts
    return len([w for w in idiom.strip().split() if w])

def main():
    data_file = "MAGPIE_unfiltered.jsonl"
    output_csv = "high_figurative_idioms.csv"

    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found in current directory.")
        sys.exit(1)

    # Aggregate per idiom:
    # idiom_stats[idiom] = {
    #     "i": <count of label=='i'>,
    #     "l": <count of label=='l'>,
    #     "total_rows": <all rows for this idiom, any label>
    # }
    idiom_stats = {}

    # 1) scan file
    with open(data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                # skip bad lines
                print(f"Warning: could not parse JSON on line {line_num}: {e}")
                continue

            idiom = entry.get("idiom")
            label = entry.get("label")

            if not idiom:
                continue

            # filter: idiom must have >= 3 words
            if word_count(idiom) < 3:
                continue

            # init stats
            if idiom not in idiom_stats:
                idiom_stats[idiom] = {
                    "i": 0,
                    "l": 0,
                    "total_rows": 0,
                }

            idiom_stats[idiom]["total_rows"] += 1

            # count only i / l for later ratio
            if label == "i":
                idiom_stats[idiom]["i"] += 1
            elif label == "l":
                idiom_stats[idiom]["l"] += 1
            # else: label in {"f", "o", "?"} → we don't count toward i/l

    # 2) filter idioms that were never used literally
    # i.e. idiom_stats[idiom]["l"] == 0 → drop
    filtered_rows = []
    for idiom, stats in idiom_stats.items():
        idiomatic_count = stats["i"]
        literal_count = stats["l"]

        if literal_count == 0:
            continue

        denom = idiomatic_count + literal_count
        if denom == 0:
            # no usable i/l occurrences
            continue

        ratio_fraction = idiomatic_count / denom
        ratio_percent = ratio_fraction * 100.0

        filtered_rows.append({
            "idiom": idiom,
            "figurative_ratio_percent": ratio_percent,
            "idiomatic_count": idiomatic_count,
            "literal_count": literal_count,
            "total_i_l": denom,
            "total_rows_any_label": stats["total_rows"],
        })

    # 3) sort by figurative ratio descending
    filtered_rows.sort(
        key=lambda r: (r["figurative_ratio_percent"], r["total_i_l"]),
        reverse=True
    )

    # 4) write CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "idiom",
            "figurative_ratio_percent",
            "idiomatic_count",
            "literal_count",
            "total_i_l",
            "total_rows_any_label"
        ])
        for row in filtered_rows:
            writer.writerow([
                row["idiom"],
                f"{row['figurative_ratio_percent']:.2f}",
                row["idiomatic_count"],
                row["literal_count"],
                row["total_i_l"],
                row["total_rows_any_label"],
            ])

    print(f"Done. Wrote {len(filtered_rows)} idioms to {output_csv}.")

if __name__ == "__main__":
    main()
