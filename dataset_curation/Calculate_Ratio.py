#!/usr/bin/env python3
import sys
import json
import os

def main():
    # Check command-line argument
    if len(sys.argv) != 2:
        print("Example Usage: python3 Calculate_Ratio.py \"on the same page\"")
        sys.exit(1)

    target_idiom = sys.argv[1].strip()

    # Dataset file (same directory)
    data_file = "MAGPIE_unfiltered.jsonl"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found in current directory.")
        sys.exit(1)

    # Counters
    total_entries_for_idiom = 0      # all rows where idiom == target
    idiomatic_count = 0              # rows where label == 'i'
    literal_count = 0                # rows where label == 'l'
    skipped_other_labels = 0         # rows for this idiom but label in {'f','o','?'} etc.

    # Read JSONL line by line
    with open(data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                # If some line is malformed, report and continue
                print(f"Warning: could not parse JSON on line {line_num}: {e}")
                continue

            # Expecting MAGPIE-style keys here:
            #   "idiom": str
            #   "label": one of {'i','l','f','o','?'}
            # If any of these keys are missing, we skip that line.
            idiom_value = entry.get("idiom")
            if idiom_value is None:
                # not in the shape we expect
                continue

            # Case-insensitive, whitespace-trimmed match
            if idiom_value.strip().lower() != target_idiom.lower():
                continue  # this row is for a different idiom

            # This entry is for the target idiom
            total_entries_for_idiom += 1

            label_value = entry.get("label")
            if label_value is None:
                # if label missing, just skip counting it
                skipped_other_labels += 1
                continue

            # Count only idiomatic ('i') and literal ('l')
            if label_value == "i":
                idiomatic_count += 1
            elif label_value == "l":
                literal_count += 1
            else:
                # labels like 'f' (false extraction), 'o' (other), '?' (unclear)
                skipped_other_labels += 1

    # After reading the file, compute ratio
    denominator = idiomatic_count + literal_count



    print("==============================================")
    print(f"Idiom: {target_idiom}")
    print(f"Total entries (this idiom) in MAGPIE_unfiltered.jsonl: {total_entries_for_idiom}")
    print(f"  - Idiomatic (label == 'i'): {idiomatic_count}")
    print(f"  - Literal   (label == 'l'): {literal_count}")
    print(f"  - Other / non i-l labels (for this idiom): {skipped_other_labels}")

    if denominator == 0:
        print("\nNo entries for this idiom had label 'i' or 'l'.")
        print("This can happen if all occurrences were marked as false extraction ('f'),")
        print("or 'other' ('o'), or 'unclear' ('?').")
        print("Figurative ratio: N/A")
    else:
        figurative_ratio = idiomatic_count / denominator
        ratio_percent = figurative_ratio * 100
        print(f"\nFigurative ratio = idiomatic_count / (idiomatic_count + literal_count)")
        print(f"                 = {idiomatic_count} / ({idiomatic_count} + {literal_count})")
        print(f"                 = {figurative_ratio:.6f}  (fraction)")
        print(f"                 = {ratio_percent:.2f}%  (percentage)")


    print("==============================================")

if __name__ == "__main__":
    main()

# To run the code: If you want to calculate the ratio for "on the same page" (replace this with the idiom you want), run the following command:
# python3 Calculate_Ratio.py "on the same page"