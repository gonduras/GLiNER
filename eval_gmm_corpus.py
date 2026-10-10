import argparse
import glob
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import torch
from gliner import GLiNER

LABELS = ["ART_STYLE", "LINEAGE", "ART_AXIS", "PIPELINE", "HARDWARE", "MODEL_ARTIFACT"]

def create_parser():
    parser = argparse.ArgumentParser(description="Evaluate GLiNER on GMM corpus")
    parser.add_argument("--ckpt", type=str, default="urchade/gliner_mediumv2.1", help="Path or HF ID of GLiNER model")
    parser.add_argument("--corpus", type=str, required=True, help="Path to folder containing markdown files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for regular entities")
    parser.add_argument("--out_file", type=str, default="eval_results.jsonl", help="Output JSONL file")
    return parser

def is_valid_model_artifact(text):
    # <style>_<arch>_<method>_r<rank>_v<run>_<date>
    pattern = r'^.+_.+_.+_r\d+_v\d+_(?:\d{8}|\d{4}-\d{2}-\d{2})$'
    return bool(re.match(pattern, text))

def main():
    parser = create_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.ckpt} on {device}...")
    model = GLiNER.from_pretrained(args.ckpt, load_tokenizer=True)
    model.to(device)

    markdown_files = glob.glob(os.path.join(args.corpus, "**/*.md"), recursive=True)
    if not markdown_files:
        print(f"No markdown files found in {args.corpus}")
        return

    # Statistics
    counts = defaultdict(int)
    low_confidence = []
    false_positives = []

    with open(args.out_file, "w", encoding="utf-8") as out_f:
        for filepath in markdown_files:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                continue

            # Use 0.3 threshold to catch low confidence entities as well
            entities = model.predict_entities(text, LABELS, threshold=0.3)

            for ent in entities:
                ent_text = ent["text"]
                label = ent["label"]
                score = ent["score"]

                # We categorize by score relative to args.threshold
                if score >= args.threshold:
                    counts[label] += 1

                    if label == "MODEL_ARTIFACT":
                        if not is_valid_model_artifact(ent_text):
                            false_positives.append({
                                "file": filepath,
                                "text": ent_text,
                                "score": score,
                                "reason": "Violates grammar <style>_<arch>_<method>_r<rank>_v<run>_<date>"
                            })
                else:
                    low_confidence.append({
                        "file": filepath,
                        "text": ent_text,
                        "label": label,
                        "score": score
                    })

                # Output per-file JSONL
                out_f.write(json.dumps({
                    "file": filepath,
                    "text": ent_text,
                    "label": label,
                    "span": [ent["start"], ent["end"]],
                    "score": score
                }) + "\n")

    print(f"\nExtraction complete. Results saved to {args.out_file}.")

    # Summary count table
    print("\n--- Summary Count Table (score >= {}) ---".format(args.threshold))
    for label in LABELS:
        print(f"{label}: {counts[label]}")

    # Low-confidence section
    print("\n--- Low-Confidence Entities (0.3 <= score < {}) ---".format(args.threshold))
    if not low_confidence:
        print("None found.")
    else:
        for item in low_confidence:
            print(f"[{item['label']}] {item['text']} (score: {item['score']:.3f}) in {item['file']}")

    # Hard-mode section: MODEL_ARTIFACT false positives
    print("\n--- HARD MODE: MODEL_ARTIFACT False Positives ---")
    if not false_positives:
        print("No false positives found! All high-confidence MODEL_ARTIFACTs match the grammar.")
    else:
        for item in false_positives:
            print(f"FAIL: '{item['text']}' (score: {item['score']:.3f}) in {item['file']} -> {item['reason']}")

if __name__ == "__main__":
    main()
