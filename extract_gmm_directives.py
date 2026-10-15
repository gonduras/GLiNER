import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

from gliner import GLiNER

LABELS = [
    "ARTIST_INTENT",
    "ARTISTIC_MATERIAL",
    "COLOR_CONCEPT",
    "SPATIAL_RELATION",
    "HARDWARE_TARGET",
    "AI_TECHNIQUE",
    "LINEAGE_TAG",
    "POLICY_DIRECTIVE"
]

def load_jsonl(filepath: str | Path) -> List[Dict[str, Any]]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], filepath: str | Path):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_batch(model: GLiNER, texts: List[str], labels: List[str], batch_size: int = 8) -> List[List[Dict[str, Any]]]:
    all_entities = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_entities = model.batch_predict_entities(batch_texts, labels)
        all_entities.extend(batch_entities)
    return all_entities

def extract_entities(input_file: str, output_file: str, model_id: str = "urchade/gliner_base", batch_size: int = 8):
    print(f"Loading GLiNER model: {model_id}")
    model = GLiNER.from_pretrained(model_id)

    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)
    texts = [item.get("text", "") for item in data]

    print(f"Processing {len(texts)} records in batches of {batch_size}")
    all_entities = process_batch(model, texts, LABELS, batch_size)

    for item, entities in zip(data, all_entities):
        item["entities"] = entities

    print(f"Saving enriched data to: {output_file}")
    save_jsonl(data, output_file)

def evaluate(model_id: str = "urchade/gliner_base"):
    print(f"Loading GLiNER model for evaluation: {model_id}")
    model = GLiNER.from_pretrained(model_id)

    # 50 mock test sentences
    eval_data = [
        {"text": "Create a gondurastration using yellow paper cutout radiating wedges from left and right.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "yellow paper cutout"}, {"label": "LINEAGE_TAG", "text": "gondurastration"}, {"label": "SPATIAL_RELATION", "text": "radiating wedges from left and right"}]},
        {"text": "I want an uplifting feeling of loving yourself in sky blue on RTX 3060 12GB using SPADE.", "annotations": [{"label": "ARTIST_INTENT", "text": "uplifting feeling of loving yourself"}, {"label": "COLOR_CONCEPT", "text": "sky blue"}, {"label": "HARDWARE_TARGET", "text": "RTX 3060 12GB"}, {"label": "AI_TECHNIQUE", "text": "SPADE"}]},
        {"text": "Litho ink and scotch tape arranged center-centered.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "Litho ink"}, {"label": "ARTISTIC_MATERIAL", "text": "scotch tape"}, {"label": "SPATIAL_RELATION", "text": "center-centered"}]},
        {"text": "StyleGAN3 generation on 4x RTX 4090 with zero-overlap policy.", "annotations": [{"label": "AI_TECHNIQUE", "text": "StyleGAN3"}, {"label": "HARDWARE_TARGET", "text": "4x RTX 4090"}, {"label": "POLICY_DIRECTIVE", "text": "zero-overlap"}]},
        {"text": "Save the gonduras-model in the sacred vault read-only.", "annotations": [{"label": "LINEAGE_TAG", "text": "gonduras-model"}, {"label": "POLICY_DIRECTIVE", "text": "sacred vault read-only"}]},
        {"text": "Apply LoRA rank 32 for the HE HER series using deep blue hues.", "annotations": [{"label": "AI_TECHNIQUE", "text": "LoRA rank 32"}, {"label": "LINEAGE_TAG", "text": "HE HER series"}, {"label": "COLOR_CONCEPT", "text": "deep blue"}]},
        {"text": "The Oklab gamut must be strictly enforced across the VRAM budget.", "annotations": [{"label": "COLOR_CONCEPT", "text": "Oklab gamut"}, {"label": "HARDWARE_TARGET", "text": "VRAM budget"}]},
        {"text": "Putting yourself on pedestal using OpenUSD.", "annotations": [{"label": "ARTIST_INTENT", "text": "Putting yourself on pedestal"}, {"label": "AI_TECHNIQUE", "text": "OpenUSD"}]},
        {"text": "A newsprint collage rendered on RTX 3060 12GB.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "newsprint"}, {"label": "HARDWARE_TARGET", "text": "RTX 3060 12GB"}]},
        {"text": "Radiating wedges from left and right in голубой.", "annotations": [{"label": "SPATIAL_RELATION", "text": "Radiating wedges from left and right"}, {"label": "COLOR_CONCEPT", "text": "голубой"}]},
        {"text": "Uplifting feeling of loving yourself expressed through gondurastration.", "annotations": [{"label": "ARTIST_INTENT", "text": "Uplifting feeling of loving yourself"}, {"label": "LINEAGE_TAG", "text": "gondurastration"}]},
        {"text": "Zero-overlap allowed when applying scotch tape.", "annotations": [{"label": "POLICY_DIRECTIVE", "text": "Zero-overlap"}, {"label": "ARTISTIC_MATERIAL", "text": "scotch tape"}]},
        {"text": "Center-centered composition in Oklab gamut.", "annotations": [{"label": "SPATIAL_RELATION", "text": "Center-centered"}, {"label": "COLOR_CONCEPT", "text": "Oklab gamut"}]},
        {"text": "Sacred vault read-only access for the gonduras-model.", "annotations": [{"label": "POLICY_DIRECTIVE", "text": "Sacred vault read-only"}, {"label": "LINEAGE_TAG", "text": "gonduras-model"}]},
        {"text": "Run SPADE within the strict VRAM budget.", "annotations": [{"label": "AI_TECHNIQUE", "text": "SPADE"}, {"label": "HARDWARE_TARGET", "text": "VRAM budget"}]},
        {"text": "The HE HER series utilizes litho ink.", "annotations": [{"label": "LINEAGE_TAG", "text": "HE HER series"}, {"label": "ARTISTIC_MATERIAL", "text": "litho ink"}]},
        {"text": "Use deep blue (синий) for a gondurastration.", "annotations": [{"label": "COLOR_CONCEPT", "text": "deep blue"}, {"label": "COLOR_CONCEPT", "text": "синий"}, {"label": "LINEAGE_TAG", "text": "gondurastration"}]},
        {"text": "Applying yellow paper cutout over the newsprint.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "yellow paper cutout"}, {"label": "ARTISTIC_MATERIAL", "text": "newsprint"}]},
        {"text": "Deploying StyleGAN3 on RTX 3060 12GB.", "annotations": [{"label": "AI_TECHNIQUE", "text": "StyleGAN3"}, {"label": "HARDWARE_TARGET", "text": "RTX 3060 12GB"}]},
        {"text": "Ensure the uplifting feeling of loving yourself is clear.", "annotations": [{"label": "ARTIST_INTENT", "text": "uplifting feeling of loving yourself"}]},
        {"text": "LoRA rank 32 training in the sacred vault read-only mode.", "annotations": [{"label": "AI_TECHNIQUE", "text": "LoRA rank 32"}, {"label": "POLICY_DIRECTIVE", "text": "sacred vault read-only"}]},
        {"text": "Putting yourself on pedestal requires 4x RTX 4090.", "annotations": [{"label": "ARTIST_INTENT", "text": "Putting yourself on pedestal"}, {"label": "HARDWARE_TARGET", "text": "4x RTX 4090"}]},
        {"text": "The composition should be center-centered with zero-overlap.", "annotations": [{"label": "SPATIAL_RELATION", "text": "center-centered"}, {"label": "POLICY_DIRECTIVE", "text": "zero-overlap"}]},
        {"text": "Export to OpenUSD format for the gonduras-model.", "annotations": [{"label": "AI_TECHNIQUE", "text": "OpenUSD"}, {"label": "LINEAGE_TAG", "text": "gonduras-model"}]},
        {"text": "Use scotch tape and yellow paper cutout for the collage.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "scotch tape"}, {"label": "ARTISTIC_MATERIAL", "text": "yellow paper cutout"}]},
        {"text": "Sky blue gradients using SPADE.", "annotations": [{"label": "COLOR_CONCEPT", "text": "Sky blue"}, {"label": "AI_TECHNIQUE", "text": "SPADE"}]},
        {"text": "A gondurastration rendered on 4x RTX 4090.", "annotations": [{"label": "LINEAGE_TAG", "text": "gondurastration"}, {"label": "HARDWARE_TARGET", "text": "4x RTX 4090"}]},
        {"text": "Maintain zero-overlap for all radiating wedges from left and right.", "annotations": [{"label": "POLICY_DIRECTIVE", "text": "zero-overlap"}, {"label": "SPATIAL_RELATION", "text": "radiating wedges from left and right"}]},
        {"text": "Litho ink patterns in Oklab gamut.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "Litho ink"}, {"label": "COLOR_CONCEPT", "text": "Oklab gamut"}]},
        {"text": "HE HER series produced using StyleGAN3.", "annotations": [{"label": "LINEAGE_TAG", "text": "HE HER series"}, {"label": "AI_TECHNIQUE", "text": "StyleGAN3"}]},
        {"text": "Strictly sacred vault read-only for newsprint archives.", "annotations": [{"label": "POLICY_DIRECTIVE", "text": "sacred vault read-only"}, {"label": "ARTISTIC_MATERIAL", "text": "newsprint"}]},
        {"text": "Uplifting feeling of loving yourself captured in sky blue.", "annotations": [{"label": "ARTIST_INTENT", "text": "Uplifting feeling of loving yourself"}, {"label": "COLOR_CONCEPT", "text": "sky blue"}]},
        {"text": "Center-centered design using OpenUSD.", "annotations": [{"label": "SPATIAL_RELATION", "text": "Center-centered"}, {"label": "AI_TECHNIQUE", "text": "OpenUSD"}]},
        {"text": "VRAM budget limits LoRA rank 32 training.", "annotations": [{"label": "HARDWARE_TARGET", "text": "VRAM budget"}, {"label": "AI_TECHNIQUE", "text": "LoRA rank 32"}]},
        {"text": "Putting yourself on pedestal in deep blue.", "annotations": [{"label": "ARTIST_INTENT", "text": "Putting yourself on pedestal"}, {"label": "COLOR_CONCEPT", "text": "deep blue"}]},
        {"text": "Applying scotch tape over the litho ink.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "scotch tape"}, {"label": "ARTISTIC_MATERIAL", "text": "litho ink"}]},
        {"text": "gonduras-model requires 4x RTX 4090.", "annotations": [{"label": "LINEAGE_TAG", "text": "gonduras-model"}, {"label": "HARDWARE_TARGET", "text": "4x RTX 4090"}]},
        {"text": "Zero-overlap rules in the Oklab gamut.", "annotations": [{"label": "POLICY_DIRECTIVE", "text": "Zero-overlap"}, {"label": "COLOR_CONCEPT", "text": "Oklab gamut"}]},
        {"text": "HE HER series using SPADE on RTX 3060 12GB.", "annotations": [{"label": "LINEAGE_TAG", "text": "HE HER series"}, {"label": "AI_TECHNIQUE", "text": "SPADE"}, {"label": "HARDWARE_TARGET", "text": "RTX 3060 12GB"}]},
        {"text": "Radiating wedges from left and right in newsprint.", "annotations": [{"label": "SPATIAL_RELATION", "text": "Radiating wedges from left and right"}, {"label": "ARTISTIC_MATERIAL", "text": "newsprint"}]},
        {"text": "Uplifting feeling of loving yourself via LoRA rank 32.", "annotations": [{"label": "ARTIST_INTENT", "text": "Uplifting feeling of loving yourself"}, {"label": "AI_TECHNIQUE", "text": "LoRA rank 32"}]},
        {"text": "Sacred vault read-only policy for OpenUSD files.", "annotations": [{"label": "POLICY_DIRECTIVE", "text": "Sacred vault read-only"}, {"label": "AI_TECHNIQUE", "text": "OpenUSD"}]},
        {"text": "Yellow paper cutout arranged center-centered.", "annotations": [{"label": "ARTISTIC_MATERIAL", "text": "Yellow paper cutout"}, {"label": "SPATIAL_RELATION", "text": "center-centered"}]},
        {"text": "gondurastration rendered in sky blue.", "annotations": [{"label": "LINEAGE_TAG", "text": "gondurastration"}, {"label": "COLOR_CONCEPT", "text": "sky blue"}]},
        {"text": "VRAM budget constraints on StyleGAN3.", "annotations": [{"label": "HARDWARE_TARGET", "text": "VRAM budget"}, {"label": "AI_TECHNIQUE", "text": "StyleGAN3"}]},
        {"text": "Putting yourself on pedestal using litho ink.", "annotations": [{"label": "ARTIST_INTENT", "text": "Putting yourself on pedestal"}, {"label": "ARTISTIC_MATERIAL", "text": "litho ink"}]},
        {"text": "Zero-overlap for scotch tape layers.", "annotations": [{"label": "POLICY_DIRECTIVE", "text": "Zero-overlap"}, {"label": "ARTISTIC_MATERIAL", "text": "scotch tape"}]},
        {"text": "Deep blue hues in the HE HER series.", "annotations": [{"label": "COLOR_CONCEPT", "text": "Deep blue"}, {"label": "LINEAGE_TAG", "text": "HE HER series"}]},
        {"text": "Radiating wedges from left and right on 4x RTX 4090.", "annotations": [{"label": "SPATIAL_RELATION", "text": "Radiating wedges from left and right"}, {"label": "HARDWARE_TARGET", "text": "4x RTX 4090"}]},
        {"text": "gonduras-model adhering to Oklab gamut.", "annotations": [{"label": "LINEAGE_TAG", "text": "gonduras-model"}, {"label": "COLOR_CONCEPT", "text": "Oklab gamut"}]}
    ]

    texts = [item["text"] for item in eval_data]
    print(f"Predicting on {len(texts)} test sentences...")
    predictions = model.batch_predict_entities(texts, LABELS)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, (pred_entities, ground_truth) in enumerate(zip(predictions, eval_data)):
        pred_set = set([(ent["text"].lower(), ent["label"]) for ent in pred_entities])
        gt_set = set([(ent["text"].lower(), ent["label"]) for ent in ground_truth["annotations"]])

        true_positives += len(pred_set.intersection(gt_set))
        false_positives += len(pred_set - gt_set)
        false_negatives += len(gt_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GMM directives using GLiNER")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="urchade/gliner_base", help="GLiNER model ID")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on mock test set")

    args, unknown = parser.parse_known_args()
    if args.eval:
        evaluate(args.model)
    elif args.input and args.output:
        extract_entities(args.input, args.output, args.model, args.batch_size)
    else:
        parser.print_help()
