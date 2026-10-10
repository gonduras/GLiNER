import json
import random
import os

# Define domain entities
entities = {
    "ART_STYLE": ["gondurastration"],
    "LINEAGE": ["GOND", "GONDWANA", "GRAD_ANIM", "ASTRON", "husk-model", "vlada-model", "gonduras-model"],
    "ART_AXIS": ["Material_Substrate", "Exposure_Fracture", "Structural_Glitch", "Graphic_Vector", "Zero_Plane", "Faceted_Decomposition"],
    "PIPELINE": ["ESRGAN", "SPADE", "ControlNet", "LoRA", "Frankenstein"],
    "HARDWARE": ["RTX 3060", "VRAM"],
}

# Synthetic MODEL_ARTIFACT generator
# <style>_<arch>_<method>_r<rank>_v<run>_<date>
styles = ["gondurastration", "abstract", "realism"]
archs = ["sdxl", "sd15", "flux"]
methods = ["lora", "dreambooth", "finetune"]
ranks = ["8", "16", "32"]
runs = ["1", "2", "3"]
dates = ["20231010", "20231111", "20240101"]

def generate_model_artifact():
    return f"{random.choice(styles)}_{random.choice(archs)}_{random.choice(methods)}_r{random.choice(ranks)}_v{random.choice(runs)}_{random.choice(dates)}"

# Add random model artifacts
entities["MODEL_ARTIFACT"] = [generate_model_artifact() for _ in range(10)]

# Templates for synthetic sentences
templates = [
    "The new {MODEL_ARTIFACT} runs smoothly on the {HARDWARE}.",
    "We applied the {ART_STYLE} style using {PIPELINE} over the {LINEAGE} lineage.",
    "Evaluating the {ART_AXIS} axis on {LINEAGE} models requires more {HARDWARE}.",
    "Our standard {PIPELINE} workflow includes {MODEL_ARTIFACT} for {ART_STYLE}.",
    "The {LINEAGE} weights were updated with {PIPELINE} targeting the {ART_AXIS} parameter.",
    "This {ART_STYLE} render was generated with {HARDWARE}.",
    "We need to test {MODEL_ARTIFACT} against the {ART_AXIS} tests.",
    "Running {PIPELINE} on {HARDWARE} can be bottlenecked by memory.",
    "The model {MODEL_ARTIFACT} inherits from {LINEAGE} and improves {ART_STYLE}.",
    "Adjusting the {ART_AXIS} in {MODEL_ARTIFACT} fixed the artifacts."
]

negative_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a standard text sentence with no specific entities.",
    "I had a great lunch today.",
    "The weather is very nice outside.",
    "She sells seashells by the seashore."
]

def generate_dataset(num_samples=100):
    dataset = []

    for _ in range(num_samples):
        if random.random() < 0.2:
            # Negative example
            sentence = random.choice(negative_sentences)
            tokenized_text = sentence.split()
            dataset.append({"tokenized_text": tokenized_text, "ner": []})
        else:
            # Positive example
            template = random.choice(templates)

            # Find which entity types are in the template
            used_entities = {}
            for entity_type in entities.keys():
                if f"{{{entity_type}}}" in template:
                    used_entities[entity_type] = random.choice(entities[entity_type])

            # Replace placeholders
            sentence = template
            for entity_type, entity_value in used_entities.items():
                sentence = sentence.replace(f"{{{entity_type}}}", entity_value)

            tokenized_text = sentence.split()
            ner = []

            # Find indices for NER
            for entity_type, entity_value in used_entities.items():
                entity_tokens = entity_value.split()

                # Simple subsequence search for exact token matches
                # We assume entity values don't span punctuation in a way that split() breaks them
                for i in range(len(tokenized_text) - len(entity_tokens) + 1):
                    # Clean tokens for matching (strip simple punctuation)
                    match = True
                    for j, e_token in enumerate(entity_tokens):
                        clean_text_token = tokenized_text[i+j].strip(".,;")
                        if clean_text_token != e_token:
                            match = False
                            break
                    if match:
                        ner.append([i, i + len(entity_tokens) - 1, entity_type])
                        break

            dataset.append({"tokenized_text": tokenized_text, "ner": ner})

    return dataset

def main():
    os.makedirs("data", exist_ok=True)
    dataset = generate_dataset(200)

    output_path = "data/gmm_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(dataset)} samples to {output_path}")

if __name__ == "__main__":
    main()
