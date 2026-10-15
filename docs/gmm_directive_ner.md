# GMM Collage Semiotics Entity Extraction

This document outlines the zero-shot/few-shot Named Entity Recognition (NER) pipeline used to extract structured training pairs from artist directives and semiotic statements by Alexander Gonduras Jitomirsky. The system relies on the GLiNER framework.

## Entity Types

The pipeline is configured to extract 8 domain-specific entity types:

- **`ARTIST_INTENT`**: Represents the emotional or conceptual goal of the artist.
  - *Examples*: "uplifting feeling of loving yourself", "putting yourself on pedestal"
- **`ARTISTIC_MATERIAL`**: Physical or digital materials used in the artwork.
  - *Examples*: "scotch tape", "newsprint", "litho ink", "yellow paper cutout"
- **`COLOR_CONCEPT`**: Specific color choices, palettes, or color spaces.
  - *Examples*: "голубой / sky blue", "синий / deep blue", "Oklab gamut"
- **`SPATIAL_RELATION`**: Compositional or layout instructions.
  - *Examples*: "pedestal", "center-centered", "radiating wedges from left and right"
- **`HARDWARE_TARGET`**: The hardware or computational limits applied to the process.
  - *Examples*: "RTX 3060 12GB", "4x RTX 4090", "VRAM budget"
- **`AI_TECHNIQUE`**: AI or algorithmic models and methods.
  - *Examples*: "SPADE", "StyleGAN3", "LoRA rank 32", "OpenUSD"
- **`LINEAGE_TAG`**: Series, style lineage, or naming conventions for models/artworks.
  - *Examples*: "gondurastration", "gonduras-model", "HE HER series"
- **`POLICY_DIRECTIVE`**: Rules or constraints imposed on the process or access.
  - *Examples*: "zero-overlap", "sacred vault read-only"

## Usage

The `extract_gmm_directives.py` script provides the primary interface for batch processing JSONL files containing text transcripts, logs, and council directives.

### Dependencies

Ensure you have the required dependencies installed (using the project's standard `uv` setup):

```bash
uv pip install -e .[dev]
uv pip install tiktoken protobuf
```

### Running Batch Extraction

To process a JSONL file (where each line is a JSON object with at least a `"text"` field) and output enriched records with entities:

```bash
python extract_gmm_directives.py --input input_data.jsonl --output enriched_data.jsonl
```

You can optionally override the default GLiNER model (`urchade/gliner_base`) and the batch size:

```bash
python extract_gmm_directives.py --input input.jsonl --output output.jsonl --model urchade/gliner_large-v2.1 --batch-size 16
```

### Running Evaluation

To verify the extraction pipeline against a 50-sentence mock benchmark dataset and output precision, recall, and F1 scores:

```bash
python extract_gmm_directives.py --eval
```
