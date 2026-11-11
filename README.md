# This repository provides the reference implementation of "DAVIN: Fine-Grained Preference-Attribute Matching on Multi-Modal Data for Visually-Aware Conversational Recommendation" that is currently submitted to ICDE 2027.

# DAVIN: Fine-Grained Preference-Attribute Matching on Multi-Modal Data for Visually-Aware Conversational Recommendation

## Overview

- **M1. Preprocessing.** GPT extracts the user's preferences from each dialogue, and the attributes of each item from its title and image.
- **M2. Multimodal encoding.** A dialogue encoder embeds the dialogue and each preference. An item encoder embeds each item (title and image) and each attribute. Both start from Qwen3-VL-Embedding-8B.
- **M3. Multi-granular matching.**
  - *Coarse-grained:* the similarity between the whole dialogue and the whole item.
  - *Fine-grained:* the similarity between each preference and each attribute, weighted by how relevant each pair is, as learned by a joint attention component.
  - The final retrieval score is a learned mix of the two.
- **M4. Training.** A contrastive loss pulls the dialogue toward its target item and away from hard negatives. Only the dialogue encoder (via LoRA), the joint attention component, and the mixing weight are trained. The item encoder is frozen, so item and attribute embeddings are computed only once.

## Setup

We use the following environment:

- Python 3.10
- PyTorch 2.5.1
- Transformers 5.8.0
- PEFT 0.19.1
- bitsandbytes 0.49.2
- Accelerate 1.13.0
- OpenAI 2.37.0

Install the required libraries with:

```bash
pip install -r requirements.txt
```

## Data

We use three domains of the Reddit–Amazon dataset released with [LaViC](https://github.com/jeon185/LaViC): Home, Fashion, and Beauty. Download the dialogues, item metadata, and item images from LaViC, and arrange them as follows:

```
data/
├── amazon_home/       {train,valid,test}.jsonl
├── amazon_fashion/    {train,valid,test}.jsonl
├── all_beauty/        {train,valid,test}.jsonl
└── item/
    ├── metadata.json  main category and title of each item
    └── images/        representative image of each item
```

## Usage

Pick a domain (`home`, `fashion`, or `beauty`) and run the two steps below.

**1. Extract preferences and attributes (M1).**

```bash
OPENAI_API_KEY=... bash scripts/extract.sh home
```

This writes `data/amazon_home/{split}_preferences.json` and `data/item/amazon_home/attributes.json`.

**2. Train and evaluate (M2–M4).**

```bash
bash scripts/train.sh home
```

This writes `runs/amazon_home/`, which holds the checkpoint with the best validation performance and its test results (Hit@1, Recall@1, nDCG@10, MRR@10) in `test_metrics.json`. The default hyperparameters follow the paper.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). See [LICENSE](LICENSE) for details.
