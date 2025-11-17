# Semantic Parser (BiLSTM-CRF + Relation Head)

A minimal, **trainable** semantic parsing skeleton aligned with your plan:
- **BIO sequence labeling** (entities/attributes) using **BiLSTM-CRF**
- **Relation classification** head for simple spatial/relational links
- Data format compatible with UD-style tokens and pseudo-labels for a quick start

> This is a teaching/research scaffold — concise, readable, and easy to extend.

## Quick Start

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate   # (Linux/Mac)
# or: .venv\Scripts\activate                       # (Windows PowerShell)
pip install -r requirements.txt
```

### 2) Train (toy data)
```bash
python train.py --data_dir data/sample --epochs 2
```

### 3) Inference (demo)
```bash
python infer.py --text "find the red cup on the table to the left of the bottle"
```

The output JSON includes:
- `entities`: list of spans with surface text
- `attributes`: basic attributes (toy)
- `relations`: head–rel–tail triples

## Files

```
semantic_parser/
├─ models/
│  └─ bilstm_crf.py        # BiLSTM-CRF + relation head (toy)
├─ utils/
│  ├─ data_loader.py       # Tiny CONLL-U-ish loader + batch collation
│  ├─ metrics.py           # BIO F1 (toy), relation acc
│  └─ vocab.py             # Minimal vocab for tokens & labels
├─ data/
│  └─ sample/
│     ├─ train.conllu      # Tiny toy dataset
│     ├─ dev.conllu
│     └─ test.conllu
├─ config.yaml             # Hyperparameters
├─ train.py                # Training loop (toy)
├─ infer.py                # Inference pipeline (rule-lite postprocessing)
└─ requirements.txt
```

## Notes
- This repo intentionally keeps the dataset tiny and synthetic so it runs fast on CPU.
- Replace `data/sample/*.conllu` with your UD English EWT splits and/or your caption-derived labels.
- Upgrade the embedding to a pretrained subword model (e.g., DistilBERT) for better performance.
- Wire the JSON output to your visual grounding module for end-to-end evaluation.
