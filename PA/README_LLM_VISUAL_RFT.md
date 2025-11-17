# LL M‑Visual‑RFT

Hybrid vision‑language object detection using LLM‑driven semantic parsing and visual reinforcement fine‑tuning.

## Overview

LLM‑Visual‑RFT combines DeepSeek‑R1 for structured query decomposition with Visual‑RFT for IoU‑guided bounding‑box prediction.  
- **Composite Entity F1:** 97.3% (↑2.6 pp over Visual‑RFT)  
- **Multi‑Entity F1:** 98.2% (↑34.0 pp over Visual‑RFT)  
- **Attribute Matching Accuracy:** 0.98  

## Features

- **Semantic Parsing** via DeepSeek‑R1  
- **Reinforcement Fine‑Tuning** with GRPO & IoU rewards  
- **High Attribute Consistency** for color, position, etc.  

## Environment
```bash
conda create -n LLM-Visual-RFT python=3.10
conda activate LLM-Visual-RFT
```
All setup steps are automated via a Bash script. Ensure you have **conda** installed and then run:

```bash
# Run the environment setup script
bash setup.sh
