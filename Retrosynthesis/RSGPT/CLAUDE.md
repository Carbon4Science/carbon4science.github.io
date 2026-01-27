# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSGPT is a generative pretrained transformer foundation model for template-free retrosynthesis planning. It uses a LlamaForCausalLM architecture (24 layers, 2048 hidden size, ~1.6B parameters) pre-trained on 10 billion synthetic reaction data generated using RDChiral template extraction.

## Commands

### Installation
```bash
conda env create -f environment.yml
```

### Training
```bash
# Multi-GPU training with DeepSpeed (8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29500 --config_file deepspeed.yaml train.py
```

### Inference
```bash
# Single molecule inference (modify paths in infer.py)
python infer.py

# USPTO dataset testing (50k, full, or mit)
python test.py
```

### Utility
```bash
# Count model parameters (uses param_eval_env)
source param_eval_env/bin/activate && python count_parameters.py
```

## Evaluation Environment

A lightweight virtual environment `param_eval_env/` exists for model evaluation tasks (e.g., parameter counting) without the full conda environment:

```bash
# Already created with Python 3.9
# Contains: torch, transformers (minimal dependencies)
source param_eval_env/bin/activate
```

To recreate if needed:
```bash
/Users/admin/opt/anaconda3/bin/python3.9 -m venv param_eval_env
source param_eval_env/bin/activate
pip install torch transformers
```

## Architecture

### Core Components

- **models/rxngpt.py**: Main model class `RxnGPT` wrapping HuggingFace `LlamaForCausalLM`. Key methods: `forward()` for training, `infer()` for inference, `beam_search_gpt()` for generation.

- **task/task.py**: Orchestrates model, dataset, optimizer setup. Handles LoRA fine-tuning via PEFT library and optional EMA.

- **task/trainer.py**: Training loop with Accelerate/DeepSpeed integration, Wandb logging, gradient clipping, checkpoint saving.

- **datasets/rxngpt_dataset.py**: LMDB-based dataset loader with pre-tokenized reactions. Collator pads to max 260 tokens.

- **tokenizer/tokenization.py**: SMILES BPE tokenizer (vocab size 1000, max length 100).

### Reaction Format

Reactions are encoded as: `<s><Isyn><O>PRODUCT<F1>REACTANT1<F2>REACTANT2...</s>`

Special tokens defined in `utils/vocab.py`: `<pad>`, `<s>`, `</s>`, `<unk>`, `[MASK]`, plus chemical elements.

### Configuration

- **configs/base.yml**: Training hyperparameters (AdamW, lr=0.0001, epochs=10, batch sizes)
- **configs/rxngpt_llama1B.json**: Llama model architecture config
- **deepspeed.yaml**: DeepSpeed settings (Zero-2, bf16, 8 GPUs)

### Key Dependencies

- PyTorch 2.1.0, Transformers 4.42.3, Accelerate 0.32.1, DeepSpeed 0.14.0
- RDKit 2022.9.5, RDChiral 1.1.0 (chemistry)
- LMDB (dataset storage), PEFT 0.5.0 (LoRA), Wandb (experiment tracking)

## Data Availability

- USPTO datasets & model weights: https://sandbox.zenodo.org/records/203391
- Synthetic 10B reaction data: https://sandbox.zenodo.org/records/213324
