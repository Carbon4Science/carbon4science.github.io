# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chemformer is a pre-trained BART transformer for computational chemistry, using SMILES strings for molecular representation. It supports forward/backward synthesis prediction, retrosynthesis, molecular optimization, and property prediction.

## Environment

**Python**: 3.7.11

**Key Dependencies** (from `pyproject.toml`):
- torch: 1.8.1
- pytorch-lightning: 1.2.3
- rdkit: 2022.9.3
- hydra-core: ^1.3.2
- fastapi: 0.98.0
- pandas: 1.3.5
- pysmilesutils: git (https://github.com/MolecularAI/pysmilesutils.git)

**Optional Dependencies**:
- wandb: ^0.16.5 (experiment tracking)
- textbrewer: ^0.2.1.post1 (knowledge distillation)
- eco2ai: ^0.3.9 (carbon footprint tracking)

## Development Setup

```bash
conda env create -f env-dev.yml
conda activate chemformer
poetry install
pip install -e .  # for editable mode
```

If you encounter `GLIBCXX_3.4.21' not found` error:
```bash
export LD_LIBRARY_PATH=/path/to/conda/envs/chemformer/lib
```

## Common Commands

### Running Scripts
All scripts use Hydra for configuration. Run from repository root:

```bash
# Pre-training
python -m molbart.pretrain data_path=data/chembl.csv batch_size=128

# Fine-tuning
python -m molbart.fine_tune \
  data_path=data/uspto_50.pickle \
  model_path=models/pretrained.ckpt \
  task=backward_prediction

# Inference/scoring
python -m molbart.inference_score \
  data_path=data.csv \
  output_score_data=metrics.csv \
  output_sampled_smiles=sampled_smiles.json

# Prediction
python -m molbart.predict model_path=model.ckpt data_path=test.csv
```

Override config parameters via command line: `param=value` or `param.subparam=value`

### Running Tests
```bash
pytest tests/
pytest tests/test_tokenizer.py  # single test file
pytest tests/test_tokenizer.py::test_function_name  # single test
```

### Code Formatting
```bash
black .
```

### FastAPI Service
```bash
cd service
export CHEMFORMER_MODEL=path/to/model.ckpt
export CHEMFORMER_VOCAB=path/to/vocab.json
export CHEMFORMER_TASK=backward_prediction
python chemformer_service.py
```

## Architecture

### Models (`molbart/models/`)
- `base_transformer.py`: `_AbsTransformerModel` - PyTorch Lightning base class with embedding layer, positional encodings, LR scheduling, and training loops
- `transformer_models.py`: `BARTModel` (encoder-decoder) and `UnifiedModel` (encoder-only) implementations with pre-norm transformer layers
- `chemformer.py`: High-level `Chemformer` wrapper class for training, inference, and scoring
- `util.py`: `PreNormEncoderLayer` and `PreNormDecoderLayer` - pre-norm style layer normalization

**Default model config**: d_model=512, num_layers=6, num_heads=8, d_feedforward=2048, ~45M parameters

### Data (`molbart/data/`)
- `base.py`: `_AbsDataset` (PyTorch Dataset) and `_AbsDataModule` (Lightning DataModule) base classes
- `datamodules.py`: Task-specific datamodules
- `seq2seq_data.py`: `SynthesisDataModule`, `Uspto50DataModule` for reaction prediction
- `mol_data.py`: `MoleculeDataModule` for pre-training on ChEMBL/ZINC
- `TokenSampler`: Buckets sequences by length for efficient batching

### Tokenization (`molbart/utils/tokenizers/`)
- `ChemformerTokenizer`: SMILES tokenization using `pysmilesutils`
- `SpanTokensMasker`, `ReplaceTokensMasker`: Masking strategies for pre-training

### Decoding (`molbart/utils/samplers/`)
- `BeamSearchSampler`: GPU-optimized batch beam search
- `GreedyDecoder`: Single best path decoding

### Configuration (`molbart/config/`)
Hydra YAML configs for each script. Key files:
- `pretrain.yaml`, `fine_tune.yaml`, `predict.yaml`, `inference_score.yaml`

### Key Patterns
- Models use PyTorch Lightning for training loop abstraction
- Callbacks and scorers are dynamically loaded via config
- Custom datamodules/callbacks/scorers can be specified with full module paths in config

## Checkpoint Migration

Old checkpoints need vocabulary key rename:
```python
model = torch.load("model.ckpt")
model["hyper_parameters"]["vocabulary_size"] = model["hyper_parameters"].pop("vocab_size")
torch.save(model, "model_v2.ckpt")
```

## Vocabulary Files
- `bart_vocab.json`: Pre-training vocabulary (523 tokens)
- `bart_vocab_downstream.json`: Fine-tuning vocabulary
