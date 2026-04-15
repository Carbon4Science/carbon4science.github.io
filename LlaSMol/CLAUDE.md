# LlaSMol - Forward Reaction Prediction

## Model Info
- **Task**: Forward reaction prediction (reactants -> product)
- **Architecture**: LLM (Mistral-7B, instruction-tuned with LoRA)
- **Year**: 2024
- **Venue**: COLM 2024
- **Parameters**: ~7B
- **Conda Environment**: `llasmol`
- **Dataset**: SMolInstruct (includes forward synthesis from USPTO)

## Reference
Yu et al., "LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset", COLM, 2024.
GitHub: https://github.com/OSU-NLP-Group/LLM4Chem

## Setup
```bash
# Model is hosted on HuggingFace - downloads automatically on first use
# Available models:
#   osunlp/LlaSMol-Mistral-7B (recommended)
#   osunlp/LlaSMol-CodeLlama-7B
#   osunlp/LlaSMol-Llama2-7B
#   osunlp/LlaSMol-Galactica-6.7B
```

## Dependencies
- Python 3.10+, transformers, accelerate, torch, bitsandbytes, PEFT, RDKit

## Usage
```python
from Forward.LlaSMol.Inference import run
results = run("CCO.CC(=O)Cl", top_k=5)
```

## Notes
- Instruction-tuned LLM for chemistry tasks
- Uses natural language prompts with SMILES in XML tags
- Prompt: `<SMILES> {reactants} </SMILES> Based on the reactants and reagents given above, suggest a possible product.`
- Beam search with num_beams=8 for top-k predictions
- Requires ~16 GB+ GPU memory
- HuggingFace model: `osunlp/LlaSMol-Mistral-7B`
