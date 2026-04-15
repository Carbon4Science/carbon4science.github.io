# LlaSMol - Retrosynthesis Prediction

## Model Info
- **Task**: Retrosynthesis (product -> reactants)
- **Architecture**: LLM (Mistral-7B, instruction-tuned with LoRA)
- **Year**: 2024
- **Venue**: COLM 2024
- **Parameters**: ~7B
- **Conda Environment**: `gpt`
- **Dataset**: SMolInstruct (includes retrosynthesis from USPTO)

## Reference
Yu et al., "LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset", COLM, 2024.
GitHub: https://github.com/OSU-NLP-Group/LLM4Chem

## Usage
```python
from Retro.LlaSMol.Inference import run
results = run("CCOC(C)=O", top_k=10)
```

## Notes
- Same model as Forward/LlaSMol but with retrosynthesis prompt
- Prompt: `<SMILES> {product} </SMILES> Based on the given product, suggest possible reactants...`
- Paper reports 32.9% top-1 EM on USPTO-full (not USPTO-50K)
- Requires ~16 GB+ GPU memory
