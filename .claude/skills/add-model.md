# Skill: Add New Model

Guide for adding a new model to the benchmark.

## Usage
```
/add-model [task] [model_name]
```

## Examples
```
/add-model Retrosynthesis MyNewModel
/add-model MolGen VAE
```

## Instructions

When the user invokes this skill, guide them through these steps:

### Step 1: Create Model Directory
```
<Task>/<ModelName>/
├── Inference.py        # Required: Uniform interface
├── environment.yml     # Required: Conda environment
├── README.md           # Recommended: Documentation
└── models/             # Model checkpoints
```

### Step 2: Implement Uniform Inference Interface

For **Retrosynthesis**:
```python
# Inference.py
from typing import List, Dict, Union

def run(smiles: Union[str, List[str]], top_k: int = 10) -> List[Dict]:
    """
    Args:
        smiles: Product SMILES string(s)
        top_k: Number of predictions per input

    Returns:
        [{'input': 'CCO', 'predictions': [{'smiles': '...', 'score': 0.95}, ...]}]
    """
    # Implementation here
    pass
```

For **MolGen**:
```python
# Inference.py
from typing import List

def run(num_samples: int = 100) -> List[str]:
    """
    Args:
        num_samples: Number of molecules to generate

    Returns:
        List of generated SMILES strings
    """
    pass
```

### Step 3: Create Conda Environment
```yaml
# environment.yml
name: mymodel
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - pytorch
  - rdkit
  - pip:
    - your-package
```

### Step 4: Register Model in Benchmark Runner

Update `benchmarks/run_benchmark.py`:
```python
TASKS = {
    "Retrosynthesis": {
        "models": {
            # Add your model here
            "MyNewModel": "Retrosynthesis.MyNewModel.Inference",
        }
    },
}
```

### Step 5: Update Setup Script

Add to `benchmarks/setup_envs.sh`:
```bash
setup_mymodel() {
    echo "Setting up MyNewModel environment..."
    cd ../Retrosynthesis/MyNewModel
    conda env create -f environment.yml
    cd -
    echo "✓ MyNewModel environment ready"
}
```

### Step 6: Test the Model
```bash
# Setup environment
./benchmarks/setup_envs.sh MyNewModel

# Test inference
python benchmarks/run_benchmark.py --task Retrosynthesis --model MyNewModel --limit 10
```

## Checklist
- [ ] `Inference.py` with uniform interface
- [ ] `environment.yml` with dependencies
- [ ] Model registered in `run_benchmark.py`
- [ ] Setup function in `setup_envs.sh`
- [ ] Test passes with `--limit 10`
