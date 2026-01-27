# RSGPT: A Generative Transformer Foundation Model Pre-trained on Ten Billion Data for Retrosynthesis Planning


Available retrosynthesis data are limited to only millions. Therefore, we pioneering utilized the RDChiral reverse synthesis template extraction algorithm to generate chemical reaction data. This method precisely aligns an existing template’s reaction center with those of synthons, yielding a complete reaction. Consequently, **over 10 billion high-quality** reaction data entries were generated. A generative pretrained transformer (GPT) foundation model called RSGPT was subsequently developed for template-free retrosynthesis planning, by pre-training using the 10 billion generated reaction data. Inspired by the strategies of LLMs, we introduced reinforcement learning from AI feedback to capture the relationships among products, reactants, and templates more accurately. Extensive experiments demonstrate that our model achieves state-of-the-art performance on the USPTO-50K dataset, with a Top-1 accuracy of **63.4%**, substantially outperforming previous models. To the best of our knowledge, RSGPT is the pioneering GPT foundation model for retrosynthesis planning, providing groundbreaking insights and potential scalability across a wide range of chemical scenarios and applications.
## Table of Contents

- [Installation](#installation)
- [Quick Inference](#quick-inference)
- [Usage](#usage)
- [Model and Data Availability](#model-and-data-availability)
## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/jogjogee/RSGPT/

# Navigate to the project directory
cd yourproject

# Install dependencies
conda env create -f environment.yml
```

## Quick Inference

Download model weights from [Zenodo](https://sandbox.zenodo.org/records/203391) and place in `weights/finetune_50k.pth`.

**Method 1: Simple function call**
```python
from inference import run

result = run('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
print(result['predictions'])
```

**Method 2: Class-based (more control)**
```python
from inference import RSGPTPredictor

predictor = RSGPTPredictor(device='cuda:0')
result = predictor.run('CC(=O)Oc1ccccc1C(=O)O', beam_size=10)
print(result['valid_predictions'])
```

**Method 3: Command line**
```bash
python inference.py "CC(=O)Oc1ccccc1C(=O)O" --device cuda:0 --beam-size 10
```

**Return format:**
```python
{
    'product': 'CC(=O)Oc1ccccc1C(=O)O',                    # Canonicalized input
    'predictions': ['CC(=O)O.Oc1ccccc1C(=O)O', ...],      # All predictions
    'valid_predictions': ['CC(=O)O.Oc1ccccc1C(=O)O', ...] # RDKit-validated
}
```

## Usage

**For Custom Data**
```
# Modify the input and model paths in the code file
python infer.py

smiles = 'N#CC1=C(OCC(C)C)C=CC(C2=NC(C)=C(C(O)=O)S2)=C1'
```

**For Uspto Data**
```
python test.py

write2txt(\
    data_name = '50k',\ # 50k or full or mit
    pt_path = 'finetune_50k_label/ train_epoch_2.pth',\  # model path folder
    label=True,         # data with reaction label
    test_aug=False      # data with augmentation
    )
```

**Train**
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port Process_Port --config_file deepspeed.yaml train.py
```

## Model and Data Availability
USPTO datasets, the weight files of RSGPT and the results of augmentation test were uploaded to Zenodo (https://sandbox.zenodo.org/records/203391).  The synthetic data generated in this study were uploaded to Zenodo (https://sandbox.zenodo.org/records/213324).
frags_dic.pkl：This is a part of the {Reactants in the template: [Matched molecular fragments]} that we use in the data generation stage for reference, so that you can use your own templates and molecular fragment libraries to efficiently generate data.

