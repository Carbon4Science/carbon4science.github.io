"""
Chemformer forward reaction prediction inference.

Fine-tuned on USPTO-MIT (480K) for 17 epochs from the pretrained BART checkpoint.
Best validation accuracy: 84.5% (epoch 17).
"""

import os
import sys
from typing import List, Dict, Union

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_CHEMFORMER_DIR = os.path.join(_REPO_ROOT, "Retro", "Chemformer")

_DEFAULT_CHECKPOINT = os.path.join(
    _THIS_DIR, "tb_logs", "forward_prediction", "version_0",
    "checkpoints", "epoch=17-step=28763.ckpt"
)
_DEFAULT_VOCAB = os.path.join(_CHEMFORMER_DIR, "bart_vocab_downstream.json")

_chemformer_instance = None


def load_model(model_path: str = None, vocabulary_path: str = None, **kwargs):
    global _chemformer_instance

    if _CHEMFORMER_DIR not in sys.path:
        sys.path.insert(0, _CHEMFORMER_DIR)

    import omegaconf as oc
    from molbart.models import Chemformer
    from molbart.data import SynthesisDataModule
    import molbart.utils.data_utils as util

    config = oc.OmegaConf.create({
        "model_path": model_path or _DEFAULT_CHECKPOINT,
        "vocabulary_path": vocabulary_path or _DEFAULT_VOCAB,
        "task": "forward_prediction",
        "n_beams": 5,
        "n_unique_beams": None,
        "batch_size": 64,
        "device": "cuda",
        "data_device": "cuda",
        "n_gpus": 1,
        "model_type": "bart",
        "train_mode": "eval",
        "datamodule": None,
    })

    _chemformer_instance = {
        "chemformer": Chemformer(config),
        "util": util,
        "SynthesisDataModule": SynthesisDataModule,
    }


def run(input_data: Union[str, List[str]], top_k: int = 5) -> List[Dict]:
    global _chemformer_instance
    if _chemformer_instance is None:
        load_model()

    if isinstance(input_data, str):
        smiles_list = [input_data]
    else:
        smiles_list = list(input_data)

    from rdkit import Chem

    def canonicalize(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True)

    smiles_list = [canonicalize(s) for s in smiles_list]

    chemformer = _chemformer_instance["chemformer"]
    util = _chemformer_instance["util"]
    SynthesisDataModule = _chemformer_instance["SynthesisDataModule"]

    chemformer.model.num_beams = top_k
    chemformer.model.n_unique_beams = top_k

    datamodule = SynthesisDataModule(
        reactants=smiles_list,
        products=smiles_list,
        tokenizer=chemformer.tokenizer,
        batch_size=64,
        max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
        dataset_path="",
    )
    datamodule.setup()

    predictions, log_lhs, original_smiles = chemformer.predict(
        dataloader=datamodule.full_dataloader()
    )

    results = []
    for inp, preds, lhs in zip(original_smiles, predictions, log_lhs):
        formatted = []
        for smi, score in zip(preds, lhs):
            formatted.append({"smiles": smi, "score": float(score)})
        results.append({"input": inp, "predictions": formatted})
    return results
