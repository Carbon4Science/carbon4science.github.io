"""
Simple inference interface for Chemformer.

Usage:
    from Inference import ChemformerInference

    # Initialize with model checkpoint and vocabulary
    model = ChemformerInference(
        model_path="path/to/model.ckpt",
        vocabulary_path="bart_vocab_downstream.json",
        task="backward_prediction",  # or "forward_prediction"
    )

    # Run prediction on single SMILES or list
    results = model.run("CCO")
    results = model.run(["CCO", "CC(=O)O"])
"""

from typing import List, Optional, Union

import omegaconf as oc
from rdkit import Chem

import molbart.utils.data_utils as util
from molbart.models import Chemformer
from molbart.data import SynthesisDataModule


def _canonicalize(smiles: str) -> str:
    """Remove atom mapping and canonicalize SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


class ChemformerInference:
    """Simple inference wrapper for Chemformer model."""

    def __init__(
        self,
        model_path: str,
        vocabulary_path: str = "bart_vocab_downstream.json",
        task: str = "backward_prediction",
        n_beams: int = 10,
        batch_size: int = 64,
        device: str = "cuda",
        n_gpus: int = 1,
    ):
        """
        Initialize Chemformer for inference.

        Args:
            model_path: Path to the trained model checkpoint (.ckpt file)
            vocabulary_path: Path to vocabulary JSON file
            task: "backward_prediction" (retrosynthesis) or "forward_prediction" (product prediction)
            n_beams: Number of beam search results to return
            batch_size: Batch size for inference
            device: Device to run on ("cuda" or "cpu")
            n_gpus: Number of GPUs (set to 0 for CPU-only)
        """
        # Build config
        self.config = oc.OmegaConf.create({
            "model_path": model_path,
            "vocabulary_path": vocabulary_path,
            "task": task,
            "n_beams": n_beams,
            "n_unique_beams": None,
            "batch_size": batch_size,
            "device": device,
            "data_device": device,
            "n_gpus": n_gpus,
            "model_type": "bart",
            "train_mode": "eval",
            "datamodule": None,
        })

        # Handle CPU mode
        if n_gpus < 1 or device == "cpu":
            self.config.n_gpus = 0
            self.config.device = "cpu"
            self.config.data_device = "cpu"

        # Initialize model
        self.chemformer = Chemformer(self.config)
        self.n_beams = n_beams

    def run(
        self,
        smiles: Union[str, List[str]],
        n_beams: Optional[int] = None,
    ) -> List[dict]:
        """
        Run inference on input SMILES.

        Args:
            smiles: Single SMILES string or list of SMILES strings
            n_beams: Number of predictions to return (overrides default if provided)

        Returns:
            List of dicts with keys:
                - "input": input SMILES
                - "predictions": list of predicted SMILES
                - "log_likelihoods": list of log-likelihood scores for each prediction
        """
        # Handle single SMILES input
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = list(smiles)

        # Remove atom mapping and canonicalize (model vocab doesn't include mapped tokens)
        smiles_list = [_canonicalize(s) for s in smiles_list]

        # Set beam size
        beam_size = n_beams if n_beams is not None else self.n_beams
        self.chemformer.model.num_beams = beam_size
        self.chemformer.model.n_unique_beams = beam_size

        # Create datamodule for input SMILES
        # Both reactants and products are set to input since we don't have ground truth
        datamodule = SynthesisDataModule(
            reactants=smiles_list,
            products=smiles_list,
            tokenizer=self.chemformer.tokenizer,
            batch_size=self.config.batch_size,
            max_seq_len=util.DEFAULT_MAX_SEQ_LEN,
            dataset_path="",
        )
        datamodule.setup()

        # Run prediction
        predictions, log_lhs, original_smiles = self.chemformer.predict(
            dataloader=datamodule.full_dataloader()
        )

        # Format output
        results = []
        for inp, preds, lhs in zip(original_smiles, predictions, log_lhs):
            results.append({
                "input": inp,
                "predictions": list(preds) if hasattr(preds, "__iter__") else [preds],
                "log_likelihoods": [float(l) for l in lhs] if hasattr(lhs, "__iter__") else [float(lhs)],
            })

        return results


# Convenience function for quick usage
_model_instance = None


def load_model(
    model_path: str,
    vocabulary_path: str = "bart_vocab_downstream.json",
    task: str = "backward_prediction",
    **kwargs,
) -> ChemformerInference:
    """
    Load and cache a Chemformer model for inference.

    Args:
        model_path: Path to the trained model checkpoint
        vocabulary_path: Path to vocabulary JSON file
        task: "backward_prediction" or "forward_prediction"
        **kwargs: Additional arguments passed to ChemformerInference

    Returns:
        ChemformerInference instance
    """
    global _model_instance
    _model_instance = ChemformerInference(
        model_path=model_path,
        vocabulary_path=vocabulary_path,
        task=task,
        **kwargs,
    )
    return _model_instance


def run(smiles: Union[str, List[str]], top_k: Optional[int] = None) -> List[dict]:
    """
    Run inference on SMILES using the loaded model.

    Must call load_model() first.

    Args:
        smiles: Single SMILES string or list of SMILES strings
        top_k: Number of predictions to return

    Returns:
        List of result dicts, one per input SMILES. Each dict contains:
            - 'input': Input SMILES string
            - 'predictions': List of prediction dicts with 'smiles' and 'score'

    Example:
        >>> load_model("model.ckpt", "vocab.json")
        >>> results = run("CCO")
        >>> results[0]['predictions'][0]
        {'smiles': 'C=C.O', 'score': -0.5}
    """
    if _model_instance is None:
        raise RuntimeError(
            "No model loaded. Call load_model(model_path, vocabulary_path) first."
        )
    raw_results = _model_instance.run(smiles, n_beams=top_k)

    # Convert to uniform format
    results = []
    for r in raw_results:
        formatted_preds = []
        for pred, score in zip(r['predictions'], r['log_likelihoods']):
            formatted_preds.append({
                'smiles': pred,
                'score': score
            })
        results.append({
            'input': r['input'],
            'predictions': formatted_preds
        })
    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Chemformer inference")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", default="bart_vocab_downstream.json", help="Path to vocabulary")
    parser.add_argument("--task", default="backward_prediction", choices=["backward_prediction", "forward_prediction"])
    parser.add_argument("--smiles", required=True, help="Input SMILES string")
    parser.add_argument("--n_beams", type=int, default=10, help="Number of predictions")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load model and run
    model = load_model(
        model_path=args.model_path,
        vocabulary_path=args.vocab_path,
        task=args.task,
        device=args.device,
        n_gpus=1 if args.device == "cuda" else 0,
    )

    results = run(args.smiles, n_beams=args.n_beams)

    # Print results
    for result in results:
        print(f"\nInput: {result['input']}")
        print("Predictions:")
        for i, (pred, lh) in enumerate(zip(result['predictions'], result['log_likelihoods']), 1):
            print(f"  {i}. {pred} (log-likelihood: {lh:.4f})")
