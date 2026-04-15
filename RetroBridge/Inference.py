"""
Simple inference interface for RetroBridge model.

Usage:
    from inference import run

    # Get predicted reactants for a product SMILES
    results = run("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    print(results)  # List of predicted reactant SMILES
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add RetroBridge directory to path so 'src' package is importable from any working directory
_RETROBRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _RETROBRIDGE_DIR not in sys.path:
    sys.path.insert(0, _RETROBRIDGE_DIR)

import torch
from rdkit import Chem

# Lazy imports to speed up module loading
_model = None
_device = None
_dataset_info = None


def _disable_rdkit_logging():
    """Disable RDKit logging messages."""
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


def _get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    # MPS doesn't support float64, which is used in the model
    # So we fall back to CPU for non-CUDA systems
    return torch.device('cpu')


def _move_tensors_to_cpu(obj, visited=None):
    """Recursively move all tensors in an object to CPU."""
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v, visited) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_tensors_to_cpu(v, visited) for v in obj)
    elif hasattr(obj, '__dict__'):
        for attr_name in list(obj.__dict__.keys()):
            try:
                attr_val = getattr(obj, attr_name)
                if isinstance(attr_val, torch.Tensor):
                    setattr(obj, attr_name, attr_val.cpu())
                elif hasattr(attr_val, '__dict__') or isinstance(attr_val, (dict, list, tuple)):
                    _move_tensors_to_cpu(attr_val, visited)
            except Exception:
                pass
    return obj


def _load_model(checkpoint_path: str = "models/retrobridge.ckpt", device=None):
    """Load the RetroBridge model."""
    global _model, _device, _dataset_info

    if device is None:
        device = _get_device()
    _device = device

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get hyperparameters
    hp = checkpoint['hyper_parameters']

    # Extract dataset_infos from checkpoint and move tensors to CPU
    _dataset_info = hp['dataset_infos']
    _move_tensors_to_cpu(_dataset_info)

    # Create dummy objects for metrics (not needed for inference)
    from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
    from src.metrics.sampling_metrics import DummySamplingMolecularMetrics

    train_metrics = TrainMolecularMetricsDiscrete(_dataset_info)
    sampling_metrics = DummySamplingMolecularMetrics()

    # Get extra_features and domain_features from checkpoint (move tensors to CPU)
    extra_features = hp['extra_features']
    domain_features = hp['domain_features']
    _move_tensors_to_cpu(extra_features)
    _move_tensors_to_cpu(domain_features)

    # Import the model class
    from src.frameworks.markov_bridge import MarkovBridge

    # Manually instantiate the model with CPU-based objects
    _model = MarkovBridge(
        experiment_name=hp.get('experiment_name', 'inference'),
        chains_dir=hp.get('chains_dir', './chains'),
        graphs_dir=hp.get('graphs_dir', './graphs'),
        checkpoints_dir=hp.get('checkpoints_dir', './checkpoints'),
        diffusion_steps=hp.get('diffusion_steps', 500),
        diffusion_noise_schedule=hp.get('diffusion_noise_schedule', 'cosine'),
        transition=hp.get('transition', None),
        lr=hp.get('lr', 0.0002),
        weight_decay=hp.get('weight_decay', 1e-12),
        n_layers=hp.get('n_layers', 5),
        hidden_mlp_dims=hp.get('hidden_mlp_dims', {'X': 256, 'E': 128, 'y': 128}),
        hidden_dims=hp.get('hidden_dims', {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}),
        lambda_train=hp.get('lambda_train', [5, 0]),
        dataset_infos=_dataset_info,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=None,
        extra_features=extra_features,
        domain_features=domain_features,
        use_context=hp.get('use_context', True),
        log_every_steps=hp.get('log_every_steps', 50),
        sample_every_val=hp.get('sample_every_val', 20),
        samples_to_generate=hp.get('samples_to_generate', 128),
        samples_to_save=hp.get('samples_to_save', 128),
        samples_per_input=hp.get('samples_per_input', 5),
        chains_to_save=hp.get('chains_to_save', 5),
        number_chain_steps_to_save=hp.get('number_chain_steps_to_save', 50),
        fix_product_nodes=hp.get('fix_product_nodes', True),
        loss_type=hp.get('loss_type', 'vlb'),
    )

    # Load only the model weights (filter out metrics)
    state_dict = checkpoint['state_dict']
    model_state_dict = {}
    for key, value in state_dict.items():
        # Only load model weights, skip metrics
        if key.startswith('model.') or key.startswith('noise_schedule.') or key.startswith('transition_model.'):
            model_state_dict[key] = value.cpu()

    # Load the filtered state dict
    _model.load_state_dict(model_state_dict, strict=False)

    # Move model to device
    _model = _model.to(device)
    _model.eval()

    print(f"Model loaded successfully on {device}")
    return _model


def _assign_trivial_atom_mapping_numbers(molecule):
    """Assign trivial atom mapping numbers to molecule atoms."""
    order = {}
    for atom in molecule.GetAtoms():
        idx = atom.GetIdx()
        atom.SetAtomMapNum(idx)
        order[idx] = idx
    return molecule, order


def run(
    smiles,
    top_k: int = 10,
    n_steps: int = 500,
    checkpoint_path: str = None,
    device: str = None,
    seed: int = 42,
    return_rdkit: bool = False,
    verbose: bool = True,
):
    """
    Run retrosynthesis prediction for a given product SMILES.

    Args:
        smiles: Product SMILES string or list of SMILES strings
        top_k: Number of predictions to return (default: 10)
        n_steps: Number of diffusion steps (default: 500)
        checkpoint_path: Path to model checkpoint (default: "models/retrobridge.ckpt")
        device: Device to use ('cuda:0', 'cpu', or None for auto-detect)
        seed: Random seed for reproducibility (default: 42)
        return_rdkit: If True, also return RDKit molecule objects (default: False)
        verbose: If True, show progress bar during sampling (default: True)

    Returns:
        List of result dicts, one per input SMILES. Each dict contains:
            - 'input': Input SMILES string
            - 'predictions': List of prediction dicts with 'smiles' and 'score'

    Example:
        >>> results = run("CCO")
        >>> results[0]['predictions'][0]
        {'smiles': 'C=C.O', 'score': 1.0}
    """
    global _model, _device, _dataset_info

    _disable_rdkit_logging()

    # Handle input
    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    # Set random seed for reproducibility
    from src.utils import set_deterministic
    set_deterministic(seed)

    # Default checkpoint path relative to this file's directory
    if checkpoint_path is None:
        checkpoint_path = os.path.join(_RETROBRIDGE_DIR, "models", "retrobridge.ckpt")

    # Load model if not already loaded
    if _model is None:
        if device is not None:
            device = torch.device(device)
        _load_model(checkpoint_path, device)

    # Import required modules
    from torch_geometric.data import Data
    from src.data.retrobridge_dataset import RetroBridgeDatasetInfos, RetroBridgeDataset
    from src.analysis.rdkit_functions import build_molecule

    # Set number of diffusion steps
    _model.T = n_steps

    results = []
    for input_smiles in smiles_list:
        # Parse input SMILES
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            results.append({
                'input': input_smiles,
                'predictions': [],
                'error': 'Invalid SMILES'
            })
            continue

        pmol, mapping = _assign_trivial_atom_mapping_numbers(mol)
        r_num_nodes = pmol.GetNumAtoms() + RetroBridgeDatasetInfos.max_n_dummy_nodes

        # Compute graph representation
        p_x, p_edge_index, p_edge_attr = RetroBridgeDataset.compute_graph(
            pmol, mapping, r_num_nodes, RetroBridgeDataset.types, RetroBridgeDataset.bonds
        )
        p_x = p_x.to(_device)
        p_edge_index = p_edge_index.to(_device)
        p_edge_attr = p_edge_attr.to(_device)

        # Sample in chunks to avoid GPU memory/compute blowup
        # (attention is O(n^2) on the batched graph, so large top_k is very expensive)
        chunk_size = 10
        all_molecule_list = []
        for chunk_start in range(0, top_k, chunk_size):
            chunk_k = min(chunk_size, top_k - chunk_start)
            dataset, batch = [], []
            idx_offset = 0
            for i in range(chunk_k):
                data = Data(idx=chunk_start + i, p_x=p_x,
                            p_edge_index=p_edge_index.clone(),
                            p_edge_attr=p_edge_attr, p_smiles=input_smiles)
                data.p_edge_index += idx_offset
                dataset.append(data)
                batch.append(torch.ones_like(data.p_x[:, 0]).to(torch.long) * i)
                idx_offset += len(data.p_x)

            data, _ = RetroBridgeDataset.collate(dataset)
            data.batch = torch.concat(batch)

            with torch.no_grad():
                molecule_list = _model.sample_chain_no_true_no_save(
                    data, batch_size=chunk_k
                )
            all_molecule_list.extend(molecule_list)

        # Convert to SMILES and format predictions
        formatted_preds = []
        for mol_data in all_molecule_list:
            rdmol, _ = build_molecule(mol_data[0], mol_data[1], _dataset_info.atom_decoder, return_n_dummy_atoms=True)
            smi = Chem.MolToSmiles(rdmol)
            formatted_preds.append({
                'smiles': smi,
                'score': 1.0  # RetroBridge doesn't provide scores
            })

        results.append({
            'input': input_smiles,
            'predictions': formatted_preds
        })

    return results


def reset_model():
    """Reset the loaded model (useful for loading a different checkpoint)."""
    global _model, _device, _dataset_info
    _model = None
    _device = None
    _dataset_info = None


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        smiles = sys.argv[1]
    else:
        # Caffeine
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    print(f"Product SMILES: {smiles}")
    print("Running retrosynthesis prediction...")
    print("-" * 50)

    results = run(smiles, top_k=5, n_steps=500)

    print("Predicted reactants:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")
