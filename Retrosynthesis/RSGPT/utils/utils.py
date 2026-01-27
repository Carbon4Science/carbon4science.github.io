import argparse
import datetime
import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
import shutil
import math
from omegaconf import OmegaConf
import copy
from rdkit.Chem import rdMolTransforms
from einops import repeat

import wandb

def get_abs_path(*name):
    fn = os.path.join(*name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(os.getcwd(), fn))


def seed_everything(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)

def get_config(config_path="", opts=[]):
    base_config = OmegaConf.load(get_abs_path('configs', 'base.yml'))
    model_config = OmegaConf.load(get_abs_path('configs', config_path)) if len(config_path) > 0 else OmegaConf.create(
        "")
    cli_config = OmegaConf.from_dotlist(opts)
    config = OmegaConf.merge(base_config, model_config, cli_config)
    return config

def args_parse(config_file=''):
    # parser = argparse.ArgumentParser(description="fast-bbdl")
    # parser.add_argument("--config_file", default="", help="path to config file", type=str)
    # parser.add_argument("--opts", help="Modify config options using the command-line key value", default=[],
    #                     nargs=argparse.REMAINDER)
    #
    # args = parser.parse_args()
    #
    # config_file = args.config_file or config_file
    cfg = get_config(config_file, [])
    return cfg

def move_tokenizer(cfg, save_path):
    if not os.path.exists(os.path.join(save_path, 'tokenizer.json')):
        shutil.copy(os.path.join(cfg.MODEL.CHECKPOINT_PATH, 'tokenizer.json'),
                    os.path.join(save_path, 'tokenizer.json'))
    if not os.path.exists(os.path.join(save_path, 'merges.txt')):
        shutil.copy(os.path.join(cfg.MODEL.CHECKPOINT_PATH, 'merges.txt'),
                    os.path.join(save_path, 'merges.txt'))
    if not os.path.exists(os.path.join(save_path, 'vocab.json')):
        shutil.copy(os.path.join(cfg.MODEL.CHECKPOINT_PATH, 'vocab.json'),
                    os.path.join(save_path, 'vocab.json'))

def save_config(cfg, model):
    save_path = os.path.join('save', cfg.save, 'model_config.json')
    if os.path.exists(save_path):
        return
    model.config.to_json_file(save_path)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def accuracy(outputs, targets, ignore=-100, use_label=False):
    if use_label:
        # TODO: hard coding
        targets[:, :6] = -100

    _, pred = outputs.topk(5, -1, True, True)
    targets_len = (targets != ignore).sum(-1)
    ignore_len = (targets == ignore).sum(-1)

    targets = repeat(targets, 'b l -> b l p', p=5)
    pred[targets == -100] = -100

    res = []
    for k in [1, 5]:
        correct = (pred.eq(targets)[..., :k].sum(-1) >= 1).sum(-1)

        acc = ((correct - ignore_len) / targets_len).mean()
        res.append(acc)

    return res[0], res[1]

def accuracy2(outputs, targets, ignore=-100):
    mask = targets.ne(ignore)
    pred_id = outputs[mask].argmax(-1)
    targets = targets[mask]
    masked_hit = (pred_id == targets).long().sum()
    masked_cnt = mask.long().sum()
    hit_rate = masked_hit/masked_cnt
    return hit_rate

def grid_to_coord(targets, max_dist, resolution):
    box_size = math.ceil(2 * max_dist // resolution + 1)
    # print(f'总共有{box_size**3}个格子')
    coords = []
    for tar in targets:
        grid_x = tar // (box_size**2)
        tar -= grid_x * box_size**2
        grid_y = tar // box_size
        tar -= grid_y * box_size
        grid_z = tar
        grid_coord = np.array([grid_x, grid_y, grid_z])
        coord = grid_coord * resolution - max_dist
        coords.append(coord)
    coords = np.array(coords)
    return coords


def get_grid_coords(coords, max_dist, grid_resolution):
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)
    return grid_coords

def coord_to_grid(coords, max_dist, resolution):
    box_size = math.ceil(2 * max_dist // resolution + 1)
    targets = []
    grid_coords = get_grid_coords(coords, max_dist, resolution)
    for i in range(coords.shape[0]):
        target = grid_coords[i][0] * box_size ** 2 + grid_coords[i][1] * box_size + grid_coords[i][2]
        targets.append(target)
    return np.array(targets)

def get_center(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    center = np.mean(conformer.GetPositions(), axis=0)
    return center


def trans(x, y, z):
    translation = np.eye(4)
    translation[:3, 3] = [x, y, z]
    return translation


def centralize(mol, confId=-1):
    mol = copy.deepcopy(mol)
    conformer = mol.GetConformer(confId)
    center = get_center(mol, confId)
    translation = trans(-center[0], -center[1], -center[2])
    rdMolTransforms.TransformConformer(conformer, translation)
    return mol

if __name__ == '__main__':
    import lmdb
    import pickle

    db = lmdb.open(
        '../data/zinc3d/train.lmdb',
        map_size=5 * (1024 * 1024 * 1024),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    # with db.begin() as txn:
    #     keys = list(txn.cursor().iternext(values=False))

    data = pickle.loads(db.begin().get(str(0).encode()))
    # mol = data['mol']
    # mol = centralize(mol)
    # coord = mol.GetConformers()[0].GetPositions().astype(np.float32)
    #
    # print(coord)
    coord = np.array([[-12, -12, -12], [-15, -5, 8]])

    max_dist = 12
    resolution = 0.5
    targets = coord_to_grid(coord, max_dist, resolution)
    print(targets)

    coord = grid_to_coord(targets, max_dist, resolution)
    print(coord)