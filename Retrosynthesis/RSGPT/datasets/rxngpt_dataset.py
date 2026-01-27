import lmdb
import os
import math
import pickle
import copy
import numpy as np
import torch
import random
from math import ceil
from rdkit import Chem
from functools import lru_cache
from rdkit.Chem import rdMolTransforms
from datasets import register_datasets
from utils.rotation import rotation
from typing import Optional
from datasets.tokenizer import *
from utils.utils import coord_to_grid, grid_to_coord
from scipy.spatial.transform import Rotation as R
from utils.vocab import mapping, id2symbol
from torch.utils.data import Dataset
import pickle
import lmdb
import pickle
import numpy as np

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    return env, txn

class MyDataset(Dataset):
    def __init__(self, input_ids,attn_masks):
        self.attn_masks = attn_masks
        self.input_ids = input_ids

        self.input_ids = torch.tensor(self.input_ids,dtype=torch.int16)
        self.attn_masks = torch.tensor(self.attn_masks,dtype=torch.int16)

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attn_mask = self.attn_masks[idx]
        return input_id,attn_mask


'''
184729660
1865957
'''

@register_datasets(['rxngpt'])
class RxnGPTDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        if mode == 'train':
            self.data_path = cfg.DATA.TRAIN_DATA_ROOT
            self.env,self.txn = read_lmdb(self.data_path)
            length = self.txn.stat()["entries"]
            self.num_data = length
        elif mode == 'valid':
            self.data_path = cfg.DATA.VALID_DATA_ROOT
            self.env,self.txn = read_lmdb(self.data_path)
            length = self.txn.stat()["entries"]
            self.num_data = length

        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_LEN
        )

        
    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)


    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        # input_id = self.dataset[idx]
        # input_id = torch.tensor(input_id,dtype=torch.int64)
        # input_id = input_id.to(torch.int64)

        datapoint_pickled = self.txn.get(str(idx).encode())
        input_id = pickle.loads(datapoint_pickled)
        input_id = torch.tensor(input_id,dtype=torch.int64)
        data = {}
        data['input_ids'] = input_id


        # input_id = self.input_ids_list[idx]
        # input_id = torch.tensor(input_id,dtype=torch.int64)
        # data = {}
        # data['input_ids'] = input_id
        return data

    def collator(self, batch):
        pad_input_ids = []
        # pad_attn_masks = []
        
        max_length = max(len(sequence['input_ids']) for sequence in batch)
        max_length = min(max_length,260) # 避免爆显存
        for b in batch:
            pad_b = pad_to_max_length_1d(b['input_ids'], max_length)
            pad_input_ids.append(pad_b)

        pad_input_ids = torch.cat(pad_input_ids)
        # pad_attn_masks = torch.cat(pad_attn_masks)
        pad_input_ids = pad_input_ids.view(len(batch),-1)
        # pad_attn_masks = pad_attn_masks.view(len(batch),-1)
        return {
            'input_ids': pad_input_ids
        }


def pad_to_max_length_1d(x, max_length):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    l = x.shape[0]
    new_x = torch.zeros_like(x,dtype=torch.int64)
    if l > max_length:
        new_x = new_x[:max_length]
    else:
        x_shape = [i for i in x.shape]
        x_shape[0] = max_length - l
        new_x = torch.cat([new_x, torch.zeros(tuple(x_shape))])
    new_x[:l] = x[:max_length]
    new_x = new_x.to(x.dtype)
    return new_x

def pad_to_max_length_2d(x, max_length):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    l = x.shape[0]
    d = x.shape[1]
    new_x = torch.zeros((max_length, d))
    new_x[:l] = x[:max_length]
    return new_x

def pad_to_max_length_2d_matrix(x, max_length):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    l = x.shape[0]
    new_x = torch.zeros((max_length, max_length))
    new_x[:l, :l] = x[:max_length, :max_length]
    return new_x