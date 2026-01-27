import os
import torch
from utils.utils import args_parse
from models.rxngpt import  RxnGPT
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributions as D

import re
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
from rdkit import Chem

import lmdb
import pickle
import os

import torch
import logging
import argparse
import wandb
import shutil

from utils.utils import args_parse, seed_everything
from task import Task, Trainer
from tokenizer.tokenization import SMILESBPETokenizer
import pandas as pd

import torch
import torch.nn.functional as F

device = 'cuda:0'
def deduplicate(data):
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def beam_search_gpt(model, tokenizer, s, beam_size=10, max_length=50,device=device):
    input_ids = tokenizer.encode(s, add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    sequences = [(input_ids, 0.0)]  # 每个元素为 (序列, 累积得分)
    end_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
    
    completed_sequences = []
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[0, -1].item() == end_token_id:
                completed_sequences.append((seq, score))
                continue
            
            logits = model.infer(input_ids=seq).logits[:, -1, :]
            logits = F.log_softmax(logits, dim=-1)
            
            topk_probs, topk_indices = torch.topk(logits, beam_size+10, dim=-1)
            
            for i in range(beam_size+10):
                candidate_seq = torch.cat([seq, topk_indices[:, i].unsqueeze(0)], dim=1)
                candidate = (candidate_seq, score - topk_probs[0, i].item())
                all_candidates.append(candidate)
        
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_size+10]
    
    completed_sequences.extend(sequences)
    completed_sequences = sorted(completed_sequences, key=lambda tup: tup[1])
    
    decoded_sequences = [tokenizer.decode(seq[0].squeeze().tolist()) for seq in completed_sequences]
    decoded_sequences = deduplicate(decoded_sequences)[:beam_size]
    return decoded_sequences

maxlen = 100
cfg = args_parse('/home/xinda/R/configs/rxngpt.yml')
tokenizer = SMILESBPETokenizer.get_hf_tokenizer("/home/xinda/codes/rxn_finetune/t.json", model_max_length=maxlen)


pt_path = '/home/xinda/codes/rxn_finetune/save/pretrain.pth'

def stand(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m)

cases = [
"N#CC1=C(OCC(C)C)C=CC(C2=NC(C)=C(C(O)=O)S2)=C1"
]


cases = [stand(case) for case in cases]
prefixs = []
input_ids = []
for c in cases:
    prefix = '<s><Isyn><O>{}<F1>'.format(c)
    prefixs.append(prefix)
    input_id = tokenizer.encode(prefix,add_special_tokens=False)
    input_ids.append(input_id)




# with open('/home/xinda/make_data/vocab2num.pkl','rb') as f:
#     vocab2num = pickle.load(f)
# num2vocab = {v: k for k, v in vocab2num.items()}

dic = torch.load(pt_path, map_location=device)
new_dic = {}
for k,v in dic.items():
    if k.startswith('module.'):
        new_dic[k[7:]] = v
model = RxnGPT(cfg,Tokenizer=1)
model.load_state_dict(new_dic)

# model = model.half()
model.half()

model.to(device)
model.eval()
print('Model Loaded')
pt_name = '_'.join(pt_path.split('/')[-2:])

end_id = 2 ## <Rxn>=9, </s>=2

for p in tqdm(prefixs):
    output_sequence = beam_search_gpt(model, tokenizer, p, beam_size=10, max_length=maxlen,device=device)
    for o in output_sequence:
        with open('txts/case3.txt','a+') as f:
            f.write(o+'\n')
