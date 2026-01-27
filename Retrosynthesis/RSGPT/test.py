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

def write2txt(data_name = '50k',\
              pt_path = '/home/xinda/codes/rxn_finetune/save/finetune_50k/train_epoch_3.pth',\
              label=False,\
              test_aug=False
              ):



    file_name = data_name+'_'+pt_path.split('/')[-2]+'_'+pt_path.split('/')[-1]+'_'+device
    if label:
        prefix_path = 'test_{}_label.txt'.format(data_name)
    else:
        prefix_path = 'test_{}.txt'.format(data_name)

    if test_aug:
        prefix_path = 'test_{}_prefixs20.txt'.format(data_name)
        file_name = file_name+'_aug20'

    maxlen = 100
    def deduplicate(data):
        seen = set()
        result = []
        for item in data:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    def beam_search_gpt(model, tokenizer, s, beam_size=10, max_length=50,device=device,deduolicated = False):
        # input_ids = tokenizer.encode(s, add_special_tokens=False)
        input_ids = torch.tensor(s).unsqueeze(0).to(device)
        
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
                
                topk_probs, topk_indices = torch.topk(logits, beam_size, dim=-1)
                
                for i in range(beam_size):
                    candidate_seq = torch.cat([seq, topk_indices[:, i].unsqueeze(0)], dim=1)
                    candidate = (candidate_seq, score - topk_probs[0, i].item())
                    all_candidates.append(candidate)
            
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_size]
        
        completed_sequences.extend(sequences)
        completed_sequences = sorted(completed_sequences, key=lambda tup: tup[1])
        
        decoded_sequences = [tokenizer.decode(seq[0].squeeze().tolist()) for seq in completed_sequences]
        if deduolicated:
            decoded_sequences = deduplicate(decoded_sequences)[:beam_size]
        else:
            decoded_sequences = decoded_sequences[:beam_size]
        while len(decoded_sequences)<beam_size:
            decoded_sequences.append('C')
        return decoded_sequences

    cfg = args_parse('/home/xinda/R/configs/rxngpt.yml')
    tokenizer = SMILESBPETokenizer.get_hf_tokenizer("t.json", model_max_length=maxlen)




    def stand(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m)
    # cases = [stand(case) for case in cases]
    # prefixs = []
    # input_ids = []
    # for c in cases:
    #     prefix = '<s><Isyn><O>{}<F1>'.format(c)
    #     prefixs.append(prefix)
    #     input_id = tokenizer.encode(prefix,add_special_tokens=False)
    #     input_ids.append(input_id)


    prefixs = []
    with open(prefix_path,'r') as f:
        lines = f.readlines()
        lenth = len(lines)
        for line in lines[:lenth//8]:
            line = line.strip().replace('\n','')
            if line:
                pre = tokenizer.encode(line,add_special_tokens=False)
                prefixs.append(pre)    


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

    model.half()

    model.to(device)
    model.eval()
    print('Model Loaded')
    pt_name = '_'.join(pt_path.split('/')[-2:])

    end_id = 2 ## <Rxn>=9, </s>=2

    # 用法示例
    # input_id_tensor = ...
    for p in tqdm(prefixs):
        output_sequence = beam_search_gpt(model, tokenizer, p, beam_size=10, max_length=maxlen,device=device)
        for o in output_sequence:
            with open('txts/{}.txt'.format(file_name),'a+') as f:
                f.write(o+'\n')
    print('Done')
if __name__ == '__main__':

    # # # pretrain
    # write2txt(\
    #     data_name = '50k',\
    #     pt_path = '/home/xinda/codes/rxn_finetune/save/finetune_50k_aug20/train_epoch_1.pth',\
    #     label=False,
    #     test_aug=True
    #         )

    # # pretrain
    write2txt(\
        data_name = '50k',\
        pt_path = '/home/xinda/codes/rxn_finetune/save/finetune_50k_label/train_epoch_2.pth',\
        label=True,
        test_aug=False
            )
    
