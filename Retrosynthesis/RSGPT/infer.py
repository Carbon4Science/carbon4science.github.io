import re
import time
import torch
from tqdm import tqdm
from rdkit import Chem
import torch.nn.functional as F

from models.rxngpt import  RxnGPT
from utils.utils import args_parse
from tokenizer.tokenization import SMILESBPETokenizer

device = 'cuda:1'

def deduplicate(data):
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def beam_search_gpt(model, tokenizer, std_smiles, beam_size=10, max_length=50, device=device):
    t0 = time.time()
    prefix = f'<s><Isyn><O>{std_smiles}<F1>'
    input_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    # 初始化序列：(序列, 累积得分)
    sequences = [(input_ids, 0.0)]
    end_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]

    completed_sequences = []
    t1 = time.time()
    # print(f'前处理耗时：{t1-t0}')

    for _ in range(max_length):
        t2 = time.time()
        all_candidates = []
        active_sequences = []
        active_scores = []

        # 分离已完成和活跃的序列
        for seq, score in sequences:
            if seq[0, -1].item() == end_token_id:
                completed_sequences.append((seq, score))
            else:
                active_sequences.append(seq)
                active_scores.append(score)

        if not active_sequences:
            break  # 所有序列都已完成

        # 批量处理所有活跃序列，只调用一次模型
        t44 = time.time()
        # 将所有活跃序列拼接成一个批次
        batch_input = torch.cat(active_sequences, dim=0)
        # 一次推理获取所有序列的logits
        logits = model.infer(input_ids=batch_input).logits[:, -1, :]
        logits = F.log_softmax(logits, dim=-1)
        t45 = time.time()
        # print(f'模型批量推理耗时：{t45-t44}')

        # 处理每个序列的结果
        for i in range(len(active_sequences)):
            seq = active_sequences[i]
            score = active_scores[i]
            seq_logits = logits[i]

            # 获取当前序列的top候选
            topk_probs, topk_indices = torch.topk(seq_logits, beam_size+10, dim=-1)

            # 生成候选序列
            for j in range(beam_size+10):
                candidate_seq = torch.cat([seq, topk_indices[j].unsqueeze(0).unsqueeze(0)], dim=1)
                candidate_score = score - topk_probs[j].item()
                all_candidates.append((candidate_seq, candidate_score))

        t5 = time.time()
        # print(f'模型后处理耗时：{t5 - t45}')

        # 筛选出最佳候选
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_size+10]
        t3 = time.time()
        # print(f'max_length {_}耗时：{t3 - t2}')

    t31 = time.time()
    # 收集所有完成的序列
    completed_sequences.extend(sequences)
    completed_sequences = sorted(completed_sequences, key=lambda tup: tup[1])

    # 解码并去重
    decoded_sequences = [tokenizer.decode(seq[0].squeeze().tolist()) for seq in completed_sequences]
    decoded_sequences = deduplicate(decoded_sequences)[:beam_size]
    t32 = time.time()
    # print(f'后处理耗时：{t32 - t31}')
    return decoded_sequences



def jiexi(input_texts):
    # 提取并转换格式
    results = []
    for text in input_texts:
        # 清理首尾标签
        cleaned = f'<F1>{text.split("<F1>")[-1]}'[:-4]
        out = f'<F1>{text.split("<F1>")[0]}'.split("<O>")[-1]
        # 提取所有F部分内容
        f_matches = re.findall(r'<F\d+>(.*?)(?=<F|$)', cleaned)
        if not f_matches:
            continue

        # 组合结果
        f_combined = '.'.join(f_matches)
        result = f"{f_combined}>>{out}"
        results.append(result)
    return  results

class RSGPT:
    def __init__(self,model_path='models/rxngpt_ready.pt', cfg_path='base.yml', tokenizer_path='tokenizer/t.json'):

        self.maxlen = 100
        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(tokenizer_path, model_max_length=self.maxlen)
        t1 = time.time()
        dic = torch.load(model_path, map_location=device)
        new_dic = {}
        for k,v in dic.items():
            if k.startswith('module.'):
                new_dic[k[7:]] = v
        self.model = RxnGPT(args_parse(cfg_path),Tokenizer=1)
        self.model.load_state_dict(new_dic)
        self.model.half()
        self.model.to(device)
        self.model.eval()
        # torch.save(self.model, 'models/rxngpt_ready.pt')
        # self.model = torch.load(model_path, map_location=device)
        # self.model.half().eval()
        print(f'模型加载完毕！耗时{time.time()-t1:.2f}s')

    def predict(self, smiles):
        t0 = time.time()
        m = Chem.MolFromSmiles(smiles)
        std_smiles = Chem.MolToSmiles(m)
        output_sequence = beam_search_gpt(self.model, self.tokenizer, std_smiles, beam_size=10, max_length=self.maxlen, device=device)
        t1 = time.time()
        print(f'推理耗时：{t1-t0:.2f}s')
        out_smiles = jiexi(output_sequence)
        # out_smiles = output_sequence
        print(f'解析耗时：{t1 - t0:.2f}s')
        return out_smiles

if __name__ == '__main__':
    gpt = RSGPT()

    smiles = 'N#CC1=C(OCC(C)C)C=CC(C2=NC(C)=C(C(O)=O)S2)=C1'
    print(gpt.predict(smiles))


