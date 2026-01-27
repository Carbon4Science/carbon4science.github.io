import math
import torch
import torch.nn as nn
from utils.utils import accuracy2
import torch.nn.functional as F
import torch.distributions as D
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from models import register_model

from transformers.utils.generic import ModelOutput
from transformers import AdamW, GPT2Model, GPT2PreTrainedModel, GPT2LMHeadModel, \
    AutoModelForCausalLM, LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GPT2Config
from utils.vocab import mapping
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

@register_model(['rxngpt'])
class RxnGPT(LlamaModel):
    def __init__(self, cfg, task=None, Tokenizer=None):
        if task is not None:
            tokenizer = task.tokenizer
        elif Tokenizer is not None:
            tokenizer = Tokenizer
        else:
            raise RuntimeError('Tokenizer is None!')
        if cfg.MODEL.GPT_MODEL.config_path:
            config = LlamaConfig.from_pretrained(cfg.MODEL.GPT_MODEL.config_path)
            cfg.MODEL.GPT_MODEL.n_layer = config.num_hidden_layers
            cfg.MODEL.GPT_MODEL.n_head = config.num_attention_heads
            cfg.MODEL.GPT_MODEL.n_embd = config.hidden_size
        else:
            config = LlamaConfig(
                vocab_size=len(mapping),
                bos_token_id=mapping['<s>'],
                eos_token_id=mapping['</s>'],
                num_hidden_layers=cfg.MODEL.GPT_MODEL.n_layer,
                num_attention_heads=cfg.MODEL.GPT_MODEL.n_head,
                hidden_size=cfg.MODEL.GPT_MODEL.n_embd,
                max_position_embeddings=cfg.DATA.MAX_ATOM_NUM,
            )
        super().__init__(config)

        self.model = LlamaForCausalLM(config)
        self.vocab_size = config.vocab_size
        self.cfg = cfg
        self.tokenizer = tokenizer

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task)


    def sample_from_topp_independent(self, topp, logits):
        logits = F.softmax(logits, dim=-1)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        sorted_indices_to_remove = cumulative_probs > topp

        # make sure at least have one point to sample
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = 0.0
        token_prob = logits / logits.sum(dim=-1, keepdim=True)
        token_prob = token_prob.type(torch.float32)
        token_dist = D.Categorical(token_prob)
        next_token = token_dist.sample((1,)).permute(1, 0)
        return next_token

    def generate_next_token(self, model, input_ids, hidden_state, temperature=1.0, topp=0.95):
        """
        对于给定的上文，生成下一个单词
        """
        # outputs = model(input_ids=input_ids)
        input_data = {'input_ids': input_ids,
                      'hidden_state': hidden_state}
        with torch.no_grad():
            outputs = model(**input_data)

        logits = outputs['logits']
        # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature
        next_token_id = self.sample_from_topp_independent(topp, next_token_logits)
        # next_token_id = torch.argmax(torch.softmax(next_token_logits, dim=-1), dim=-1)
        # next_token_id = next_token_id.unsqueeze(1)
        return next_token_id, outputs


    def _generate(self, model, encoder_hidden_state):
        end_id = self.tokenizer.eos_token_id
        unk_id = self.tokenizer.unk_token_id
        batch = encoder_hidden_state.shape[0]
        input_ids = torch.LongTensor([[self.tokenizer.bos_token_id]]).cuda()
        input_ids = input_ids.repeat(batch, 1)
        cur_len = input_ids.shape[1]
        end_tensor = torch.zeros((input_ids.shape[0]), dtype=torch.bool, device=input_ids.device)
        while True:
            next_token_id, outputs = self.generate_next_token(model=model,
                                                              input_ids=input_ids,
                                                              hidden_state=encoder_hidden_state,
                                                              temperature=self.cfg.temperature, topp=self.cfg.topp)

            if end_tensor.sum() == input_ids.shape[0]:
                break
            if input_ids.shape[1] > self.cfg.DATA.MAX_LEN:
                break

            end_tensor |= (next_token_id == end_id).squeeze()
            next_token_id[end_tensor] = end_id
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            cur_len += 1
            # print(input_ids.shape)

        return input_ids    
    
    def LMLoss(self, lm_logits, labels):
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # acc = accuracy2(lm_logits[:, :-1], labels[:, 1:])
        return loss
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss: Optional[bool] = True,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ):  
        if loss:
            outputs = self.model(input_ids=input_ids, labels=input_ids,attention_mask=attention_mask)
            return {'loss':outputs.loss}
        else:
            outputs = self.model(input_ids=input_ids, labels=input_ids,attention_mask=attention_mask,output_hidden_states=True)

            return outputs
        
    def load_weights(self,pt_path):
        
        dic = torch.load(pt_path)
        new_dic = {}
        for k,v in dic.items():
            if k.startswith('module.'):
                new_dic[k[7:]] = v
                
        self.model.load_state_dict(dic)

    def infer(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        
        return self.model(input_ids=input_ids)
    
    def sentence_embedding(self,input_ids):
        outputs = self.model.forward(
            input_ids=input_ids,
            loss=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # 获取最后一层的hidden states
        hidden_states = outputs.hidden_states[-1]

        # 使用第一个token的向量作为句子的嵌入
        sentence_embeddings = hidden_states[:, 0, :]
        return sentence_embeddings



@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    lm_logits_atom: torch.FloatTensor = None
    lm_logits_coord: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None