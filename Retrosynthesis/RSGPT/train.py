import os

import torch
import logging
import argparse
import wandb
import shutil

from utils.utils import args_parse, seed_everything
from task import Task, Trainer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, input_ids,attn_masks):
        self.attn_masks = attn_masks
        self.input_ids = input_ids

        self.input_ids = torch.tensor(self.input_ids,dtype=torch.int8)
        self.attn_masks = torch.tensor(self.attn_masks,dtype=torch.int8)

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attn_mask = self.attn_masks[idx]
        return input_id,attn_mask

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_NVFUSER_DISABLE"] = "fallback"
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="rxngpt")
    parser.add_argument("--config", type=str, default='rxngpt', # name1 <---> configs/name1.yml datasets/ @register_datasets(['pretrain'])
                        choices=['rxngpt'], # name1
                        help="Selected a config for this task.")
    parser.add_argument("--gpus", type=str,  default='0')
    parser.add_argument('--debug', type=bool, default=True, help='is or not debug') 
    parser.add_argument('--project_name', type=str, default='UniGPT-molecule-generate', help='project name') 
    parser.add_argument('--task_name', type=str,
                        default='rxngpt', help='task name') 
    parser.add_argument('--save', type=str, default='rxngpt_multistep') 
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args, parser

def updata_cfg(cfg, args):
    for k, v in args.__dict__.items():
        cfg[k] = v
    return cfg

def main():
    # configs
    args, parser = parse_args()
    cfg = args_parse(f'{args.config}.yml')
    cfg = updata_cfg(cfg, args)

    # set
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[ddp_scaler])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    seed_everything(args.seed)

    if accelerator.is_main_process:
        if args.save is not None:
            os.makedirs(os.path.join('./save', args.save), exist_ok=True)
            shutil.copy(os.path.join('configs', args.config + '.yml'), os.path.join('./save', args.save, 'configs.yml'))
        if not args.debug:
            wandb.login(key=cfg.WANDB.KEY)
            wandb.init(project=args.project_name, entity="fxj", name=args.task_name) # wandb改名
            wandb.config = args
            # pass

    accelerator.wait_for_everyone()

    # load task
    task = Task.setup_task(cfg)
    task.set(accelerator, logger, wandb)

    # model, datasets, loss
    task.build_dataset(cfg)
    task.build_model(cfg)

    # optim
    task.build_optim(cfg)

    # train
    trainer = Trainer(task, cfg)
    trainer.train()


if __name__ == '__main__':
    main()
