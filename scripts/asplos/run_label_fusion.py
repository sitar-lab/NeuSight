import subprocess
import torch
import os
import sys
from pathlib import Path

def cuda_rr():
  idx = cuda_rr.cuda_idx % cuda_rr.device_num
  cuda_rr.cuda_idx += 1
  return "cuda:"+str(idx)
cuda_rr.cuda_idx = 0
cuda_rr.device_num = torch.cuda.device_count()

jobs = []

def launch_task(model_name, mode, batch_size, seqlen, overwrite=True, blocking=False):
    model_config_path = Path.home() / Path("NeuSight/scripts/asplos/data/DLmodel_configs")/(model_name+".json")

    p = subprocess.Popen(
        [
            "python3", "./collect_label_fusion.py",
            "--model_config_path", model_config_path,
            "--batch_size", str(batch_size),
            "--seqlen", str(seqlen),
            "--result_dir", "./label",
        ],
        stdout=sys.stdout, 
        stderr=sys.stdout, 
    )

    if blocking:
        p.wait()

    jobs.append(p)

def trace():
    launch_task(model_name="bert_large", mode="inf", batch_size=8, seqlen=512, blocking=True)
    launch_task(model_name="bert_large", mode="inf", batch_size=16, seqlen=512, blocking=True)
    launch_task(model_name="gpt2_large", mode="inf", batch_size=4, seqlen=1024, blocking=True)
    launch_task(model_name="gpt2_large", mode="inf", batch_size=8, seqlen=1024, blocking=True)

    for p in jobs:
        p.wait()

def main():
    trace()

try:
    main()
except Exception as e:
    print(e)
    for p in jobs:
        print(p.stdout)
        p.kill()