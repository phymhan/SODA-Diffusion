# set visiable GPU
import os
gpu = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
import torch,time

a = torch.ones(20000, 300000).to('cuda')
print(f"GPU {gpu} is running...")
time.sleep(86400)