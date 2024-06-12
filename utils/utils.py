import os
from os.path import join as ospj
import torch
import random
import copy
import shutil
import sys
import yaml

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_norm_values(norm_family = 'imagenet'):
    '''
        Inputs
            norm_family: String of norm_family
        Returns
            mean, std : tuple of 3 channel values
    '''
    if norm_family == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError('Incorrect normalization family')
    return mean, std

def save_args(args, log_path, argfile):
    # shutil.copy()Python中的方法用于将源文件的内容复制到目标文件或目录。它还会保留文件的权限模式，但不会保留文件的其他元数据(例如文件的创建和修改时间)。
    shutil.copy('train.py', log_path)
    shutil.copy('dataset.py', log_path)
    modelfiles = ospj(log_path, 'models')
    try:
        shutil.copy(argfile, log_path)  # 复制配置文件到log_path
    except:
        print('Save to log_path: Config file exists')
    try:
        shutil.copytree('models/', modelfiles)  # 把所有模型复制过来
    except:
        print('Save to log_path: Models already exists')

    with open(ospj(log_path,'args_all.yaml'),'w') as f:
        # yaml.dump()函数，就是将yaml文件一次性全部写入你创建的文件
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    with open(ospj(log_path, 'args.txt'), 'w') as f:
        # 把命令行（python train.py --config configs/cge/xxx.yml）文件名后面的参数都写进这个txt中
        f.write('\n'.join(sys.argv[1:]))    # 用换行符连接argv中的字符串

class UnNormalizer:
    '''
    Unnormalize a given tensor using mean and std of a dataset family
    Inputs
        norm_family: String, dataset
        tensor: Torch tensor
    Outputs
        tensor: Unnormalized tensor
    '''
    def __init__(self, norm_family = 'imagenet'):
        self.mean, self.std = get_norm_values(norm_family=norm_family)
        self.mean, self.std = torch.Tensor(self.mean).view(1, 3, 1, 1), torch.Tensor(self.std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        return (tensor * self.std) + self.mean

def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)