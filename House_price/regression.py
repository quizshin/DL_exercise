import hashlib
import os
import tarfile
import zipfile
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    # 断言name在DATA_HUB中
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    # 获取url和sha1_hash
    url, sha1_hash = DATA_HUB[name]
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    # 获取本地文件名
    fname = os.path.join(cache_dir, url.split('/')[-1])
    # 如果本地文件存在
    if os.path.exists(fname):
        # 计算本地文件的sha1值
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        # 如果本地文件的sha1值与DATA_HUB中的sha1值相同
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    # 如果本地文件不存在，则从url下载
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))



