# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

"""
数据加载
rstrip() 和 strip() 是 Python 字符串对象的两个方法
主要区别在于它们去除空白字符的位置：

rstrip():
只去除字符串右侧（末尾）的空白字符（如空格、换行符等）。

strip():
同时去除字符串两端（即开头和末尾）的空白字符。

最长的句子的长度为311
这里设置截取300个字符


"""




class MyDataset:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        
        self.load()

    def load(self):
        self.data = []
        max_seq_length = self.config["max_seq_length"]
        with open(self.path, encoding="utf8") as f:
            for line in f:
                prompt, answer = line.strip().split("\t")
                prompt_encode = self.tokenizer.encode(prompt, add_special_tokens=False)
                answer_encode = self.tokenizer.encode(answer, add_special_tokens=False)
                # print(prompt_encode)  # [3241, 7599, 3031, 3409, 3409, 6820, 2923]
                # print(answer_encode)  # [3247, 7463, 3883, 5709, 5709, 3291, 5273]
                # input()
                x = [self.tokenizer.cls_token_id] + prompt_encode + [self.tokenizer.sep_token_id] + answer_encode + [self.tokenizer.sep_token_id]
                y = [-1] + len(prompt_encode) * [-1] + answer_encode + [self.tokenizer.sep_token_id] + [-1]
                # 构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
                mask = self.create_mask(len(prompt_encode), len(answer_encode))
                # padding
                x = x[:self.config["max_seq_length"]] + [0] * (self.config["max_seq_length"] - len(x))
                y = y[:self.config["max_seq_length"]] + [0] * (self.config["max_seq_length"] - len(y))
                x = torch.LongTensor(x)
                y = torch.LongTensor(y)
                # target_shape = (max_seq_length, max_seq_length)
                mask = pad_mask(mask, (max_seq_length, max_seq_length))
                self.data.append([x, mask, y])


    # 构造掩码，输入两个字符串的长度
    def create_mask(self, s1, s2):
        len_s1 = s1 + 2 # cls + sep
        len_s2 = s2 + 1 # sep
        mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
        # 遍历s1的每个token
        for i in range(len_s1):
            # s1的当前token不能看到s2的任何token
            mask[i, len_s1:] = 0  
        # 遍历s2的每个token
        for i in range(len_s2):
            # s2的当前token不能看到后面的s2 token
            mask[len_s1 + i, len_s1 + i + 1:] = 0
        return mask
    
    # def pad_mask(tensor, target_shape):
    #     # 获取输入张量和目标形状的长宽
    #     height, width = tensor.shape
    #     target_height, target_width = target_shape
    #     # 创建一个全零张量,形状为目标形状
    #     result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    #     # 计算需要填充或截断的区域
    #     h_start = 0
    #     w_start = 0
    #     h_end = min(height, target_height)
    #     w_end = min(width, target_width)
    #     # 将原始张量对应的部分填充到全零张量中
    #     result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    #     return result

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def pad_mask(tensor, target_shape):
        # 获取输入张量和目标形状的长宽
        height, width = tensor.shape
        target_height, target_width = target_shape
        # 创建一个全零张量,形状为目标形状
        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        # 计算需要填充或截断的区域
        h_start = 0
        w_start = 0
        h_end = min(height, target_height)
        w_end = min(width, target_width)
        # 将原始张量对应的部分填充到全零张量中
        result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
        return result



# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dataset = MyDataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    from config import Config
    dataset = MyDataset("./data/data.txt", Config)



