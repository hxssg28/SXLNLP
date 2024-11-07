# -*- coding: utf-8 -*-

import json
import re
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载

数据的格式
[[[一个 title 对应的序列], [label]], [[一个 title 对应的序列], [label]], [[一个 title 对应的序列], [label]]...]

"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label) # 0 或 1
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)   # 记录字表的大小
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if line.startswith("0,"):  # 这段代码用于检查当前行是否以字符串 "0," 开头。如果是，则执行相应的操作。
                    label = 0
                elif line.startswith("1,"): # 这段代码用于检查当前行是否以字符串 "1," 开头。如果是，则执行相应的操作。
                    label = 1
                else:
                    continue
                review = line[2:].strip()  # 因为每一行格式是 1,味道很正点！餐具很好用！送餐速度快！line[0] 对应标签，line[1]对应逗号，所以从2开始取。
                review = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", review)  # 去掉一些标点

                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(review)  # 传入的是一个句子,将句子转化为序列
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return


    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 加载字表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()  # token = line.strip() 去除每行内容的空白字符（如空格、换行符等）。
            token_dict[token] = index + 1 # 0留给padding位置，所以从1开始
    return token_dict


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(Config["valid_data_path"], Config)
    print(dg[1])

