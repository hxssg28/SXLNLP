# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import csv
import logging
from config import Config
from model import TorchModel, choose_optimizer
from transformers import BertTokenizer


class Predictor:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # self.model = TorchModel(config)
        # self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        
    def predict(self, input_text, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        input_ids = self.tokenizer.encode(input_text)
        with torch.no_grad():
            while len(input_ids) <= 30:
                x = torch.LongTensor([input_ids])
                if torch.cuda.is_available():
                    x = x.cuda()
                pred_probability = self.model(x)[0][-1]
                target_index = int(torch.argmax(pred_probability))
                input_ids.append(target_index)
        return self.tokenizer.decode(input_ids)
        
        
