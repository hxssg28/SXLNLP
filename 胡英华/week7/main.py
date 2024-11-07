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
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def split_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # discard the first line "label review"  以列表的形式将一行行数据存储起来‘
        random.shuffle(lines)  # 打乱数据集
        len_lines = len(lines)
        len_train = int(0.8 * len_lines)  # 训练集占比

        train_lines = lines[:len_train]
        test_lines = lines[len_train:]

    with open('train_data.txt', 'w', encoding='utf8') as f_train:
            f_train.writelines(train_lines)
    Config["train_data_path"] = 'train_data.txt'

    with open('valid_data.txt', 'w', encoding='utf8') as f_valid:
        f_valid.writelines(test_lines)
    Config["valid_data_path"] = 'valid_data.txt'
    

def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 是否用GPU进行训练
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 选择优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []  # 记录训练的损失值
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            # print(batch_data)
            # input()
            optimizer.zero_grad()
            input_ids, labels = batch_data   # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)  # 计算损失
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item()) # loss.item() 是 PyTorch 中用于从损失张量（通常是一个标量张量）中提取标量值的方法。
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        if acc > 0.997:
            model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
            torch.save(model.state_dict(), model_path)
    return acc


if __name__ == "__main__":
    split_file(Config["data_path"])

    with open('model_result.csv', 'w', newline='') as csvfile:
        fieldnames = ['model_type', 'accuracy', 'learning_rate', 'pooling_style', 'batch_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model in ["bert", "fast_text", "lstm", "gru", "rnn", "gated_cnn", "gated_cnn", "rcnn"]:
        # for model in ["bert"]:
            Config["model_type"] = model
            for lr in [1e-3, 1e-4]:
                Config["learning_rate"] = lr
                for batch_size in [64, 128]:
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config))
                        writer.writerow({'model_type': model, 'accuracy': main(Config), 'learning_rate': lr, 'pooling_style': pooling_style, 'batch_size': batch_size})
                        # print("最后一轮准确率：", main(Config), "当前配置：", Config)

