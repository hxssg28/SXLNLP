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
from predict import Predictor
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
    # print(cuda_flag)
    # input()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 选择优化器
    optimizer = choose_optimizer(config, model)
    # 测试预测效果
    predictor = Predictor(config, model, logger)
    # 开始训练
    for epoch in range(config["epochs"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []  # 记录训练的损失值
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            x, mask, y = batch_data
            loss = model(x, mask, y)   # 计算loss
            loss.backward()            # 计算梯度
            optimizer.step()           # 更新权重
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        # 预测
        output1 = predictor.predict("世上功名皆看淡", epoch)
        print(output1)
        output2 = predictor.predict("雪岭红梅独一品", epoch)
        print(output2)
        print("------------------------------------------------------------------------------")
    model_path = os.path.join(config["model_path"], "generate_model.pth")
    torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    main(Config)





