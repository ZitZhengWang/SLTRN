# 基于close look at 的加载实现；
# 20230303 16.38
# ========================================导入相关库===========================================
# 导入官方库
import os
import random

import torch
import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter

# 导入自建库
from model import backbone
from model.my_featureEncoder import featureEncoder
from model.relation_module import RelationModuleBasedOnTEOnBatch, RelationNetwork
from utils.utils import Logger, increment_path, weights_init, mean_confidence_interval
from config import parser, data_dir
from data.datamgr import SetDataManager

# ===========================解析参数=============================================================

args = parser.parse_args()

CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
VALIDATION_QUERY_NUM_PER_CLASS = args.validation_query_num_per_class

EPOCHS = args.epochs
TEST_EPISODE = args.test_episode

# 解析地址
root_dir = args.load_dir
logfile_dir = root_dir + "/logfile.txt"
feature_encoder_dir = root_dir + f"/feature_encoder_{CLASS_NUM}way_{SHOT_NUM_PER_CLASS}shot.pkl"
relation_network_dir = root_dir + f"/relation_network_{CLASS_NUM}way_{SHOT_NUM_PER_CLASS}shot.pkl"


def main():
    device = torch.device(args.device)

    model_dict = dict(
        CNNEncoder=backbone.CNNEncoder(use_GAP=True),
        Conv4=backbone.Conv4(),
        Conv4S=backbone.Conv4S(),
        Conv6=backbone.Conv6(),
        ResNet10=backbone.ResNet10(),
        ResNet18=backbone.ResNet18(),
        ResNet34=backbone.ResNet34(),
        ResNet50=backbone.ResNet50(),
        ResNet101=backbone.ResNet101())

    # ======================================实验记录部分=========================================
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 获取时间

    # 记录实验日志
    logfile = Logger()
    logfile.open(logfile_dir, mode="a")
    logfile.write(f"\n\n==========测试开始时间:{start_time} ==========\n")
    logfile.write(f"测试时参数记录：\n\t")
    logfile.write(
        f"小样本参数：{CLASS_NUM}way-{SHOT_NUM_PER_CLASS}shot-{QUERY_NUM_PER_CLASS}train_query-{VALIDATION_QUERY_NUM_PER_CLASS}test_query\n\t")
    for arg in args._get_kwargs():
        logfile.write(f"实验参数：{arg}\n\t")

    # =================================加载数据部分=============================================
    noval_file = data_dir[args.dataset] + 'novel.json'

    test_datamgr = SetDataManager(image_size=args.image_size,
                                  n_way=CLASS_NUM,
                                  n_support=SHOT_NUM_PER_CLASS,
                                  n_query=VALIDATION_QUERY_NUM_PER_CLASS,
                                  n_eposide=TEST_EPISODE
                                  )
    test_loader = test_datamgr.get_data_loader(noval_file, aug=False)

    # =================================加载模型部分================================================
    logfile.write("\n\tinit neural networks")

    # 选择主干网络
    feature_encoder = featureEncoder(backbone=model_dict[args.backbone])

    # 构建关系模块并初始化
    if args.RelationModule == "RelationModuleBasedOnTEOnBatch":
        relation_network = RelationModuleBasedOnTEOnBatch(input_dim=args.transformerlayer_input_dim,
                                                          head_num=args.head_num,
                                                          feedforeard_dim=args.transformerlayer_feedforward_dim,
                                                          norm_method=args.norm_method,
                                                          transformerlayer_num=args.transformerlayer_num,
                                                          hidden_size=args.hidden_size)
    elif args.RelationModule == "relation_network":
        relation_network = RelationNetwork()
    else:
        raise ValueError("UnKnown Relation Moudel!")

    logfile.write(f"\n\t{feature_encoder}\n")
    logfile.write(f"\n\t{relation_network}")

    feature_encoder.to(device)
    relation_network.to(device)

    # ========================================加载模型部分===================================================
    # 端点续训
    if args.use_trained_model:
        if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SHOT_NUM_PER_CLASS) + "shot.pkl")):
            feature_encoder.load_state_dict(torch.load(
                str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                    SHOT_NUM_PER_CLASS) + "shot.pkl"), map_location='cuda:0'))
            logfile.write("\n\tload feature encoder success")

    elif args.use_my_model:
        feature_encoder.load_state_dict(torch.load(feature_encoder_dir))
        logfile.write("\n\tload feature encoder success")
        relation_network.load_state_dict(torch.load(relation_network_dir))
        logfile.write("\n\tload relation network success")

    # ===================================测试部分==========================================
    # 与验证部分几乎一致
    logfile.write("\n\tStart testing...")

    # 进行10次测试，并计算平均值；
    total_accuracy = 0.0
    for epoch in range(EPOCHS):
        # 进行一次测试，TEST_EPISODE个任务取平均值
        print("\n\tTesting...")
        feature_encoder.eval()
        relation_network.eval()

        # 需要累计的变量，每次测试的时候置0

        accuracies = []
        for i, (images, labels) in enumerate(test_loader):
            total_rewards = 0
            counter = 0
            # 前向计算
            support_features, query_features = feature_encoder.parse_feature(images, CLASS_NUM, SHOT_NUM_PER_CLASS,
                                                                             VALIDATION_QUERY_NUM_PER_CLASS)

            # 计算原型
            if args.aggregation == "proto_sum" and SHOT_NUM_PER_CLASS > 1:
                support_features = support_features.view(CLASS_NUM, SHOT_NUM_PER_CLASS, -1)
                support_features = torch.sum(support_features, 1).squeeze(1)
            elif args.aggregation == "proto_mean" and SHOT_NUM_PER_CLASS > 1:
                support_features = support_features.view(CLASS_NUM, SHOT_NUM_PER_CLASS, -1)
                support_features = torch.mean(support_features, 1).squeeze(1)

            support_features_ext = support_features.unsqueeze(0).repeat(query_features.shape[0], 1,
                                                                        1)  # [75,5,64]
            query_features_ext = query_features.unsqueeze(1).repeat(1, support_features.shape[0],
                                                                    1)  # [75,5,64]

            relation_pairs = torch.cat((support_features_ext, query_features_ext), 2)  # [75,5,128]

            relations = relation_network(relation_pairs)  # 输入[75,5,128]    输入出【75,5】

            query_labels = torch.from_numpy(np.repeat(range(CLASS_NUM), VALIDATION_QUERY_NUM_PER_CLASS))

            _, predict_labels = torch.max(relations.data, 1)

            rewards = [1 if predict_labels[j] == query_labels[j] else 0 for j in range(query_features.shape[0])]

            total_rewards += np.sum(rewards)
            counter += query_features.shape[0]
            accuracy = total_rewards / 1.0 / counter
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        logfile.write(f"\n\t\ttest accuracy:{test_accuracy}, h:{h}")

        total_accuracy += test_accuracy

    # 计算十次测试平均结果
    logfile.write(f"\n\tAverage accuracy of 10 tests::{total_accuracy / EPOCHS}")

    end_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())  # 获取结束时间
    logfile.write(f"\n==========测试结束时间：{end_time}  ===============")

if __name__ == '__main__':
    main()
