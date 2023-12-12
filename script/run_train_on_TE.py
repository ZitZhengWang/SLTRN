# 基于close look at 的加载实现；
# 20230302 16.38
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
from model.relation_module import RelationNetwork, RelationModuleBasedOnTEOnBatch
from utils.utils import Logger, increment_path, weights_init, mean_confidence_interval
from config import parser, data_dir
from data.datamgr import  SetDataManager
# ===========================解析参数=============================================================
args = parser.parse_args()

CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
VALIDATION_QUERY_NUM_PER_CLASS = args.validation_query_num_per_class

TRAIN_EPISODE = args.train_episode
VAL_EPISODE = args.val_episode
LEARNING_RATE = args.learning_rate

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
    # 记录运行代码的时间
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 获取时间

    # 每次实验更新保存结果的路径
    path_result = increment_path(args.save_dir)
    path_log = path_result / "log"
    path_checkpoint = path_result / "checkpoint"

    # 若保存的路径不存在，则生成路径
    if not os.path.exists(path_result):
        os.makedirs(path_result)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)

    # 记录实验日志，保存主要实验参数
    logfile = Logger()
    logfile.open(path_result / "logfile.txt", mode="a")
    logfile.write(f"==========实验开始时间:{start_time} ==========\n")
    logfile.write(f"实验编号记录：{path_result}\n\t")
    logfile.write(f"实验参数记录：\n\t")
    logfile.write(
        f"小样本参数：{CLASS_NUM}way-{SHOT_NUM_PER_CLASS}shot-{QUERY_NUM_PER_CLASS}train_query-{VALIDATION_QUERY_NUM_PER_CLASS}val_query\n\t")
    for arg in args._get_kwargs():
        logfile.write(f"实验参数：{arg}\n\t")

    # tensorboard训练日志
    writer = SummaryWriter(path_log)

    logfile.write("\n训练过程：")
    logfile.write("\n\tinit data_processing folders")

    # =================================加载数据部分=============================================

    base_file = data_dir[args.dataset] + 'base.json'
    val_file = data_dir[args.dataset] + 'val.json'

    train_datamgr = SetDataManager(image_size=args.image_size,
                                   n_way=CLASS_NUM,
                                   n_support=SHOT_NUM_PER_CLASS,
                                   n_query=QUERY_NUM_PER_CLASS,
                                   n_eposide=TRAIN_EPISODE
                                  )
    train_loader = train_datamgr.get_data_loader(base_file, aug=args.train_aug)

    val_datamgr = SetDataManager(image_size=args.image_size,
                                 n_way=CLASS_NUM,
                                 n_support=SHOT_NUM_PER_CLASS,
                                 n_query=VALIDATION_QUERY_NUM_PER_CLASS,
                                 n_eposide=VAL_EPISODE
                                 )
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    # if args.dataset == "miniImageNet":
    #
    # elif args.dataset == "tieredImageNet":
    #
    # elif args.dataset == "CUB":
    #
    # elif args.dataset == "Omniglot":
    #
    # else:
    #     raise ValueError("No such dataset!")

    # =================================加载模型部分================================================
    logfile.write("\n\tinit neural networks")

    # 选择主干网络
    if args.backbone == "CNNEncoder":
        # CNNEncoder 需要手动初始化；
        feature_encoder = featureEncoder(backbone=model_dict[args.backbone])
        feature_encoder.apply(weights_init)
    else:
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

    # 关系模块都需要weights_init 来初始化；
    relation_network.apply(weights_init)

    logfile.write(f"\n\t{feature_encoder}\n")
    logfile.write(f"\n\t{relation_network}")

    feature_encoder.to(device)
    relation_network.to(device)

    # =========================================加载优化器===================================================
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    # feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=args.lr_decay_interval,
    #                                    gamma=args.lr_decay_rate)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim, step_size=args.lr_decay_interval,
    #                                     gamma=args.lr_decay_rate)

    # ========================================预训练部分===================================================
    # 端点续训
    if args.use_trained_model:
        if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                SHOT_NUM_PER_CLASS) + "shot.pkl")):
            feature_encoder.load_state_dict(torch.load(
                str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                    SHOT_NUM_PER_CLASS) + "shot.pkl"), map_location='cuda:0'))
            logfile.write("\n\tload feature encoder success")

    elif args.use_my_model:
        feature_encoder.load_state_dict(torch.load(args.feature_encoder_dir))
        logfile.write("\n\tload feature encoder success")
        relation_network.load_state_dict(torch.load(args.relation_network_dir))
        logfile.write("\n\tload relation network success")

    # =========================================训练部分===========================================
    logfile.write("\n\tStart training...")

    last_accuracy = 0.0
    total_train_loss = []
    avg_train_loss = 0

    # 训练TRAIN_EPISODE个任务
    for task_num, (images, labels) in enumerate(train_loader):
        feature_encoder.train()
        relation_network.train()

        # 前向计算 # images [5,17,3,224,224]
        support_features, query_features = feature_encoder.parse_feature(images, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)

        # 计算原型
        if args.aggregation == "proto_sum" and SHOT_NUM_PER_CLASS > 1:
            support_features = support_features.view(CLASS_NUM, SHOT_NUM_PER_CLASS, -1)
            support_features = torch.sum(support_features, 1).squeeze(1)
        elif args.aggregation == "proto_mean" and SHOT_NUM_PER_CLASS > 1:
            support_features = support_features.view(CLASS_NUM, SHOT_NUM_PER_CLASS, -1)
            support_features = torch.mean(support_features, 1).squeeze(1)

        support_features_ext = support_features.unsqueeze(0).repeat(query_features.shape[0], 1, 1)  # [75,5,64]
        query_features_ext = query_features.unsqueeze(1).repeat(1, support_features.shape[0], 1)  # [75,5,64]

        relation_pairs = torch.cat((support_features_ext, query_features_ext), 2)  # [75,5,128]

        # use transformer
        # 计算关系分数；
        relations = relation_network(relation_pairs)  # 输入[75,5,128]    输入出【75,5】

        query_labels = torch.from_numpy(np.repeat(range(CLASS_NUM), QUERY_NUM_PER_CLASS))
        one_hot_labels = Variable(torch.zeros(query_features.shape[0], CLASS_NUM).scatter_(1, query_labels.view(-1, 1), 1).to(device))

        mse = nn.MSELoss().to(device)

        loss = mse(relations, one_hot_labels)

        # training
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        # torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        # torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        # feature_encoder_scheduler.step(task_num)
        # relation_network_scheduler.step(task_num)

        total_train_loss.append(loss.item())

        # 每训练100轮输出训练损失
        if (task_num + 1) % 100 == 0 or task_num == 0:
            avg_train_loss = np.mean(total_train_loss)
            logfile.write(f"\n\t\tepisode:{task_num + 1}, current_loss:{loss.item()}, avg_trainloss_per_100_episodes:{avg_train_loss}")
            total_train_loss = []

        # ===========================模型验证部分==========================================
        # 每5000轮进行验证
        if (task_num + 1) % 5000 == 0 or task_num == 0:
            logfile.write("\n\tStart testing...")

            with torch.no_grad():
                feature_encoder.eval()
                relation_network.eval()
                # 需要累计的变量，每次验证的时候置0

                accuracies = []
                total_val_loss = []

                # for i in range(VAL_EPISODE):
                for i, (images, labels) in enumerate(val_loader):
                    # 每一个任务将reward置0；
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
                    one_hot_labels = Variable(torch.zeros(query_features.shape[0], CLASS_NUM).scatter_(1, query_labels.view(-1, 1), 1).to(device))

                    mse = nn.MSELoss().to(device)

                    val_loss = mse(relations, one_hot_labels)
                    total_val_loss.append(val_loss.item())

                    _, predict_labels = torch.max(relations.data, 1)

                    rewards = [1 if predict_labels[j] == query_labels[j] else 0 for j in range(query_features.shape[0])]

                    total_rewards += np.sum(rewards)
                    counter += query_features.shape[0]
                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                valid_accuracy, h = mean_confidence_interval(accuracies)
                logfile.write(f"\n\t\ttest accuracy:{valid_accuracy}, h:{h}, avg_val_loss:{np.mean(total_val_loss)}")

                if valid_accuracy > last_accuracy:
                    torch.save(feature_encoder.state_dict(),
                               path_result / f"feature_encoder_{CLASS_NUM}way_{SHOT_NUM_PER_CLASS}shot.pkl")
                    torch.save(relation_network.state_dict(),
                               path_result / f"relation_network_{CLASS_NUM}way_{SHOT_NUM_PER_CLASS}shot.pkl")

                    logfile.write(f"\n\tSave networks for episode:{task_num+1}")
                    last_accuracy = valid_accuracy

                torch.save(feature_encoder.state_dict(),
                           path_checkpoint / f"feature_encoder_{(task_num + 1) / 5000}.pkl")
                torch.save(relation_network.state_dict(),
                           path_checkpoint / f"relation_network_{(task_num + 1) / 5000}.pkl")

        # 一轮训练（一个任务）以后，在tensorboard中记录损失和准确率
        writer.add_scalars('Loss', {'TrainLoss': loss.item(), f"avg_trainloss_per_100_episodes": avg_train_loss, 'avg_ValLoss': np.mean(total_val_loss)}, task_num+1)
        writer.add_scalar('mAP', valid_accuracy, task_num+1)

    writer.close()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 获取结束时间
    logfile.write(f"\nEnd of all training. The highest accuracy is {last_accuracy}")
    logfile.write(f"\n==========实验结束时间：{end_time}  ==========")


if __name__ == '__main__':
    main()


