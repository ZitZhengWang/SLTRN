import argparse

# 定义一个字典，存放数据集的相对路径
data_dir = {
    'CUB':          './filelist/CUB/',
    'miniImageNet': './filelist/miniImageNet/',

}

parser = argparse.ArgumentParser(description="SLTRN")
# 小样本参数
parser.add_argument("--class_num", type=int, default=5)
parser.add_argument("--shot_num_per_class", type=int, default=1, help="支持集样本数，对于训练和测试都是一样的")
parser.add_argument("--query_num_per_class", type=int, default=16, help="训练的query数量")
parser.add_argument("--validation_query_num_per_class", type=int, default=16, help="验证和测试的query数量")

# 模型参数
parser.add_argument("--backbone", type=str, default="CNNEncoder", help="optional:{CNNEncoder|Conv4|ResNet10|ResNet18} ")
parser.add_argument("--RelationModule", type=str, default="RelationModuleBasedOnTFOnBatch", help="optional:{RelationModuleBasedOnTFOnBatch|"
                                                                                                 "RelationNetwork|"
                                                                                                 "RelationModuleBasedOnTFOnFeature"
                                                                                                 "RelationModuleBasedOnTEOnBatch}")
parser.add_argument("--aggregation", type=str, default="proto_mean", help="optional:{proto_sum/proto_mean}")

# transformer参数
parser.add_argument("--transformerlayer_input_dim", type=int, default=128, help="The input vector dimension of the transformer layer is equal to twice the output vector dimension of the CNN")
parser.add_argument("--head_num", type=int, default=8, help="the number of self attention head")
parser.add_argument("--transformerlayer_feedforward_dim", type=int, default=128*4)
parser.add_argument("--transformerlayer_num", type=int, default=2, help="the number of transformerEncoder's transformer layer")
parser.add_argument("--norm_method", type=str, default="LayerNorm", help="optional:{LayerNorm, BatchNorm1d}, the norm method in transfomer layer")
parser.add_argument("--hidden_size", type=int, default=8, help="The num of fully-connect layer mid unit ")
parser.add_argument("--src_mask", action="store_true", default=False, help="")
parser.add_argument("--memory_mask", action="store_true", default=False, help="")
parser.add_argument("--tgt_mask", action="store_false", default=True, help="")
parser.add_argument("--mask_type", type=str, default="diagonal", help="optional:{square_subsequent \ diagonal }")

# 训练参数
parser.add_argument("--dataset", type=str, default="CUB", help="optional:{ miniImageNet | CUB }")
parser.add_argument("--train_aug", action="store_false", default=True, help="Using data augument when training")
parser.add_argument("--image_size", type=int, default=84, help="optional:{84,224}")

parser.add_argument("--train_episode", type=int, default=700000)
parser.add_argument("--val_episode", type=int, default=600)
parser.add_argument("--learning_rate", type = float, default=0.0001)
# parser.add_argument("--lr_decay_rate", type = float, default=0.5, help="")
# parser.add_argument("--lr_decay_interval", type=int, default=50000, help="")
parser.add_argument("--device", type=str, default="cuda:0", help="optional:{cuda:0|cpu}")

# checkpoint相关参数
parser.add_argument("--use_trained_model", action="store_true", default=False, help="use the model trained by author")
parser.add_argument("--checkpoint_dir", type=str, default="Results/exp_on_CUB_on_backbone_server43")
parser.add_argument("--checkpont_epoches", type=int, default=405000)

# 结果保存路径参数
parser.add_argument("--save_dir", type=str, default="./Results/exp")

# 测试专用参数
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--test_episode", type=int, default=600)
parser.add_argument("--use_my_model", action="store_true", default=False, help="use the model trained by myself")
parser.add_argument("--load_dir", type=str, default="Results/exp_on_miniImageNet12")



