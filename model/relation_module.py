
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import torch.nn.functional as F
from model.my_transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
# from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

class RelationModuleBasedOnTFOnBatch(Module):
    def __init__(self, input_dim, head_num, feedforeard_dim, transformerlayer_num, norm_method, hidden_size=8):
        super(RelationModuleBasedOnTFOnBatch, self).__init__()
        self.transformerEncoderlayer = TransformerEncoderLayer(d_model=input_dim,
                                                               nhead=head_num,
                                                               dim_feedforward=feedforeard_dim,
                                                               norm_method=norm_method,
                                                               batch_first=True)
        self.encoder = TransformerEncoder(self.transformerEncoderlayer, transformerlayer_num)

        self.transformerDecoderlayer = TransformerDecoderLayer(d_model=input_dim,
                                                               nhead=head_num,
                                                               dim_feedforward=feedforeard_dim,
                                                               norm_method=norm_method,
                                                               batch_first=True)
        self.decoder = TransformerDecoder(self.transformerDecoderlayer, transformerlayer_num)

        self.fc1 = Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, support, query, tgt_mask_flag, mask_type, memory_mask_flag=False):
        # 输入 support set 样本维度 [75,5,128]
        # 输入 query 样本维度【75，5,128】
        support_shape = support.shape
        query_shape = query.shape

        if tgt_mask_flag and memory_mask_flag:
            if mask_type == "square_subsequent":
                memory_mask = self.generate_square_subsequent_mask(support_shape[1]).cuda()
                query_mask = self.generate_square_subsequent_mask(query_shape[1]).cuda()
            elif mask_type=="diagonal":
                memory_mask = self.generate_diagonal_mask(support_shape[1]).cuda()
                query_mask = self.generate_diagonal_mask(query_shape[1]).cuda()

            memory = self.encoder(support)  # encoder层输出与输入的尺寸一致为：【75,5，64】
            out = self.decoder(query, memory, memory_mask=memory_mask, tgt_mask=query_mask)  # decoder输出与输入的quey尺寸一致：【75,5，64】
        elif tgt_mask_flag:
            if mask_type=="square_subsequent":
                query_mask = self.generate_square_subsequent_mask(query_shape[1]).cuda()
            elif mask_type=="diagonal":
                query_mask = self.generate_diagonal_mask(query_shape[1]).cuda()

            memory = self.encoder(support)   # encoder层输出与输入的尺寸一致为：【75,5，128】
            out = self.decoder(query, memory, tgt_mask=query_mask)  # decoder输出与输入的quey尺寸一致：【75,5，128】
        else:
            memory = self.encoder(support)  # encoder层输出与输入的尺寸一致为：【75,5，128】
            out = self.decoder(query, memory)  # decoder输出与输入的quey尺寸一致：【75,5，128】

        # 先将向量变成标量
        out = F.relu(self.fc1(out))  # 【75,5，128】-【75,5，8】
        out = F.sigmoid(self.fc2(out))  # 【75,5，1】
        out = out.squeeze()     # 【75，5】
        return out  # 【75，5】

    @staticmethod
    def generate_square_subsequent_mask(sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    @staticmethod
    def generate_diagonal_mask(sz: int):
        """The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0)."""
        # 输入为query的长度
        mask = torch.full((sz, sz), float('-inf'))  # 全 -inf 的矩阵
        index = torch.arange(0, sz).view(-1, 1)  # 生成二维的索引
        mask = mask.scatter_(1, index, 0)   # 第一个参数表示维度，第二个参数表示索引，第三个参数表示指定的值；
        return mask


# ==================================================================================================
# 原始关系模块
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork
    """
    def __init__(self,input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


class RelationModuleBasedOnTFOnFeature(Module):
    def __init__(self, input_dim, head_num, feedforeard_dim, transformerlayer_num, norm_method, hidden_size=8):
        super(RelationModuleBasedOnTFOnFeature, self).__init__()
        self.transformerEncoderlayer = TransformerEncoderLayer(d_model=input_dim,
                                                               nhead=head_num,
                                                               dim_feedforward=feedforeard_dim,
                                                               norm_method=norm_method,
                                                               batch_first=True)
        self.encoder = TransformerEncoder(self.transformerEncoderlayer, transformerlayer_num)

        self.transformerDecoderlayer = TransformerDecoderLayer(d_model=input_dim,
                                                               nhead=head_num,
                                                               dim_feedforward=feedforeard_dim,
                                                               norm_method=norm_method,
                                                               batch_first=True)
        self.decoder = TransformerDecoder(self.transformerDecoderlayer, transformerlayer_num)

        self.fc1 = Linear(input_dim, 1)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, support, query, tgt_mask_flag, mask_type, memory_mask_flag=False):
        # 输入 support set 样本维度 [75,5,128]
        # 输入 query 样本维度【75，5,128】
        support_shape = support.shape
        query_shape = query.shape

        if tgt_mask_flag and memory_mask_flag:
            if mask_type == "square_subsequent":
                memory_mask = self.generate_square_subsequent_mask(support_shape[1]).cuda()
                query_mask = self.generate_square_subsequent_mask(query_shape[1]).cuda()
            elif mask_type=="diagonal":
                memory_mask = self.generate_diagonal_mask(support_shape[1]).cuda()
                query_mask = self.generate_diagonal_mask(query_shape[1]).cuda()

            memory = self.encoder(support)  # encoder层输出与输入的尺寸一致为：【75,5，64】
            out = self.decoder(query, memory, memory_mask=memory_mask, tgt_mask=query_mask)  # decoder输出与输入的quey尺寸一致：【75,5，64】
        elif tgt_mask_flag:
            if mask_type=="square_subsequent":
                query_mask = self.generate_square_subsequent_mask(query_shape[1]).cuda()
            elif mask_type=="diagonal":
                query_mask = self.generate_diagonal_mask(query_shape[1]).cuda()

            memory = self.encoder(support)   # encoder层输出与输入的尺寸一致为：【75,5，128】
            out = self.decoder(query, memory, tgt_mask=query_mask)  # decoder输出与输入的quey尺寸一致：【75,5，128】
        else:
            memory = self.encoder(support)  # encoder层输出与输入的尺寸一致为：【75,5，128】
            out = self.decoder(query, memory)  # decoder输出与输入的quey尺寸一致：【75,5，128】

        # 先将向量变成标量
        out = F.relu(self.fc1(out))  # 【375，128， 361】-【375，128， 1】
        out = out.squeeze()  # 【375，128】
        out = F.sigmoid(self.fc2(out))  # 【375，1】
        out = out.reshape(-1, 5)
        return out  # 【75，5】

    @staticmethod
    def generate_square_subsequent_mask(sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    @staticmethod
    def generate_diagonal_mask(sz: int):
        """The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0)."""
        # 输入为query的长度
        mask = torch.full((sz, sz), float('-inf'))  # 全 -inf 的矩阵
        index = torch.arange(0, sz).view(-1, 1)  # 生成二维的索引
        mask = mask.scatter_(1, index, 0)   # 第一个参数表示维度，第二个参数表示索引，第三个参数表示指定的值；
        return mask


class RelationModuleBasedOnTEOnBatch(Module):
    def __init__(self, input_dim, head_num, feedforeard_dim, transformerlayer_num, norm_method, hidden_size=8):
        super(RelationModuleBasedOnTEOnBatch, self).__init__()
        self.transformerEncoderlayer = TransformerEncoderLayer(d_model=input_dim,
                                                               nhead=head_num,
                                                               dim_feedforward=feedforeard_dim,
                                                               norm_method=norm_method,
                                                               batch_first=True)
        self.encoder = TransformerEncoder(self.transformerEncoderlayer, transformerlayer_num)

        self.fc1 = Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, input):
        # 输入 support set 样本维度 [75,5,128]
        # 输入 query 样本维度【75，5,128】

        out = self.encoder(input)  # encoder层输出与输入的尺寸一致为：【75,5，128】

        # 先将向量变成标量
        out = F.relu(self.fc1(out))  # 【75,5，128】-【75,5，8】
        out = F.sigmoid(self.fc2(out))  # 【75,5，1】
        out = out.squeeze()     # 【75，5】
        return out  # 【75，5】