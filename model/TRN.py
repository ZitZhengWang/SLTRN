import torch
from torch import nn


class TRN(nn.Module):
    def __init__(self, FEM, TRM):
        super().__init__()
        self.FEM = FEM
        self.TRM = TRM

    def forward(self, images):

        support_features, query_features = self.FEM.parse_feature(images, CLASS_NUM, SHOT_NUM_PER_CLASS,
                                                                         VALIDATION_QUERY_NUM_PER_CLASS)
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

        relations = self.TRM(relation_pairs, relation_pairs, args.tgt_mask, args.mask_type)  # 输入[75,5,128]    输入出【75,5】

        return relations