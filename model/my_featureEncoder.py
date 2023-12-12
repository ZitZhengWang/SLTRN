from torch import nn
from torch.autograd import Variable


class featureEncoder(nn.Module):
    def __init__(self, backbone):
        super(featureEncoder, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        return out  # 64

    def parse_feature(self, x, n_way, n_support, n_query, is_feature=False):
        x = Variable(x.cuda())
        # images [5,17,3,224,224]
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
            z_all       = self.forward(x)
            z_all       = z_all.view(n_way, n_support + n_query, -1)

        z_support   = z_all[:, :n_support, :]
        z_query     = z_all[:, n_support:, :]

        z_support = z_support.contiguous().view(n_way*n_support, -1)
        z_query = z_query.reshape(n_way*n_query, -1)

        return z_support, z_query

    def parse_feature_map(self, x, n_way, n_support, n_query, is_feature=False):
        x = Variable(x.cuda())
        # images [5,17,3,224,224]
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
            z_all       = self.forward(x)   # [B,C,H,W]
            z_all       = z_all.view(n_way, n_support + n_query, z_all.size()[1], -1)

        z_support   = z_all[:, :n_support, :, :]
        z_query     = z_all[:, n_support:, :, :]

        return z_support, z_query