import torch
import torch.nn as nn
import torch.nn.functional as F
from .trans_utils.position_encoding import build_position_encoding
from .trans_utils.transformer import build_transformer


class TopKMaxPooling(nn.Module):
    def __init__(self, kmax=1.0):
        super(TopKMaxPooling, self).__init__()
        self.kmax = kmax

    @staticmethod
    def get_positive_k(k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)
        n = h * w  # number of regions
        kmax = self.get_positive_k(self.kmax, n)
        sorted, indices = torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True)
        region_max = sorted.narrow(2, 0, kmax)
        output = region_max.sum(2).div_(kmax)
        return output.view(batch_size, num_channels)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ')'


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphConvolution, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, adj, nodes):
        nodes = torch.matmul(nodes, adj)
        nodes = self.relu(nodes)
        nodes = self.weight(nodes)
        nodes = self.relu(nodes)
        return nodes


class TDRG(nn.Module):
    def __init__(self, model, num_classes):
        super(TDRG, self).__init__()
        # backbone
        self.layer1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            # model.layer2,
            # model.layer3,
            # model.layer4,
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.backbone = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # hyper-parameters
        self.num_classes = num_classes
        self.in_planes = 2048
        self.transformer_dim = 512
        self.gcn_dim = 512
        self.num_queries = 1
        self.n_head = 4
        self.num_encoder_layers = 3
        self.num_decoder_layers = 0

        # transformer
        self.transform_14 = nn.Conv2d(self.in_planes, self.transformer_dim, 1)
        self.transform_28 = nn.Conv2d(self.in_planes // 2, self.transformer_dim, 1)
        self.transform_7 = nn.Conv2d(self.in_planes, self.transformer_dim, 3, stride=2)

        self.query_embed = nn.Embedding(self.num_queries, self.transformer_dim)
        self.positional_embedding = build_position_encoding(hidden_dim=self.transformer_dim, mode='learned')
        self.transformer = build_transformer(d_model=self.transformer_dim, nhead=self.n_head,
                                             num_encoder_layers=self.num_encoder_layers,
                                             num_decoder_layers=self.num_decoder_layers)

        self.kmp = TopKMaxPooling(kmax=0.05)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GAP1d = nn.AdaptiveAvgPool1d(1)

        self.trans_classifier = nn.Linear(self.transformer_dim * 3, self.num_classes)

        # GCN
        self.constraint_classifier = nn.Conv2d(self.in_planes, num_classes, (1, 1), bias=False)

        self.guidance_transform = nn.Conv1d(self.transformer_dim, self.transformer_dim, 1)
        self.guidance_conv = nn.Conv1d(self.transformer_dim * 3, self.transformer_dim * 3, 1)
        self.guidance_bn = nn.BatchNorm1d(self.transformer_dim * 3)
        self.relu = nn.LeakyReLU(0.2)
        self.gcn_dim_transform = nn.Conv2d(self.in_planes, self.gcn_dim, (1, 1))

        self.matrix_transform = nn.Conv1d(self.gcn_dim + self.transformer_dim * 4, self.num_classes, 1)

        self.forward_gcn = GraphConvolution(self.transformer_dim+self.gcn_dim, self.transformer_dim+self.gcn_dim)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.gcn_classifier = nn.Conv1d(self.transformer_dim + self.gcn_dim, self.num_classes, 1)

    def forward_backbone(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4

    @staticmethod
    def cross_scale_attention(x3, x4, x5):
        h3, h4, h5 = x3.shape[2], x4.shape[2], x5.shape[2]
        h_max = max(h3, h4, h5)
        x3 = F.interpolate(x3, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h_max, h_max), mode='bilinear', align_corners=True)

        mul = x3 * x4 * x5
        x3 = x3 + mul
        x4 = x4 + mul
        x5 = x5 + mul

        x3 = F.interpolate(x3, size=(h3, h3), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h4, h4), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h5, h5), mode='bilinear', align_corners=True)
        return x3, x4, x5

    def forward_transformer(self, x3, x4):
        # cross scale attention
        x5 = self.transform_7(x4)
        x4 = self.transform_14(x4)
        x3 = self.transform_28(x3)

        x3, x4, x5 = self.cross_scale_attention(x3, x4, x5)

        # transformer encoder
        mask3 = torch.zeros_like(x3[:, 0, :, :], dtype=torch.bool).cuda()
        mask4 = torch.zeros_like(x4[:, 0, :, :], dtype=torch.bool).cuda()
        mask5 = torch.zeros_like(x5[:, 0, :, :], dtype=torch.bool).cuda()

        pos3 = self.positional_embedding(x3)
        pos4 = self.positional_embedding(x4)
        pos5 = self.positional_embedding(x5)

        _, feat3 = self.transformer(x3, mask3, self.query_embed.weight, pos3)
        _, feat4 = self.transformer(x4, mask4, self.query_embed.weight, pos4)
        _, feat5 = self.transformer(x5, mask5, self.query_embed.weight, pos5)

        # f3 f4 f5: structural guidance
        f3 = feat3.view(feat3.shape[0], feat3.shape[1], -1).detach()
        f4 = feat4.view(feat4.shape[0], feat4.shape[1], -1).detach()
        f5 = feat5.view(feat5.shape[0], feat5.shape[1], -1).detach()

        feat3 = self.GMP(feat3).view(feat3.shape[0], -1)
        feat4 = self.GMP(feat4).view(feat4.shape[0], -1)
        feat5 = self.GMP(feat5).view(feat5.shape[0], -1)

        feat = torch.cat((feat3, feat4, feat5), dim=1)
        feat = self.trans_classifier(feat)

        return f3, f4, f5, feat

    def forward_constraint(self, x):
        activations = self.constraint_classifier(x)
        out = self.kmp(activations)
        return out

    def build_nodes(self, x, f4):
        mask = self.constraint_classifier(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.gcn_dim_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        v_g = torch.matmul(x, mask)

        v_t = torch.matmul(f4, mask)
        v_t = v_t.detach()
        v_t = self.guidance_transform(v_t)
        nodes = torch.cat((v_g, v_t), dim=1)
        return nodes

    def build_joint_correlation_matrix(self, f3, f4, f5, x):
        f4 = self.GAP1d(f4)
        f3 = self.GAP1d(f3)
        f5 = self.GAP1d(f5)
        trans_guid = torch.cat((f3, f4, f5), dim=1)

        trans_guid = self.guidance_conv(trans_guid)
        trans_guid = self.guidance_bn(trans_guid)
        trans_guid = self.relu(trans_guid)
        trans_guid = trans_guid.expand(trans_guid.size(0), trans_guid.size(1), x.size(2))

        x = torch.cat((trans_guid, x), dim=1)
        joint_correlation = self.matrix_transform(x)
        joint_correlation = torch.sigmoid(joint_correlation)
        return joint_correlation

    def forward(self, x):
        x2, x3, x4 = self.forward_backbone(x)

        # structural relation
        f3, f4, f5, out_trans = self.forward_transformer(x3, x4)

        # semantic relation
        # semantic-aware constraints
        out_sac = self.forward_constraint(x4)
        # graph nodes
        V = self.build_nodes(x4, f4)
        # print('V', V.shape)
        # joint correlation
        A_s = self.build_joint_correlation_matrix(f3, f4, f5, V)
        G = self.forward_gcn(A_s, V) + V
        out_gcn = self.gcn_classifier(G)
        mask_mat = self.mask_mat.detach()
        out_gcn = (out_gcn * mask_mat).sum(-1)

        return out_trans, out_gcn, out_sac

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.backbone.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

