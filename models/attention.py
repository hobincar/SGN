import torch
import torch.nn as nn


class SemanticAlignment(nn.Module):
    def __init__(self, query_size, feat_size, bottleneck_size):
        super(SemanticAlignment, self).__init__()
        self.query_size = query_size
        self.feat_size = feat_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.feat_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, phr_feats, vis_feats):
        Wh = self.W(phr_feats)
        Uv = self.U(vis_feats)

        energies = self.w(torch.tanh(Wh[:, :, None, :] + Uv[:, None, :, :] + self.b)).squeeze(-1)
        weights = torch.softmax(energies, dim=2)
        aligned_vis_feats = torch.bmm(weights, vis_feats)
        semantic_group_feats = torch.cat([ phr_feats, aligned_vis_feats ], dim=2)
        return semantic_group_feats, weights, energies


class SemanticAttention(nn.Module):
    def __init__(self, query_size, key_size, bottleneck_size):
        super(SemanticAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.key_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, query, keys, values, masks=None):
        Wh = self.W(query)
        Uv = self.U(keys)
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.w(torch.tanh(Wh + Uv + self.b))
        if masks is not None:
            masks = masks[:, :, None]
            energies[masks] = -float('inf')
        weights = torch.softmax(energies, dim=1)
        weighted_feats = values * weights.expand_as(values)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats, weights, energies

