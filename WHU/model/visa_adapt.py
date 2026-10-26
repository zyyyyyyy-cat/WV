"""Modality-conditioned ETGM and query-centered DLFM for UAD.

The shared expert input, selected patch values and graph adjacency have
different stop-gradient boundaries. Keep them explicit here.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from .backbones.vit_pytorch import Attention


class PostAttention(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.attn = Attention(dim, num_heads=heads, qkv_bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class QueryExpert(nn.Module):
    def __init__(self, dim, queries, heads):
        super().__init__()
        self.prompt = nn.Parameter(torch.empty(1, queries, dim))
        nn.init.trunc_normal_(self.prompt, std=.02)
        self.condition = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.transform = PostAttention(dim, heads, 2.)

    def forward(self, feature):
        x = self.norm(self.prompt + self.condition(feature).unsqueeze(1))
        return self.transform(x)


class ExpertPool(nn.Module):
    def __init__(self, dim, experts, top_k, queries, heads):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError('TOP_K must lie between 1 and EXPERTS')
        self.gate = nn.Linear(dim, experts)
        self.experts = nn.ModuleList(QueryExpert(dim, queries, heads) for _ in range(experts))
        self.top_k = top_k
        self.queries = queries
        self.dim = dim

    def forward(self, feature):
        if feature.shape[0] == 0:
            return feature.new_zeros((0, self.queries, self.dim)), feature.new_zeros(())
        logits = self.gate(feature)
        chosen_logits, chosen = logits.topk(self.top_k, dim=1)
        weights = chosen_logits.softmax(dim=1)
        probabilities = logits.softmax(dim=1).mean(dim=0)
        balance = len(self.experts) * probabilities.square().sum()
        result = feature.new_zeros((feature.shape[0], self.queries, self.dim))
        for expert_id, expert in enumerate(self.experts):
            # Each selected expert is executed once, regardless of its Top-K slot.
            batch_ids, slots = (chosen == expert_id).nonzero(as_tuple=True)
            if batch_ids.numel():
                value = expert(feature[batch_ids]) * weights[batch_ids, slots, None, None]
                result = result.index_add(0, batch_ids, value)
        return result, balance


class ETGM(nn.Module):
    def __init__(self, dim=768, experts=8, top_k=2, queries=16, heads=12):
        super().__init__()
        self.shared = ExpertPool(dim, experts, top_k, queries, heads)
        self.rgb = ExpertPool(dim, experts, top_k, queries, heads)
        self.ir = ExpertPool(dim, experts, top_k, queries, heads)

    def forward(self, shared_feature, specific_feature, modality_ids):
        # Only the input to the shared pool is detached. Its gate and experts train.
        shared, shared_balance = self.shared(shared_feature.detach())
        specific = torch.zeros_like(shared)
        total = shared_balance * shared_feature.shape[0]
        count = shared_feature.shape[0]
        for mask, pool in ((modality_ids == 0, self.rgb), (modality_ids != 0, self.ir)):
            ids = mask.nonzero(as_tuple=True)[0]
            if ids.numel():
                value, balance = pool(specific_feature[ids])
                specific = specific.index_add(0, ids, value)
                total = total + balance * ids.numel()
                count += ids.numel()
        return shared, specific, total / count


class QueryGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.uniform_(self.weight, -dim ** -.5, dim ** -.5)

    def forward(self, queries, patches):
        batch, query_count, dim = queries.shape
        nodes = torch.cat((queries.unsqueeze(2),
                           patches.detach().unsqueeze(1).expand(-1, query_count, -1, -1)), dim=2)
        # FP32 adjacency and propagation even inside the outer AMP context.
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            with torch.no_grad():
                normalized = F.normalize(nodes.float(), dim=-1)
                adjacency = (normalized @ normalized.transpose(-1, -2)).clamp_min_(0)
                identity = torch.eye(adjacency.shape[-1], device=nodes.device, dtype=adjacency.dtype)
                adjacency = adjacency + identity
                degree = adjacency.sum(dim=-1).clamp_min_(1e-6)
                inverse = degree.rsqrt()
                adjacency = inverse.unsqueeze(-1) * adjacency * inverse.unsqueeze(-2)
            transformed = nodes.float() @ self.weight.float()
            query_node = (adjacency @ transformed)[:, :, 0, :]
        return query_node.tanh().to(queries.dtype)


class DLFM(nn.Module):
    def __init__(self, dim=768, heads=12, graph_k=40):
        super().__init__()
        self.graph_k = graph_k
        self.shared_gcn = QueryGCN(dim)
        self.specific_gcn = QueryGCN(dim)
        self.shared_norm = nn.LayerNorm(dim)
        self.specific_norm = nn.LayerNorm(dim)
        self.out_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.trunc_normal_(self.out_token, std=.02)
        self.aggregate = PostAttention(dim, heads, 4.)

    def _select(self, feature, patches):
        with torch.no_grad(), torch.autocast(device_type=feature.device.type, enabled=False):
            score = (F.normalize(feature.float(), dim=-1).unsqueeze(1)
                     @ F.normalize(patches.float(), dim=-1).transpose(1, 2)).squeeze(1)
            indices = score.topk(self.graph_k, dim=1).indices
        return patches.gather(1, indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])).detach()

    def forward(self, shared_feature, specific_feature, patch11, shared_queries, specific_queries):
        if self.graph_k > patch11.shape[1]:
            raise ValueError('GRAPH_K exceeds the number of patch tokens')
        shared_patch = self._select(shared_feature, patch11)
        specific_patch = self._select(specific_feature, patch11)
        shared = shared_queries + self.shared_norm(self.shared_gcn(shared_queries, shared_patch))
        specific = specific_queries + self.specific_norm(self.specific_gcn(specific_queries, specific_patch))
        sequence = torch.cat((self.out_token.expand(shared.shape[0], -1, -1), shared, specific), dim=1)
        return self.aggregate(sequence)[:, 0, :]


def make_retrieval_feature(shared, local, beta=0.5, detach_global_for_loss=False):
    global_unit = F.normalize(shared.float(), dim=1)
    local_unit = F.normalize(local.float(), dim=1)
    if detach_global_for_loss:
        global_unit = global_unit.detach()
    return torch.cat((global_unit, beta * local_unit), dim=1) / math.sqrt(1 + beta * beta)
