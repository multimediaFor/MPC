import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


class MyInfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', metric='cosine'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.metric = metric

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode, metric=self.metric)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired',
             metric='cosine'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        # positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        sim_matrix = my_distance(query, query, metric)
        mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)

        positive_logit = torch.mean(sim_matrix, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = my_distance(query, negative_keys, metric)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def my_distance(x, y, metric='cosine'):
    if metric == 'cosine':
        return torch.mm(x, y.t())
    else:
        return torch.cdist(x, y)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


myInfoNCE = MyInfoNCE(metric='cosine')


# intra-image loss
def intra_infonce_loss(Features, GTs):
    Feature = Features[0]
    GT = GTs[0]
    query = Feature[GT == 0]
    negative = Feature[GT == 1]
    dict_size = 5000  # could be larger according to gpu memory
    query_sample = query[torch.randperm(query.size()[0])[:dict_size]]
    negative_sample = negative[torch.randperm(negative.size(0))[:dict_size]]
    return myInfoNCE(query_sample, query_sample, negative_sample)


# cross-scale loss
def cross_scale_infonce_loss(Features, GTs):
    FeatureX = Features[0]
    queryX = FeatureX[GTs[0] == 0]
    negativeX = FeatureX[GTs[0] == 1]

    FeatureY_1 = Features[1]
    queryY_1 = FeatureY_1[GTs[1] == 0]
    negativeY_1 = FeatureY_1[GTs[1] == 1]

    FeatureY_2 = Features[2]
    queryY_2 = FeatureY_2[GTs[2] == 0]
    negativeY_2 = FeatureY_2[GTs[2] == 1]

    FeatureY_3 = Features[3]
    queryY_3 = FeatureY_3[GTs[3] == 0]
    negativeY_3 = FeatureY_3[GTs[3] == 1]

    queryY = torch.cat((queryY_1, queryY_2, queryY_3), dim=0)
    negativeY = torch.cat((negativeY_1, negativeY_2, negativeY_3), dim=0)

    dict_size = 5000  # could be larger according to gpu memory
    query_sampleX = queryX[torch.randperm(queryX.size()[0])[:dict_size]]
    negative_sampleX = negativeX[torch.randperm(negativeX.size(0))[:dict_size]]

    query_sampleY = queryY[torch.randperm(queryY.size()[0])[:dict_size]]
    negative_sampleY = negativeY[torch.randperm(negativeY.size(0))[:dict_size]]

    return myInfoNCE(query_sampleX, query_sampleX, negative_sampleY) + myInfoNCE(query_sampleY, query_sampleY,
                                                                                 negative_sampleX)


# cross-modality loss
def cross_modality_infonce_loss(FeaturesX, FeaturesY, GTs):
    FeatureX = FeaturesX[0]
    FeatureY = FeaturesY[0]
    GT = GTs[0]
    dict_size = 5000  # could be larger according to gpu memory

    queryX = FeatureX[GT == 0]
    negativeX = FeatureX[GT == 1]
    query_sampleX = queryX[torch.randperm(queryX.size()[0])[:dict_size]]
    negative_sampleX = negativeX[torch.randperm(negativeX.size(0))[:dict_size]]

    queryY = FeatureY[GT == 0]
    negativeY = FeatureY[GT == 1]
    query_sampleY = queryY[torch.randperm(queryY.size()[0])[:dict_size]]
    negative_sampleY = negativeY[torch.randperm(negativeY.size(0))[:dict_size]]

    return myInfoNCE(query_sampleX, query_sampleX, negative_sampleY) + myInfoNCE(query_sampleY, query_sampleY,
                                                                                 negative_sampleX)


def MyLoss(FeaturesX, FeaturesY, GT):
    H, W = GT.shape
    mask2 = F.interpolate(GT.unsqueeze(0).unsqueeze(0), size=(H//2, W//2), mode='nearest').squeeze()
    mask3 = F.interpolate(GT.unsqueeze(0).unsqueeze(0), size=(H // 4, W // 4), mode='nearest').squeeze()
    mask4 = F.interpolate(GT.unsqueeze(0).unsqueeze(0), size=(H // 8, W // 8), mode='nearest').squeeze()
    GTs = [GT, mask2, mask3, mask4]
    intro_loss = intra_infonce_loss(FeaturesX, GTs)
    cross_scale_loss = cross_scale_infonce_loss(FeaturesX, GTs)
    cross_modality_loss = cross_modality_infonce_loss(FeaturesX, FeaturesY, GTs)
    return intro_loss + cross_scale_loss + cross_modality_loss
