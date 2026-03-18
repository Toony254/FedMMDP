import itertools
import math
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class LabelConditionalGenerator(nn.Module):
    def __init__(self, num_classes, embed_dim, noise_dim=None, hidden_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.noise_dim = noise_dim or min(embed_dim, 128)
        self.hidden_dim = hidden_dim or max(embed_dim, 256)
        self.network = nn.Sequential(
            nn.Linear(self.num_classes + self.noise_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(self, labels, noise=None):
        if labels.dtype != torch.long:
            labels = labels.long()
        one_hot = torch.zeros(labels.size(0), self.num_classes, device=labels.device, dtype=torch.float32)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        if noise is None:
            noise = torch.randn(labels.size(0), self.noise_dim, device=labels.device, dtype=torch.float32)
        else:
            noise = noise.to(labels.device, dtype=torch.float32)
        output = self.network(torch.cat([one_hot, noise], dim=1))
        return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)


def clone_state_dict(state_dict):
    cloned = OrderedDict()
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().clone()
        if torch.is_floating_point(value):
            value = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
            value = torch.clamp(value, min=-10.0, max=10.0)
        cloned[name] = value
    return cloned


def weighted_average_state_dicts(state_dicts, weights=None):
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")

    if weights is None:
        weights = [1.0] * len(state_dicts)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / weights.sum()

    averaged = OrderedDict()
    for key in state_dicts[0].keys():
        ref = state_dicts[0][key]
        accum = None
        for state_dict, weight in zip(state_dicts, weights):
            tensor = state_dict[key].detach().cpu()
            if not torch.is_floating_point(tensor):
                averaged[key] = tensor.clone()
                accum = None
                break
            tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=1e4, neginf=-1e4) * float(weight)
            tensor = torch.clamp(tensor, min=-10.0, max=10.0)
            accum = tensor if accum is None else accum + tensor
        if accum is not None:
            averaged[key] = torch.clamp(
                torch.nan_to_num(accum, nan=0.0, posinf=1e4, neginf=-1e4),
                min=-10.0,
                max=10.0,
            ).to(ref.dtype)
    return averaged


def sanitize_module_parameters(module, clamp_value=10.0):
    with torch.no_grad():
        for parameter in module.parameters():
            if torch.is_floating_point(parameter.data):
                parameter.data = torch.nan_to_num(parameter.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
                parameter.data.clamp_(min=-clamp_value, max=clamp_value)


def compute_generator_signature(generator_state, generator_factory, labels, device):
    generator = generator_factory().to(device)
    generator.load_state_dict(generator_state, strict=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.zeros(labels.size(0), generator.noise_dim, device=device, dtype=torch.float32)
        signature = generator(labels.to(device), noise=noise).detach().cpu().reshape(-1).numpy()
    return signature


def cluster_generator_states(generator_states, generator_factory, labels, device, num_clusters):
    if len(generator_states) == 1:
        return np.array([0]), {0: [0]}

    signatures = np.stack(
        [compute_generator_signature(state, generator_factory, labels, device) for state in generator_states],
        axis=0,
    )
    signatures = np.nan_to_num(signatures, nan=0.0, posinf=1e4, neginf=-1e4)
    cluster_count = max(1, min(num_clusters, len(generator_states)))
    if cluster_count == 1:
        assignments = np.zeros(len(generator_states), dtype=np.int64)
    else:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        assignments = model.fit_predict(signatures)
    clusters = {cluster_id: np.where(assignments == cluster_id)[0].tolist() for cluster_id in np.unique(assignments)}
    return assignments, clusters


def shapley_values(num_players, utility_fn, max_permutations=24, seed=42):
    if num_players <= 0:
        return []
    players = list(range(num_players))
    factorial = math.factorial(num_players)
    rng = random.Random(seed)

    if factorial <= max_permutations:
        permutations = list(itertools.permutations(players))
    else:
        permutations = [tuple(rng.sample(players, len(players))) for _ in range(max_permutations)]

    contributions = np.zeros(num_players, dtype=np.float64)
    for permutation in permutations:
        subset = []
        current_utility = 0.0
        for player in permutation:
            subset.append(player)
            next_utility = utility_fn(tuple(sorted(subset)))
            contributions[player] += next_utility - current_utility
            current_utility = next_utility

    contributions /= max(1, len(permutations))
    min_value = contributions.min()
    if min_value < 0:
        contributions = contributions - min_value
    if contributions.sum() <= 0:
        contributions = np.ones(num_players, dtype=np.float64) / num_players
    else:
        contributions /= contributions.sum()
    return contributions.tolist()
