""" Caption encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import os
import sys
import clip
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext

sys.path.append("../")
sys.path.append("../../")
from src.networks.models.pie_model import PIENet
from src.networks.models.uncertainty_module import UncertaintyModuleText
from src.utils.tensor_utils import l2_normalize, sample_gaussian_tensors


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask


class EncoderText(nn.Module):
    def __init__(self, opt, mlp_local):
        super(EncoderText, self).__init__()

        word_dim, embed_dim = \
            opt.word_dim, opt.embed_dim

        self.embed_dim = embed_dim

        # Word embedding
        self.embed = nn.Embedding(49408, word_dim)
        self.embed.weight.requires_grad = True

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)

        clip_model, _ = clip.load(opt.clip_model, device="cuda")
        with torch.no_grad():
            self.embed.weight.data.copy_(clip_model.token_embedding.weight.data)
        self.embed.weight.requires_grad = False

        self.n_samples_inference = opt.get('n_samples_inference', 0)

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512)
            )

    def forward(self, x):
        # Embed word ids to vectors
        lengths = torch.full((x.shape[0],), 77, dtype=torch.long, device=x.device)
        lengths = lengths.cpu()
        wemb_out = self.embed(x)  # [bsz, seq_len, 512]

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)  # padded[0]: [bsz, seq_len, dim], padded[1]: [128], len

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I.to(x.device)).squeeze(1)  # [bsz, dim]

        output = {}

        pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
        # print('1', out.device, wemb_out.device, pad_mask.device)
        out, attn, residual = self.pie_net(out, wemb_out, pad_mask.to(out.device))  # out: [bsz, dim]

        out = l2_normalize(out)

        if self.mlp_local:
            out = self.head_proj(out)

        output['embedding'] = out

        return output
