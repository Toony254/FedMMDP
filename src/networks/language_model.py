import os
import pickle

import torch
import torch.nn as nn
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import clip
import torch.nn.functional as F

try:
    from pie_model import PIENet
except ImportError:
    try:
        from models.pie_model import PIENet
    except:
        from src.networks.models.pie_model import PIENet


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask


class EncoderText(nn.Module):
    def __init__(self, word_dim=512, embed_dim=1024, num_class=50, scale=128, mlp_local=False):
        super(EncoderText, self).__init__()
        self.embed_dim = embed_dim

        # Word embedding
        self.embed = nn.Embedding(49408, word_dim)

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)
        if torch.cuda.is_available():
            self.pie_net = self.pie_net.cuda()

        self.relu = nn.ReLU(inplace=False)
        self.class_fc = nn.Linear(embed_dim, num_class)
        self.class_fc_2 = nn.Linear(embed_dim, 80)
        clip_model, _ = clip.load("RN50", device="cuda")
        with torch.no_grad():
            self.embed.weight.data.copy_(clip_model.token_embedding.weight.data)
        self.embed.weight.requires_grad = False
        self.is_train = True
        self.phase = ''
        self.scale = scale

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            )

    def forward(self, x):
        lengths = torch.full((x.shape[0],), 77, dtype=torch.long, device=x.device)
        lengths = lengths.cpu()
        # Embed word ids to vectors
        wemb_out = self.embed(x)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I.to(x.device)).squeeze(1)

        pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
        # print('1', out.device, wemb_out.device, pad_mask.device)
        out, attn, residual = self.pie_net(out, wemb_out, pad_mask.to(out.device))
        out = out * self.scale
        out = self.relu(out)

        if self.is_train:
            fc_weight_relu = self.relu(self.class_fc.weight)
            self.class_fc.weight.data = fc_weight_relu
            x = self.class_fc(out)

            fc_weight_relu2 = self.relu(self.class_fc_2.weight)
            self.class_fc_2.weight.data = fc_weight_relu2
            x2 = self.class_fc_2(out)

            return x, x2, fc_weight_relu, fc_weight_relu2

        if self.mlp_local:
            out = self.head_proj(out)

        out = F.normalize(out, p=2, dim=1)
        return out