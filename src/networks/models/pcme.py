import sys

import torch.nn as nn

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from src.networks.clip_model import CLIPImageEncoder, CLIPTextEncoder
from src.networks.models.caption_encoder import EncoderText
from src.networks.models.image_encoder import EncoderImage
    
class PCME(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self, config, mlp_local):
        super(PCME, self).__init__()

        self.config = config
        self.embed_dim = config.embed_dim
        if config.get('n_samples_inference', 0):
            self.n_embeddings = config.n_samples_inference
        else:
            self.n_embeddings = 1

        if config.name == 'clip':
            self.img_enc = CLIPImageEncoder(config, mlp_local=mlp_local)
            self.txt_enc = CLIPTextEncoder(config, mlp_local=mlp_local)
        elif config.name == 'resnet':
            self.img_enc = EncoderImage(config, mlp_local=mlp_local)
            self.txt_enc = EncoderText(config, mlp_local=mlp_local)

    def forward(self, images, captions):
        image_output = self.img_enc(images)
        if self.config.name == 'clip':
            caption_output = self.txt_enc(captions)
            caption_output = {'embedding': caption_output}
        if self.config.name == 'resnet':
            caption_output = self.txt_enc(captions)

        return {
            'image_features': image_output['embedding'],
            'image_attentions': image_output.get('attention'),
            'image_residuals': image_output.get('residual'),
            'image_logsigma': image_output.get('logsigma'),
            'image_logsigma_att': image_output.get('uncertainty_attention'),
            'caption_features': caption_output['embedding'],
            'caption_attentions': caption_output.get('attention'),
            'caption_residuals': caption_output.get('residual'),
            'caption_logsigma': caption_output.get('logsigma'),
            'caption_logsigma_att': caption_output.get('uncertainty_attention'),
        }

    def image_forward(self, images):
        return self.img_enc(images)

    def text_forward(self, captions):
        return self.txt_enc(captions)
