import torch.nn as nn
import clip
import sys
import torch.nn.functional as F
import torch
from torchvision import models
from transformers import MobileBertModel, MobileBertTokenizer

sys.path.append("../")
sys.path.append("../../")
from src.utils.tensor_utils import l2_normalize

class ClientImageEncoder(nn.Module):
    def __init__(self, mlp_local, embed_dim=1280, num_class=4, is_train=True, **kwargs):
        super(ClientImageEncoder, self).__init__()
        self.is_train = is_train
        # 使用MobileNetV2作为特征提取器
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = mobilenet.features
        self.embed_dim = embed_dim  # MobileNetV2输出通道为1280
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.class_fc_2 = nn.Linear(self.embed_dim, num_class)
        self.relu = nn.ReLU(inplace=False)
        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            )

    def forward(self, images):
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.feature_extractor(images)
        x = self.avgpool(x)
        x = self.flatten(x)
        if self.is_train:
            fc_weight_relu = self.relu(self.class_fc_2.weight)
            self.class_fc_2.weight.data = fc_weight_relu
            if self.mlp_local:
                x = self.head_proj(x)
            x1 = self.class_fc_2(x)
            return x1, fc_weight_relu, x
        return l2_normalize(x)

class CLIPImageEncoder(nn.Module):
    def __init__(self, config, mlp_local):
        super(CLIPImageEncoder, self).__init__()
        self.config = config
        clip_model, _= clip.load("RN50", device="cuda")
        for param in clip_model.visual.parameters():
            param.requires_grad = False
        emb_dim = clip_model.visual.output_dim
        self.visual_projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        ).half()
        projector_weight_path = "saved/projector_weights/norm.pth"
        projector_weight = torch.load(projector_weight_path)
        self.visual_projector.load_state_dict(projector_weight['visual_projector'].state_dict())
        class CombinedVisualModel(nn.Module):
            def __init__(self, clip_visual, projector):
                super(CombinedVisualModel, self).__init__()
                self.clip_visual = clip_visual
                self.projector = projector
                
            def forward(self, x):
                x = self.clip_visual(x)
                x = self.projector(x)
                return x
        
        self.clip_visual = CombinedVisualModel(clip_model.visual, self.visual_projector)
        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim, emb_dim)
            ).half()

    def forward(self, images):
        images = images.half()
        out = self.clip_visual(images)
        if self.mlp_local:
            out = self.head_proj(out)
        out = l2_normalize(out)
        output = {'embedding': out}
        return output

class ClientTextEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_class=4, scale=128, mlp_local=False, is_train=True):
        super(ClientTextEncoder, self).__init__()
        # 使用transformers的MobileBert
        self.bert = MobileBertModel.from_pretrained('google/mobilebert-uncased')
        self.embed_dim = self.bert.config.hidden_size
        self.class_fc_2 = nn.Linear(self.embed_dim, num_class)
        self.relu = nn.ReLU(inplace=False)
        self.is_train = is_train
        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim)
            )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        if self.is_train:
            fc_weight_relu = self.relu(self.class_fc_2.weight)
            self.class_fc_2.weight.data = fc_weight_relu
            if self.mlp_local:
                x = self.head_proj(x)
            x1 = self.class_fc_2(x)
            return x1, fc_weight_relu, x
        if self.mlp_local:
            x = self.head_proj(x)
        return l2_normalize(x)

class CLIPTextEncoder(nn.Module):
    def __init__(self, config, embed_dim=1024, num_class=4, scale=128, mlp_local=False):
        super(CLIPTextEncoder, self).__init__()
        self.config = config
        clip_model, _ = clip.load("RN50", device="cuda")
        for param in clip_model.transformer.parameters():
            param.requires_grad = False
        emb_dim = clip_model.text_projection.shape[1]
        dtype = clip_model.dtype
        self.text_projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        ).half()
        projector_weight_path = "saved/projector_weights/norm.pth"
        projector_weight = torch.load(projector_weight_path)
        self.text_projector.load_state_dict(projector_weight['text_projector'].state_dict())
        class CombinedTextModel(nn.Module):
            def __init__(self, projector):
                super(CombinedTextModel, self).__init__()
                self.token_embedding = clip_model.token_embedding
                self.transformer = clip_model.transformer
                self.ln_final = clip_model.ln_final
                self.text_projection = clip_model.text_projection
                self.positional_embedding = clip_model.positional_embedding
                self.projector = projector

            def forward(self, text):
                x = self.token_embedding(text).type(dtype)

                x = x + self.positional_embedding.type(dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.ln_final(x).type(dtype)
                
                x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
                
                x = self.projector(x)
                return x
        
        self.clip_text = CombinedTextModel(self.text_projector)
    
    def forward(self, inputs):
        out = self.clip_text(inputs)
        out = l2_normalize(out)
        return out