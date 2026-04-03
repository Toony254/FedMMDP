import torch.nn as nn
import sys
import torch

sys.path.append("../")
sys.path.append("../../")
from src.utils.tensor_utils import l2_normalize
from src.utils.projector_utils import resolve_projector_weight_path


def _maybe_load_projector(
    projector,
    projector_key,
    emb_dim,
    use_pretrained_proj,
    model_name='clip',
    projector_variant='',
    projector_path='',
):
    if not use_pretrained_proj:
        return

    projector_weight_path = resolve_projector_weight_path(
        model_name=model_name,
        emb_dim=emb_dim,
        projector_name='mlp+norm',
        variant=projector_variant,
        override_path=projector_path,
    )
    projector_weight = torch.load(projector_weight_path, map_location='cpu')
    projector_state = projector_weight[projector_key]
    if hasattr(projector_state, 'state_dict'):
        projector_state = projector_state.state_dict()
    projector.load_state_dict(projector_state)


class ClientImageEncoder(nn.Module):
    def __init__(self, mlp_local, **kwargs):
        super(ClientImageEncoder, self).__init__()
        self.is_train = bool(kwargs['is_train'])
        emb_dim = kwargs.get('embed_dim')
        use_pretrained_proj = bool(kwargs.get('use_pretrained_proj', True))
        model_name = kwargs.get('model_name', 'clip')
        projector_variant = kwargs.get('pretrained_proj_variant', '')
        projector_path = kwargs.get('pretrained_proj_path', '')

        self.visual_projector = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
        _maybe_load_projector(
            self.visual_projector,
            'visual_projector',
            emb_dim,
            use_pretrained_proj,
            model_name=model_name,
            projector_variant=projector_variant,
            projector_path=projector_path,
        )

        self.embed_dim = kwargs.get('embed_dim', emb_dim)
        self.relu = nn.ReLU(inplace=False)
        if self.embed_dim != emb_dim:
            self.linear = nn.Linear(emb_dim, self.embed_dim)
        self.class_fc_2 = nn.Linear(self.embed_dim, kwargs['num_class'])
        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

    def forward(self, embeddings):
        embeddings = embeddings.float()
        x = self.visual_projector(embeddings)

        if hasattr(self, 'linear'):
            x = self.linear(x)

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
        emb_dim = config['embed_dim']
        use_pretrained_proj = bool(config.get('use_pretrained_proj', True))
        model_name = config.get('name', 'clip')
        projector_variant = config.get('pretrained_proj_variant', '')
        projector_path = config.get('pretrained_proj_path', '')

        self.visual_projector = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
        _maybe_load_projector(
            self.visual_projector,
            'visual_projector',
            emb_dim,
            use_pretrained_proj,
            model_name=model_name,
            projector_variant=projector_variant,
            projector_path=projector_path,
        )

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim, emb_dim),
            )

    def forward(self, embeddings):
        embeddings = embeddings.float()
        out = self.visual_projector(embeddings)
        if self.mlp_local:
            out = self.head_proj(out)
        out = l2_normalize(out)
        output = {'embedding': out}
        return output


class ClientTextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        num_class=4,
        scale=128,
        mlp_local=False,
        use_pretrained_proj=True,
        model_name='clip',
        pretrained_proj_variant='',
        pretrained_proj_path='',
    ):
        super(ClientTextEncoder, self).__init__()
        emb_dim = embed_dim

        self.text_projector = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
        _maybe_load_projector(
            self.text_projector,
            'text_projector',
            emb_dim,
            use_pretrained_proj,
            model_name=model_name,
            projector_variant=pretrained_proj_variant,
            projector_path=pretrained_proj_path,
        )

        self.relu = nn.ReLU(inplace=False)
        self.class_fc_2 = nn.Linear(embed_dim, num_class)
        self.is_train = True
        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim, emb_dim),
            )

    def forward(self, embeddings, lengths=None):
        embeddings = embeddings.float()
        out = self.text_projector(embeddings)

        if self.is_train:
            fc_weight_relu = self.relu(self.class_fc_2.weight)
            self.class_fc_2.weight.data = fc_weight_relu
            x = self.class_fc_2(out)
            return x, fc_weight_relu, out
        if self.mlp_local:
            out = self.head_proj(out)
        out = l2_normalize(out)
        return out


class CLIPTextEncoder(nn.Module):
    def __init__(self, config, embed_dim=1024, num_class=4, scale=128, mlp_local=False):
        super(CLIPTextEncoder, self).__init__()
        self.config = config
        emb_dim = config['embed_dim']
        use_pretrained_proj = bool(config.get('use_pretrained_proj', True))
        model_name = config.get('name', 'clip')
        projector_variant = config.get('pretrained_proj_variant', '')
        projector_path = config.get('pretrained_proj_path', '')

        self.text_projector = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
        _maybe_load_projector(
            self.text_projector,
            'text_projector',
            emb_dim,
            use_pretrained_proj,
            model_name=model_name,
            projector_variant=projector_variant,
            projector_path=projector_path,
        )

    def forward(self, embeddings):
        embeddings = embeddings.float()
        out = self.text_projector(embeddings)
        out = l2_normalize(out)
        return out
