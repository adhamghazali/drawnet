import torch
from torch import nn
import torch.nn.functional as F
import configs
import T5
from T5 import get_encoded_text

class Swish(nn.Module):
    #Swish actiavation function
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    ### Embeddings for t

    def __init__(self, n_channels: int):
        # `n_channels` is the number of dimensions in the embedding

        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

        def forward(self,t):
            half_dim = self.n_channels // 8
            emb = math.log(10_000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
            emb = t[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=1)

            # Transform with the MLP
            emb=self.lin1(emb)
            emb = self.act(emb)
            emb = self.lin2(emb)

            return emb

class Unet(nn.Module):
    def __init__(self):
        super()__init__()

    def forward(self,x):
        return x

class

class Imagen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,text,embeds,text_masks):
        #text_embeds, text_masks = get_encode_text(texts, model_name=configs.text_encoder_name)
        #TODO: Generate encodings offline

        return x