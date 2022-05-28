import torch
from torch import nn
import torch.nn.functional as F
import configs
import T5
from T5 import get_encoded_text



class Unet(nn.Module):
    def __init__(self):
        super()__init__()

    def forward(self,x):
        return x

class

class Imagen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        text_embeds, text_masks = get_encode_text(texts, model_name=configs.text_encoder_name)

        return x