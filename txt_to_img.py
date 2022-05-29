import configs
import T5


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .nn import timestep_embedding
from .unet import l


class txt_to_img(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.
    Expects an extra kwarg `tokens` of text.
    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    """

    def __init__(
        self,
        xf_width,
        *args,
        cache_text_emb=True,
        **kwargs,
    ):
        self.xf_width = xf_width
        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=xf_width)

        self.sen_emb_proj = nn.Linear(configs.embedding_dim, self.model_channels * 4)
        self.to_xf_width = nn.Linear(self.t5.shared.embedding_dim, xf_width)
        self.cache_text_emb = cache_text_emb
        self.cache = None
    def convert_to_fp16(self):

        super().convert_to_fp16()
        self.sen_emb_proj.to(th.float16)

        self.to_xf_width.to(th.float16)
    def get_text_emb(self, encoded_text):
        #load_encoded text from file
        if self.cache is not None and self.cache_text_emb:
            return self.cache
        xf_proj = self.sen_emb_proj(encoded_text[:, -1])
        xf_out2 = self.to_xf_width(xf_out)
        xf_out2 = xf_out2.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out2)
        if self.cache_text_emb:
            self.cache = outputs
        return outputs


    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps,encoded_text):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.xf_width:
            text_outputs = self.get_text_emb(encoded_text)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h
