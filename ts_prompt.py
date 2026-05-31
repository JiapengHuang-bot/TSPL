import os.path as osp
import os
import datetime
import time
from collections import OrderedDict
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Union, List, Tuple

from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    # import pdb
    # pdb.set_trace()
    if isinstance(texts, str):
        texts = [texts] #['a photo of a face.']

    sot_token = _tokenizer.encoder["<|startoftext|>"] #49406
    eot_token = _tokenizer.encoder["<|endoftext|>"] #49407
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long) #1,77

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length: #context_length 77
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)

        x = x[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x, (1)) / (x.shape[1])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([1, 0, 2]) - self.mu(x)).permute([1, 0, 2]) ** 2, (1)) + 0.000000023) / (x.shape[1]))

class image_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList(nn.Linear(768, 512) for _ in range(12))
        self.adain = AdaIN()
        self.lin = nn.Linear(12, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, data, n_imgctx):
        data_prompt = []
        for i in range(len(data)):
            x_gap = self.gap(data[i]).squeeze(1)
            x_lin = self.linear[i](x_gap)
            data_prompt.append(x_lin)
        feat = torch.stack(data_prompt, dim=1)
        output = []
        for i in range(n_imgctx):  # L decoders
            x = self.lin(feat.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            output.append(x)
        feat_tokens = torch.stack(output, dim=1).squeeze(2)
        return feat_tokens


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_imgctx = 4
        n_ctx = 24 + n_imgctx

        dtype = clip_model.dtype

        self.image_tokens = image_projector()

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_imgctx = n_imgctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts

    @autocast()
    def forward(self, source_data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_imgctx = self.n_imgctx

        source_tokens = self.image_tokens(source_data, n_imgctx)

        source_prompts = []
        for tokens_i in source_tokens:
            ctx_i = tokens_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            source_prompts.append(pts_i)
        source_prompts = torch.stack(source_prompts)

        return source_prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

    @autocast()
    def forward(self, source_data):
        source_prompts = self.prompt_learner(source_data)
        tokenized_prompts = self.tokenized_prompts

        source_text_features = []
        for pts_i in source_prompts:
            tf = self.text_encoder(pts_i, tokenized_prompts)
            source_text_features.append(tf)
        source_text_features = torch.stack(source_text_features)
        #source_text_features = source_text_features / source_text_features.norm(dim=-1, keepdim=True)

        return source_text_features
