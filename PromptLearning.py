import torch
import torch.nn as nn
from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from typing import Union, List, Tuple
import torch.nn.functional as F

import os
from model.clip_model import _MODELS, _download, available_models, convert_weights, CLIP, VisionTransformer


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

class PromptLearner(nn.Module):
    def __init__(self, num_class, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."
        #ctx_init = "A photo of a X X X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        #n_ctx = 6

        tokenized_prompts = tokenize(ctx_init).cpu()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        #n_cls_ctx = 8
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #loss = - mean_log_prob_pos.mean()

        # 初始化损失
        #t2i_loss = 0.0
        #if t2i == True:
        #    # 对每个文本 token，找到相同 ID 的图像，并进行匹配
        #    for i in range(text_features.size(0)):
        #        # 找到所有与当前文本相同 ID 的图像
        #        positive_indices = torch.where(t_label == t_label[i])[0]
        #        # 取出这些正样本图像的 logits
        #        positive_logits = logits[i, positive_indices]
        #        # 计算交叉熵损失，并取平均
        #        loss = F.cross_entropy(positive_logits, torch.zeros(len(positive_indices), dtype=torch.long).to(positive_logits.device))
        #        t2i_loss += loss
        #    # 平均损失
        #    t2i_loss = t2i_loss / text_features.size(0)

        #return loss
        return (- mean_log_prob_pos)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        #print(f"x shape: {prompts.shape}")
        #print(f"x shape: {(self.positional_embedding.type(self.dtype)).shape}")
        #print(f"x shape: {x.shape}")  # 检查 x 的形状，应该是 [batch_size, sequence_length, embedding_dim]

        x = x.permute(1, 0, 2)  # NLD -> LND

        #print(f"x shape: {x.shape}")  # 检查 x 的形状，应该是 [batch_size, sequence_length, embedding_dim]

        outputs = self.transformer([x])
        x = outputs[0]

        #print(f"x shape: {x.shape}")

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

#def load_clip(name: str, args):
#    """Load a CLIP model
#
#    Parameters
#    ----------
#    name : str
#        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
#    Returns
#    -------
#    model : torch.nn.Module
#        The CLIP model
#    """
#    if name in _MODELS:
#        model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
#    elif os.path.isfile(name):
#        model_path = name
#    else:
#        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
#
#    with open(model_path, 'rb') as opened_file:
#        try:
#            # loading JIT archive
#            model = torch.jit.load(opened_file, map_location="cpu").eval()
#            state_dict = None
#        except RuntimeError:
#            # loading saved state dict
#            state_dict = torch.load(opened_file, map_location="cpu")
#
#    model = build_model(state_dict or model.state_dict(), args.stride_size).to("cpu")
#    model.float()
#    return model
#
#
#def build_model(state_dict: dict, stride_size: int):
#    vit = "visual.proj" in state_dict
#
#    if vit:
#        vision_width = state_dict["visual.conv1.weight"].shape[0]
#        vision_layers = len(
#            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
#        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#        image_resolution = vision_patch_size * grid_size
#    else:
#        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
#                        [1, 2, 3, 4]]
#        vision_layers = tuple(counts)
#        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
#        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#        vision_patch_size = None
#        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
#        image_resolution = output_width * 32
#
#    embed_dim = state_dict["text_projection"].shape[1]
#    context_length = state_dict["positional_embedding"].shape[0]
#    vocab_size = state_dict["token_embedding.weight"].shape[0]
#    transformer_width = state_dict["ln_final.weight"].shape[0]
#    transformer_heads = transformer_width // 64
#    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
#
#    model = PromptCLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, stride_size,
#                       context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
#
#    for key in ["input_resolution", "context_length", "vocab_size"]:
#        if key in state_dict:
#            del state_dict[key]
#
#    convert_weights(model)
#    model.load_state_dict(state_dict)
#    return model.eval()
#
#
#class PromptCLIP(CLIP):
#    def __init__(self,
#                 embed_dim: int,
#                 # vision
#                 image_resolution: int,
#                 vision_layers: Union[Tuple[int, int, int, int], int],
#                 vision_width: int,
#                 vision_patch_size: int,
#                 stride_size: int,
#                 # text
#                 context_length: int,
#                 vocab_size: int,
#                 transformer_width: int,
#                 transformer_heads: int,
#                 transformer_layers: int
#                 ):
#        super().__init__(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, stride_size, context_length,
#                         vocab_size, transformer_width, transformer_heads, transformer_layers)
#        if not isinstance(vision_layers, (tuple, list)):
#            vision_heads = vision_width // 64
#            self.visual = PromptVisionTransformer(
#                input_resolution=image_resolution,
#                patch_size=vision_patch_size,
#                stride_size=stride_size,
#                width=vision_width,
#                layers=vision_layers,
#                heads=vision_heads,
#                output_dim=embed_dim)
#
#
#class PromptVisionTransformer(VisionTransformer):
#    def forward(self, x: torch.Tensor, prompt: torch.Tensor = None):
#        x = self.conv1(x)  # shape = [*, width, grid, grid]
#        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#        x = torch.cat(
#            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#        x = x + self.positional_embedding.to(x.dtype)
#        if prompt is not None:
#            # prompt should be of shape [*, K, width]
#            x = torch.cat([prompt, x], dim=1)  # [*, grid ** 2 + 1 + K, width]
#        x = self.ln_pre(x)
#
#        x = x.permute(1, 0, 2)  # NLD -> LND
#        x = self.transformer(x)
#        x = x.permute(1, 0, 2)  # LND -> NLD
#
#        x = self.ln_post(x[:, 0, :])
#
#        if self.proj is not None:
#            x = x @ self.proj
#
#        return x
