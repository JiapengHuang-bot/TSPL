from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights, QuickGELU, LayerNorm, Cross_Transformer
import torch
import torch.nn as nn 
import torch.nn.functional as F
from model.feature_enhance import BiAttentionBlock
from MMC import MMc
from collections import OrderedDict
from PromptLearning import PromptLearner, SupConLoss, TextEncoder
#from visual_prompt import ContextDecoder
#from mmseg.models import builder
from visual_prompt import ImageSpecificPrompt
from timm.models.layers import trunc_normal_
from torch.cuda.amp import autocast
import random

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def unfreeze_ln(m):
    if isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(True)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(True)

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        self.feature_fusion_layer = BiAttentionBlock(512, 512, 2048, 8, 0.1, 0.0)  # 原embed_dim=2048

        self.classifier = nn.Linear(self.embed_dim, self.num_classes)  # 如果包括分类任务，就创建一个线性分类器
        nn.init.normal_(self.classifier.weight.data, std=0.001)  # 对分类器的权重进行初始化，使用正态分布来初始化权重，均值为 0，标准差为 0.001
        nn.init.constant_(self.classifier.bias.data, val=0.0)

        self.classifier_tse = nn.Linear(1024, self.num_classes)  # 如果包括分类任务，就创建一个线性分类器
        nn.init.normal_(self.classifier.weight.data, std=0.001)  # 对分类器的权重进行初始化，使用正态分布来初始化权重，均值为 0，标准差为 0.001
        nn.init.constant_(self.classifier.bias.data, val=0.0)

        self.classifier_f = nn.Linear(self.embed_dim * 2, self.num_classes)  # 如果包括分类任务，就创建一个线性分类器
        nn.init.normal_(self.classifier_f.weight.data, std=0.001)  # 对分类器的权重进行初始化，使用正态分布来初始化权重，均值为 0，标准差为 0.001
        nn.init.constant_(self.classifier_f.bias.data, val=0.0)

        self.classifier_f_tse = nn.Linear(2048, self.num_classes)  # 如果包括分类任务，就创建一个线性分类器
        nn.init.normal_(self.classifier_f.weight.data, std=0.001)  # 对分类器的权重进行初始化，使用正态分布来初始化权重，均值为 0，标准差为 0.001
        nn.init.constant_(self.classifier_f.bias.data, val=0.0)

        self.prompt_learner = PromptLearner(num_classes, self.base_model.dtype, self.base_model.token_embedding)
        self.sup = SupConLoss('cuda')
        self.text_encoder = TextEncoder(self.base_model)

        #self.context_decoder = ContextDecoder()
        #self.gamma = nn.Parameter(torch.ones(512) * 1e-4)

        self.context_decoder = ImageSpecificPrompt()

        #clip_model, _ = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        #for param in clip_model.parameters():
        #    param.requires_grad_(False)
        #visual = clip_model.visual
        #visual.apply(unfreeze_ln)
        #visual.proj.requires_grad_(True)
        #self.photo_encoder = visual
        #self.photo_prompt = nn.Parameter(torch.randn(3, self.photo_encoder.class_embedding.shape[0]))

        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)  # 创建了一个多头注意力模块，用于处理图像和文本之间的交叉注意力。使用self.embed_dim 作为输入维度，并将 self.embed_dim // 64 作为注意力头的数量。batch_first=True 表示输入数据的第一个维度是批次大小
        self.cross_modal_transformer = Cross_Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
        scale = self.cross_modal_transformer.width ** -0.5  # 计算了一个缩放因子，将在后续初始化中使用。这个缩放因子是transformer宽度的倒数的平方根
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)  # 创建了三个 Layer Normalization（LN） 操作，分别用于文本输入的前 LN、图像输入的前 LN 以及整体输出的 LN
        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5  # 计算了用于初始化参数的标准差。proj_std 用于初始化变换器中的投影层参数，attn_std 用于初始化注意力模块的参数，fc_std 用于初始化全连接层的参数
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)  # 这些权重是注意力层和全连接层中的参数，用于模型的前向传播过程
        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)  # 初始化全局的交叉注意力层 (self.cross_attn.in_proj_weight 和 self.cross_attn.out_proj.weight) 的权重
        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                         # 这是一个线性全连接层，它将输入的特征维度 self.embed_dim 映射到相同的维度。这一层通常被称为 "dense" 层
                         ('gelu', QuickGELU()),  # 这是 GELU (Gaussian Error Linear Unit) 激活函数的一种快速实现
                         ('ln', LayerNorm(self.embed_dim)),  # 这是批标准化 (Batch Normalization) 层，用于规范化输入数据，以加速训练并提高模型的泛化性能
                         ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))  # 这是最后一层全连接层，它将特征映射到一个输出空间，该输出空间的维度是 args.vocab_size，通常用于预测文本数据中的词汇。
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, _ = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, _ = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)

        v_feats, l_feats = self.feature_fusion_layer(image_feats, text_feats)
        va_feats = v_feats[:, 0, :].float()
        la_feats = l_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        image_logits = self.classifier(va_feats.half()).float()  # 计算图像特征 i_feats 的分类器输出, i_feats 用于将张量中的数据类型转换为半精度浮点数类型
        text_logits = self.classifier(la_feats.half()).float()  # 计算文本特征 t_feats的分类器输出
        fusion = torch.cat([la_feats, va_feats], dim=-1)
        output_mm = self.classifier_f(fusion.half())
        criterion0 = nn.CrossEntropyLoss(reduction="none")
        loss_f = criterion0(output_mm, batch['pids'])
        I = MMc(self.args)
        exp_logits, mask = I.mmc_2(la_feats, va_feats, text_logits, image_logits, batch['pids'])
        masked_exp_logits = mask * exp_logits
        ratio = masked_exp_logits / exp_logits.sum()
        ratio = torch.where(ratio == 0, torch.tensor(1e-10), ratio)
        lossC = -0.1 * torch.log(ratio / mask.sum()) + loss_f + objectives.compute_id(image_logits, text_logits, batch['pids'])* 1.0

        i_tse_logits = self.classifier_tse(i_tse_f.half()).float()  # 计算图像特征 i_feats 的分类器输出, i_feats 用于将张量中的数据类型转换为半精度浮点数类型
        t_tse_logits = self.classifier_tse(t_tse_f.half()).float()  # 计算文本特征 t_feats的分类器输出
        tse_fusion = torch.cat([t_tse_f, i_tse_f], dim=-1)
        output_tse_mm = self.classifier_f_tse(tse_fusion.half())
        loss_tse_f = criterion0(output_tse_mm, batch['pids'])
        exp_logits_tse, mask_tse = I.mmc_2(t_tse_f, i_tse_f, t_tse_logits, i_tse_logits, batch['pids'])
        masked_exp_logits_tse = mask_tse * exp_logits_tse
        ratio_tse = masked_exp_logits_tse / exp_logits_tse.sum()
        ratio_tse = torch.where(ratio_tse == 0, torch.tensor(1e-10), ratio_tse)
        lossD = -0.1 * torch.log(ratio_tse / mask_tse.sum()) + loss_tse_f + objectives.compute_id(i_tse_logits, t_tse_logits, batch['pids'])* 1.0

        with autocast():
            prompt_feats = self.prompt_learner(batch['pids'])
            p_feats = prompt_feats[torch.arange(prompt_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            #p_feats = self.text_encoder(prompt_feats, self.prompt_learner.tokenized_prompts)
            prompt_loss_i, _ = objectives.compute_per_loss(i_feats, p_feats, batch['pids'], loss_type='TAL')
            prompt_loss_t, _ = objectives.compute_per_loss(t_feats, p_feats, batch['pids'], loss_type='InfoNCE')

            #prompt_loss_p2i = self.sup(p_feats, i_feats, batch['pids'], batch['pids'])
            #prompt_loss_i2p = self.sup(i_feats, p_feats, batch['pids'], batch['pids'])

        lossE = prompt_loss_i + prompt_loss_t
        #lossE = prompt_loss_p2i + prompt_loss_i2p

        with autocast():
            p_tse_f = self.texual_emb_layer(prompt_feats, caption_ids, atten_t)
            prompt_loss_i_tse, _ = objectives.compute_per_loss(i_tse_f, p_tse_f, batch['pids'], loss_type='TAL')
            prompt_loss_t_tse, _ = objectives.compute_per_loss(t_tse_f, p_tse_f, batch['pids'], loss_type='InfoNCE')

            #print(p_tse_f.shape)
            #print(i_feats.shape)

            #prompt_loss_p2i_tse = self.sup(p_tse_f, i_tse_f, batch['pids'], batch['pids'])
            #prompt_loss_i2p_tse = self.sup(i_tse_f, p_tse_f, batch['pids'], batch['pids'])

        lossF = prompt_loss_i_tse + prompt_loss_t_tse
        #lossF = prompt_loss_p2i_tse + prompt_loss_i2p_tse

        text_diff = self.context_decoder(prompt_feats, image_feats)
        prompt_feats_upgrade = prompt_feats + text_diff  # visual upgrade text prompts
        #p_feats_upgrade = prompt_feats_upgrade[torch.arange(prompt_feats_upgrade.shape[0]), caption_ids.argmax(dim=-1)].float()
        #prompt_loss_i_upgrade, _ = objectives.compute_per_loss(i_feats, p_feats_upgrade, batch['pids'], loss_type='TAL')
        #prompt_loss_t_upgrade, _ = objectives.compute_per_loss(t_feats, p_feats_upgrade, batch['pids'], loss_type='InfoNCE')
        #lossG = prompt_loss_i_upgrade + prompt_loss_t_upgrade

        with autocast():
            p_feats_upgrade = self.text_encoder(prompt_feats_upgrade, self.prompt_learner.tokenized_prompts)
            prompt_loss_p2i_upgrade = self.sup(p_feats_upgrade, i_feats, batch['pids'], batch['pids'])
            prompt_loss_i2p_upgrade = self.sup(i_feats, p_feats_upgrade, batch['pids'], batch['pids'])
        lossG = prompt_loss_p2i_upgrade + prompt_loss_i2p_upgrade

        p_feats_upgrade_tse = self.texual_emb_layer(prompt_feats_upgrade.half(), caption_ids, atten_t)
        #prompt_loss_i_upgrade_tse, _ = objectives.compute_per_loss(i_tse_f, p_feats_upgrade_tse, batch['pids'], loss_type='TAL')
        #prompt_loss_t_upgrade_tse, _ = objectives.compute_per_loss(t_tse_f, p_feats_upgrade_tse, batch['pids'], loss_type='InfoNCE')
        #lossH = prompt_loss_i_upgrade_tse + prompt_loss_t_upgrade_tse

        with autocast():
            prompt_loss_p2i_upgrade_tse = self.sup(p_feats_upgrade_tse, i_tse_f, batch['pids'], batch['pids'])
            prompt_loss_i2p_upgrade_tse = self.sup(i_tse_f, p_feats_upgrade_tse, batch['pids'], batch['pids'])
        lossH = prompt_loss_p2i_upgrade_tse + prompt_loss_i2p_upgrade_tse

        return lossA.detach().cpu(), lossB.detach().cpu(), lossC.detach().cpu(), lossD.detach().cpu(), lossE.detach().cpu(), lossF.detach().cpu(), lossG.detach().cpu(), lossH.detach().cpu()

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0] #使用输入的查询（q）、键（k）和值（v）执行跨模态自注意力操作。这意味着模型将学习如何在不同的模态之间进行注意力分布
        x = x.permute(1, 0, 2)  # NLD -> LND 交换维度的顺序，以便适应 Transformer 模型的输入要求
        x = self.cross_modal_transformer(x) #执行跨模态 Transformer 操作
        x = x.permute(1, 0, 2)  # LND -> NLD 将张量的维度再次调整为原始形状

        x = self.ln_post(x) #Layer Normalization：再次对输出进行层标准化
        return x

    #def get_matching_loss_attr(self, image_embeds, text_embeds, label):
    #    bs = image_embeds.size(0)
    #    labels = []
    #    for i in range(label.size(1)):
    #        l = 1 - label[:, i]
    #        l = torch.where(l == 2, -1, l)
    #        labels.append(l)
    #        labels.append(label[:, i])
    #    labels = torch.stack(labels, dim=1)
    #    r = random.sample(range(0, text_embeds.size(0)), 5)
    #    ll = 0
    #    for t in r:
    #        text_embeds_0 = text_embeds[t].repeat(bs, 1, 1)
    #        x = self.cross_former(text_embeds_0, image_embeds, image_embeds)
    #        x = self.mlm_head(x)
    #        ll = ll + objectives.label_smooth_loss(x, labels[:, t])
    #        #ll = ll + objectives.compute_mlm(x, labels[:, t])
    #    return ll / 5

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)

        v_feats, l_feats = self.feature_fusion_layer(image_feats, text_feats)

        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        va_feats = v_feats[:, 0, :].float()
        la_feats = l_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() #text_feats->l_feats

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device)
     
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})

        label_hat_id = batch['label_hat_id'].to(i_feats.device)
        image_logits = self.classifier(va_feats.half()).float()  # 计算图像特征 i_feats 的分类器输出, i_feats 用于将张量中的数据类型转换为半精度浮点数类型
        text_logits = self.classifier(la_feats.half()).float()  # 计算文本特征 t_feats的分类器输出
        fusion = torch.cat([la_feats, va_feats], dim=-1)
        output_mm = self.classifier_f(fusion.half())
        criterion0 = nn.CrossEntropyLoss(reduction="mean")
        loss_f = criterion0(output_mm, batch['pids'])
        I = MMc(self.args)
        exp_logits, mask = I.mmc_2(la_feats, va_feats, text_logits, image_logits, batch['pids'])
        ret.update({'id_loss': (label_hat_id * (-0.1 * torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum()) + loss_f + objectives.compute_id(image_logits, text_logits, batch['pids'])* 1.0)).mean()})  # 计算 ID 损失并将结果添加到 ret 字典中，还乘以 self.args.id_loss_weight

        # mlm and id-tse will decrease the model performance
        #i_tse_logits = self.classifier_tse(i_tse_f.half()).float()  # 计算图像特征 i_feats 的分类器输出, i_feats 用于将张量中的数据类型转换为半精度浮点数类型
        #t_tse_logits = self.classifier_tse(t_tse_f.half()).float()  # 计算文本特征 t_feats的分类器输出
        #tse_fusion = torch.cat([t_tse_f, i_tse_f], dim=-1)
        #output_tse_mm = self.classifier_f_tse(tse_fusion.half())
        #loss_tse_f = criterion0(output_tse_mm, batch['pids'])
        #exp_logits_tse, mask_tse = I.mmc_2(t_tse_f, i_tse_f, t_tse_logits, i_tse_logits, batch['pids'])
        #ret.update({'id_tse_loss': (label_hat_id * (-0.1 * torch.log(((mask_tse * exp_logits_tse).sum() / exp_logits_tse.sum()) / mask_tse.sum()) + loss_tse_f + objectives.compute_id(i_tse_logits, t_tse_logits, batch['pids'])* 1.0)).mean()})  # 计算 ID 损失并将结果添加到 ret 字典中，还乘以 self.args.id_loss_weight

        #image_pred = torch.argmax(image_logits, dim=1)
        #text_pred = torch.argmax(text_logits, dim=1)
        #image_precision = (image_pred == batch['pids']).float().mean()
        #text_precision = (text_pred == batch['pids']).float().mean()
        #ret.update({'img_acc': image_precision})
        #ret.update({'txt_acc': text_precision})

        #image_tse_pred = torch.argmax(i_tse_logits, dim=1)
        #text_tse_pred = torch.argmax(t_tse_logits, dim=1)
        #image_tse_precision = (image_tse_pred == batch['pids']).float().mean()
        #text_tse_precision = (text_tse_pred == batch['pids']).float().mean()
        #ret.update({'img_tse_acc': image_tse_precision})
        #ret.update({'txt_tse_acc': text_tse_precision})

        mlm_ids = batch['mlm_ids']  # 从输入批次 batch 中获取 mlm_ids 数据
        mlm_feats, _ = self.base_model.encode_text(mlm_ids)
        x = self.cross_former(mlm_feats, image_feats, image_feats)
        x = self.mlm_head(x)  # [batch_size, text_len, num_colors] 将 x 通过 mlm_head 层，得到预测的概率分布，其中每个值对应于词汇表中的一个词
        scores = x.float().reshape(-1, self.args.vocab_size)  # 将预测的概率分布转换为形状为 (batch_size * text_len, vocab_size) 的得分矩阵
        mlm_labels = batch['mlm_labels'].reshape(-1)
        ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * 1.0})  # 计算 MLM 损失，并将结果添加到 ret 字典中，同时乘以 self.args.mlm_loss_weight

        #ret.update({'mlm_loss': self.get_matching_loss_attr(image_feats, mlm_feats, mlm_ids) * 1.0})

        #pred = scores.max(1)[1]
        #mlm_label_idx = torch.nonzero(mlm_labels)  # 这行代码的目的是找到非零元素的索引，也就是被遮盖的词语的索引
        #acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()  # 计算 MLM 的准确率
        #ret.update({'mlm_acc': acc})

        label_hat_p = batch['label_hat_p'].to(i_feats.device)
        with autocast():
            prompt_feats = self.prompt_learner(batch['pids']) #fusion is not effective
            #print(prompt_feats.shape)
            #p_feats = self.text_encoder(prompt_feats, self.prompt_learner.tokenized_prompts)
            #prompt_feats, _ = self.base_model.encode_text(prompts.long())
            p_feats = prompt_feats[torch.arange(prompt_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        #    prompt_loss_p2i = self.sup(p_feats, i_feats, batch['pids'], batch['pids'])
        #    prompt_loss_i2p = self.sup(i_feats, p_feats, batch['pids'], batch['pids'])
        #ret.update({'prompt_loss_bge': (label_hat_p * (prompt_loss_p2i + prompt_loss_i2p)).mean()})

            prompt_loss_i, _ = objectives.compute_per_loss(i_feats, p_feats, batch['pids'], loss_type='TAL')
            prompt_loss_t, _ = objectives.compute_per_loss(t_feats, p_feats, batch['pids'], loss_type='InfoNCE')
        ret.update({'prompt_loss_bge': (label_hat_p * prompt_loss_i).sum() + (label_hat_p * prompt_loss_t).sum()/label_hat_p.sum()})

        #with autocast():
        #    p_tse_f = self.texual_emb_layer(prompt_feats, caption_ids, atten_t)
        #    prompt_loss_p2i_tse = self.sup(p_tse_f, i_tse_f, batch['pids'], batch['pids'])
        #    prompt_loss_i2p_tse = self.sup(i_tse_f, p_tse_f, batch['pids'], batch['pids'])
        #ret.update({'prompt_loss_tse': (label_hat_p * (prompt_loss_p2i_tse + prompt_loss_i2p_tse)).mean()})
        #prompt_loss_i_tse, _ = objectives.compute_per_loss(i_tse_f, p_tse_f, batch['pids'], loss_type='TAL')
        #prompt_loss_t_tse, _ = objectives.compute_per_loss(t_tse_f, p_tse_f, batch['pids'], loss_type='InfoNCE')
        #ret.update({'prompt_loss_tse': (label_hat * prompt_loss_i_tse).sum() + (label_hat * prompt_loss_t_tse).sum()/label_hat.sum()})

        #label_hat_pv = batch['label_hat_pv'].to(i_feats.device)
        #text_diff = self.context_decoder(prompt_feats, image_feats)
        ##print(text_diff)
        #prompt_feats_upgrade = prompt_feats + text_diff #visual upgrade text prompts
        #p_feats_upgrade = prompt_feats_upgrade[torch.arange(prompt_feats_upgrade.shape[0]), caption_ids.argmax(dim=-1)].float()
        #prompt_loss_i_upgrade, _ = objectives.compute_per_loss(i_feats, p_feats_upgrade, batch['pids'], loss_type='TAL')
        #prompt_loss_t_upgrade, _ = objectives.compute_per_loss(t_feats, p_feats_upgrade, batch['pids'], loss_type='InfoNCE')
        #ret.update({'prompt_upgrade_loss_bge': (label_hat_pv * prompt_loss_i_upgrade).sum() + (label_hat_pv * prompt_loss_t_upgrade).sum() / label_hat_pv.sum()})

        #with autocast():
        #    p_feats_upgrade = self.text_encoder(prompt_feats_upgrade, self.prompt_learner.tokenized_prompts)
        #    prompt_loss_p2i_upgrade = self.sup(p_feats_upgrade, i_feats, batch['pids'], batch['pids'])
        #    prompt_loss_i2p_upgrade = self.sup(i_feats, p_feats_upgrade, batch['pids'], batch['pids'])
        #ret.update({'prompt_upgrade_loss_bge': (label_hat_pv * (prompt_loss_p2i_upgrade + prompt_loss_i2p_upgrade)).mean()})

        #p_feats_upgrade_tse = self.texual_emb_layer(prompt_feats_upgrade.half(), caption_ids, atten_t)
        #prompt_loss_i_upgrade_tse, _ = objectives.compute_per_loss(i_tse_f, p_feats_upgrade_tse, batch['pids'], loss_type='TAL')
        #prompt_loss_t_upgrade_tse, _ = objectives.compute_per_loss(t_tse_f, p_feats_upgrade_tse, batch['pids'], loss_type='InfoNCE')
        #ret.update({'prompt_upgrade_loss_tse': (label_hat_pv * prompt_loss_i_upgrade_tse).sum() + (label_hat_pv * prompt_loss_t_upgrade_tse).sum() / label_hat_pv.sum()})

        #with autocast():
        #   prompt_loss_p2i_upgrade_tse = self.sup(p_feats_upgrade_tse, i_tse_f, batch['pids'], batch['pids'])
        #   prompt_loss_i2p_upgrade_tse = self.sup(i_tse_f, p_feats_upgrade_tse, batch['pids'], batch['pids'])
        #ret.update({'prompt_upgrade_loss_bge': (label_hat_pv * (prompt_loss_p2i_upgrade_tse + prompt_loss_i2p_upgrade_tse)).mean()})

        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
