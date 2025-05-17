import os
import time
import math
import warnings
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import get_peft_model, LoraConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')
basepath = "../datasets"

class SFTDataset(Dataset):
    def __init__(self, data_path_lst, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len

        data_lst = []
        for data_path in data_path_lst:
            with open(data_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16)
                data_lst.append(data)
        data = np.concatenate(data_lst)
        data = data[:max_length * int(len(data) / max_length)]
        self.data = data.reshape(-1, max_length)
        print("train data.shape:{}".format(self.data.shape))
        print("SFTDataset finished.....")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)

class MyPretrainConfig(PretrainedConfig):
    model_type = "myllm"

    def __init__(
            self,
            dim: int=1024,
            n_layers: int=16,
            n_heads: int=16,
            n_kv_heads: int=8,
            vocab_size: int=19200,
            hidden_dim: int=None,
            multiple_of: int=64,
            norm_eps: float=1e-5,
            max_seq_len: int=512,
            dropout: float=0.0,
            flash_attn: bool=True,
            num_experts_per_tok=2,
            n_routed_experts=4,
            n_shared_experts: bool = True,
            scoring_func='softmax',
            aux_loss_alpha=0.01,
            seq_aux=True,
            norm_topk_prob=True,
            use_moe=False,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts        # 总的专家数量
        self.n_shared_experts = n_shared_experts        # 共享专家
        self.scoring_func = scoring_func                # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha            # 辅助损失的alpha参数
        self.seq_aux = seq_aux                          # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob            # 是否标准化top-k概率
        self.use_moe = use_moe
        super().__init__(**kwargs)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis

def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.reshape(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: MyPretrainConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
            self,
            x: torch.Tensor,
            pos_cis: torch.Tensor,
            use_kv_cache: bool = False,
            past_kv: Tuple[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        # QKV
        # inference
        if use_kv_cache:
            # 只计算最后一个token的Q
            current_token = x[:, -1:, :]

            if not past_kv:
                xq = self.wq(x)
                xk, xv = self.wk(x), self.wv(x)
            else:
                past_key, past_value = past_kv
                xq = torch.cat((torch.zeros_like(x[:, :-1, :]), self.wq(current_token)), dim=1)
                xk = torch.cat((past_key, self.wk(current_token)), dim=1)
                xv = torch.cat((past_value, self.wv(current_token)), dim=1)

            past_kv = (xk, xv)
        else:
            xq = self.wq(x)
            xk, xv = self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoEGate(nn.Module):
    def __init__(self, config: MyPretrainConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.reshape(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.reshape(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.reshape(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: MyPretrainConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )
            for _ in range(config.n_routed_experts)
        ])

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.reshape(-1, x.shape[-1])
        flat_topk_idx = topk_idx.reshape(-1)

        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.reshape(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.reshape(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.reshape(-1, 1)).reshape(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.reshape(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: MyPretrainConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )

    def forward(self, x, pos_cis, use_kv_cache=False, past_kv: Tuple[torch.Tensor] = None):
        attn_res, past_kv = self.attention(self.attention_norm(x), pos_cis, use_kv_cache, past_kv)
        h = x + attn_res
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv

class Transformer(PreTrainedModel):
    config_class = MyPretrainConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: MyPretrainConfig = None):
        super().__init__(params)
        if not params:
            params = MyPretrainConfig()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight  # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("pos_cis", pos_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None,
                use_kv_cache=False, past_kvs=None, **keyargs):
        if past_kvs is None:
            past_kvs = [None for _ in range(self.n_layers)]
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        pos_cis = self.pos_cis[:seqlen]
        for idx, layer in enumerate(self.layers):
            h, past_kvs[idx] = layer(h, pos_cis, use_kv_cache, past_kvs[idx])

        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :])  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)

        if use_kv_cache:
            return self.OUT, past_kvs
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.):
        index = idx.shape[1]
        use_kv_cache = True
        past_kvs = [None for _ in range(self.n_layers)]
        while idx.shape[1] < max_new_tokens - 1:
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx  # if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            inference_res = self(idx_cond, use_kv_cache=use_kv_cache, past_kvs=past_kvs)
            if use_kv_cache:
                logits, past_kvs = inference_res[0].logits, inference_res[1]
            else:
                logits = inference_res.logits

            logits = logits[:, -1, :]  # crop to just the final time step

            # Apply repetition penalty
            for token in set(idx.tolist()[0]):
                logits[:, token] /= repetition_penalty

            if temperature == 0.0:
                # "sample" the single most likely index
                __, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, __ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)
            # append sampled index to the running sequence and continue
            if idx_next == eos:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            if stream:
                yield idx[:, index:]

        if not stream:
            yield idx[:, index:]

    @torch.inference_mode()
    def eval_answer(self, idx):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        # forward the model to get the logits for the index in the sequence
        past_kvs = [None for _ in range(self.n_layers)]
        inference_res = self(idx_cond, use_kv_cache=False, past_kvs=past_kvs)
        logits = inference_res.logits
        logits = logits[:, -1, :]
        return logits

def get_lr(it, all):
    warmup_iters = 0
    lr_decay_iters = all
    min_lr = learning_rate / 10

    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        return list(lora_module_names)

    model = Transformer(lm_config)
    ckp = f'./out/pretrain_{lm_config.dim}.pth.{batch_size}'
    state_dict = torch.load(ckp, map_location=device_type)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)

    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        r=8,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(device_type)
    return model


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    lm_config = MyPretrainConfig()
    max_seq_len = lm_config.max_seq_len
    out_dir = 'out'
    epochs = 20             # 训练轮数
    batch_size = 8          # batch_size
    learning_rate = 1e-5    # 学习率
    device = 'cuda:0'       # or cpu
    dtype = 'bfloat16'
    save_dir = os.path.join(out_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    tokens_per_iter = batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = device if "cuda" in device else "cpu"
    print(f"device_type: {device_type}")
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast()
    )
    # -----------------------------------------------------------------------------

    # init model
    model = init_model()
    print(model)

    # -----init dataloader------
    data_path_list = [f'{basepath}/sft_data.bin']
    train_ds = SFTDataset(data_path_list, max_length=max_seq_len)
    train_sampler = None
    num_workers = 16  # 可以根据系统的 CPU 核心数来调整
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == dtype))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    accumulation_steps = 8
    iter_per_epoch = len(train_loader)
    for epoch in range(epochs):
        start_time = time.time()

        for step, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)

            # 设置学习率
            lr = get_lr(epoch * iter_per_epoch + step, epochs * iter_per_epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向传播和损失计算
            with ctx:
                out = model(X, Y)
                loss = out.last_loss

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度剪裁和更新参数
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                # 清零梯度
                optimizer.zero_grad(set_to_none=True)

            if step % 1000 == 0:
                spend_time = time.time() - start_time
                print(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch,
                        epochs,
                        step,
                        iter_per_epoch,
                        loss.item(),
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
                model.eval()
                ckp = f'{save_dir}/lora_sft_{lm_config.dim}.pth.{batch_size}'
                state_dict = model.state_dict()
                torch.save(state_dict, ckp)
                model.train()