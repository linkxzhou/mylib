"""
pre_transformer.py - 预训练 Transformer 模型实现

本文件实现了一个完整的 Transformer 语言模型，包含以下主要功能：

核心组件：
- MyPretrainConfig: 模型配置类，定义模型架构参数
- RMSNorm: RMS 归一化层，用于稳定训练
- Attention: 多头注意力机制，支持 Flash Attention 和 KV 缓存
- FeedForward: 前馈神经网络层，使用 SiLU 激活函数
- TransformerBlock: Transformer 块，包含注意力和前馈层
- Transformer: 完整的 Transformer 模型，支持预训练和推理

关键特性：
- 旋转位置编码 (RoPE): 相对位置编码机制
- 分组多查询注意力 (GQA): 优化的注意力机制
- KV 缓存: 推理时的性能优化
- 权重共享: 嵌入层和输出层权重共享
- 配置验证: 完整的参数验证和错误检查

验证功能：
- validate_config: 验证模型配置参数的有效性
- validate_model_inputs: 验证模型输入的合法性
- validate_model_state: 验证模型状态的一致性
- validate_training_step: 验证训练过程中的损失和参数

工具函数：
- count_parameters: 计算模型参数数量
- get_lr: 学习率调度器
- precompute_pos_cis: 预计算位置编码
- apply_rotary_emb: 应用旋转位置编码
- repeat_kv: 重复键值对以适应多查询注意力

使用示例：
    config = MyPretrainConfig(dim=1024, n_layers=16, n_heads=16)
    model = Transformer(config)
    output = model(input_tokens)
"""

import os
import math
import torch
from torch import nn
from transformers import PreTrainedModel
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from pre_pretrainconfig import (
    MyPretrainConfig, validate_config, validate_model_state, validate_model_inputs
)
from pre_moe import MOEFeedForward
from pre_attention import Attention, precompute_pos_cis
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RMSNorm(torch.nn.Module):
    """RMSNorm: A normalization layer that normalizes the input using the root mean square."""
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps # 防止除以零的微小值
        self.weight = nn.Parameter(torch.ones(dim)) # 可学习的权重参数，初始化为1

    def _norm(self, x):
        """计算输入的RMS归一化。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向传播，应用RMS归一化和权重。"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))   

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

        # 根据配置选择使用 MOE 还是普通前馈网络
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
        
        # 处理前馈网络输出，MOE 会返回辅助损失
        if hasattr(self.feed_forward, 'gate'):  # MOE 前馈网络
            ffn_out, aux_loss = self.feed_forward(self.ffn_norm(h))
            out = h + ffn_out
            return out, past_kv, aux_loss
        else:  # 普通前馈网络
            out = h + self.feed_forward(self.ffn_norm(h))
            return out, past_kv, None

class Transformer(PreTrainedModel):
    config_class = MyPretrainConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: MyPretrainConfig = None):
        super().__init__(params)
        self.params = MyPretrainConfig() if not params else params
        
        # 验证配置参数
        validate_config(self.params)
        
        self.vocab_size = self.params.vocab_size
        self.n_layers = self.params.n_layers

        self.tok_embeddings = nn.Embedding(self.params.vocab_size, self.params.dim)
        self.dropout = nn.Dropout(self.params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers): # 定义层级
            self.layers.append(TransformerBlock(layer_id, self.params))
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.output = nn.Linear(self.params.dim, self.params.vocab_size, bias=False)

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
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.params.n_layers))
        
        # 验证模型状态
        validate_model_state(self)

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
        
        # 验证输入
        if tokens is not None:
            validate_model_inputs(tokens, self.params)

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        pos_cis = self.pos_cis[:seqlen]
        
        # 收集 MOE 辅助损失
        aux_losses = []
        
        for idx, layer in enumerate(self.layers):
            # 统一处理：TransformerBlock 现在总是返回3个值
            h, past_kvs[idx], aux_loss = layer(h, pos_cis, use_kv_cache, past_kvs[idx])
            
            # 只有在 MoE 模式下且 aux_loss 不为 None 时才收集辅助损失
            if self.params.use_moe and aux_loss is not None:
                aux_losses.append(aux_loss)

        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            
            # 添加 MOE 辅助损失
            if aux_losses and self.params.use_moe:
                aux_loss_total = torch.stack(aux_losses).mean()
                self.last_loss = main_loss + aux_loss_total
                # 存储辅助损失信息用于监控
                self.aux_loss = aux_loss_total
            else:
                self.last_loss = main_loss
                self.aux_loss = None
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :])  # note: using list [-1] to preserve the time dim
            self.last_loss = None
            self.aux_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        if hasattr(self, 'aux_loss'):
            self.OUT.__setitem__('aux_loss', self.aux_loss)

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

def get_lr(it, all, learning_rate=1e-4):
    """计算当前迭代的学习率（余弦退火调度）。
    
    Args:
        it: 当前迭代次数
        all: 总迭代次数
        learning_rate: 基础学习率
        
    Returns:
        float: 当前迭代的学习率
    """
    # 验证输入参数
    if it < 0:
        raise ValueError(f"迭代次数不能为负数: {it}")
    if all <= 0:
        raise ValueError(f"总迭代次数必须大于0: {all}")
    if learning_rate <= 0:
        raise ValueError(f"学习率必须大于0: {learning_rate}")
    
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

def validate_training_step(model: 'Transformer', loss: torch.Tensor, step: int) -> None:
    """验证训练步骤的有效性。
    
    Args:
        model: Transformer 模型实例
        loss: 当前步骤的损失值
        step: 当前训练步骤
        
    Raises:
        ValueError: 当训练状态无效时抛出异常
    """
    if not isinstance(loss, torch.Tensor):
        raise ValueError(f"损失必须是 torch.Tensor 类型，当前类型: {type(loss)}")
    
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(f"检测到无效的损失值: {loss.item()}")
    
    if loss.item() < 0:
        print(f"[WARNING] 损失值为负数: {loss.item()}，这可能表示训练有问题")
    
    if step < 0:
        raise ValueError(f"训练步骤不能为负数: {step}")
    
    # 检查模型参数是否包含 NaN 或 Inf
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise ValueError(f"参数 {name} 包含 NaN 值")
        if torch.isinf(param).any():
            raise ValueError(f"参数 {name} 包含 Inf 值")

def count_parameters(model: 'Transformer') -> dict:
    """计算模型参数数量。
    
    Args:
        model: Transformer 模型实例
        
    Returns:
        dict: 包含参数统计信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设 float32
    }
    
    return param_info

def init_model(config: MyPretrainConfig, device: str) -> Transformer:
    """初始化模型。
    
    Args:
        config: 模型配置
        device: 设备类型
        
    Returns:
        Transformer: 初始化的模型实例
    """
    # 验证配置
    validate_config(config)
    
    # 创建模型
    model = Transformer(config)
    
    # 计算参数数量
    param_info = count_parameters(model)
    print(f'LLM总参数量：{param_info["total_parameters"] / 1e6:.3f} 百万')
    
    # 移动到指定设备
    model = model.to(device)
    
    return model

def load_checkpoint(model: Transformer, optimizer, checkpoint_path: str, device: str = 'cpu') -> tuple:
    """加载检查点。
    
    Args:
        model: 模型实例
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    Returns:
        tuple: (epoch, step, loss)
    """
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, 0, float('inf')
    
    try:
        # 使用安全全局变量上下文管理器加载检查点
        with torch.serialization.safe_globals([MyPretrainConfig]):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # 加载模型状态
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        
        # 加载优化器状态
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"成功加载检查点: epoch={epoch}, step={step}, loss={loss:.4f}")
        return epoch, step, loss
        
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return 0, 0, float('inf')

def estimate_memory_usage(config, batch_size, max_seq_len, dtype='bfloat16'):
    """
    预估模型训练时的显存和内存使用量
    
    Args:
        config: MyPretrainConfig 实例
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        dtype: 数据类型
    
    Returns:
        dict: 包含各项内存使用估算的字典
    """
    # 数据类型字节数
    dtype_bytes = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int64': 8,
        'int32': 4
    }
    
    model_dtype_bytes = dtype_bytes.get(dtype, 2)
    
    # 1. 模型参数内存
    # Embedding层参数
    embedding_params = config.vocab_size * config.dim
    
    # Transformer层参数
    # 每层包含: attention (qkv + o) + feed_forward (gate + up + down) + layer_norm
    attention_params_per_layer = (
        config.dim * config.dim * 3 +  # qkv projection
        config.dim * config.dim        # output projection
    )
    
    # Feed forward参数 (SwiGLU)
    hidden_dim = config.hidden_dim or int(2.0 / 3.0 * 4 * config.dim)
    hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
    
    ffn_params_per_layer = (
        config.dim * hidden_dim * 2 +  # gate and up projections
        hidden_dim * config.dim        # down projection
    )
    
    # Layer norm参数
    norm_params_per_layer = config.dim * 2  # attention norm + ffn norm
    
    transformer_params = config.n_layers * (
        attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
    )
    
    # 输出层参数 (如果不共享embedding)
    output_params = 0 if config.tie_word_embeddings else config.vocab_size * config.dim
    
    # 最终norm参数
    final_norm_params = config.dim
    
    total_params = embedding_params + transformer_params + output_params + final_norm_params
    model_memory = total_params * model_dtype_bytes
    
    # 2. 激活值内存 (前向传播)
    # 输入embedding
    input_activations = batch_size * max_seq_len * config.dim * model_dtype_bytes
    
    # 每层的激活值 (attention + ffn)
    attention_activations_per_layer = (
        batch_size * config.n_heads * max_seq_len * max_seq_len * model_dtype_bytes +  # attention scores
        batch_size * max_seq_len * config.dim * model_dtype_bytes * 3  # qkv outputs
    )
    
    ffn_activations_per_layer = (
        batch_size * max_seq_len * hidden_dim * model_dtype_bytes * 2  # gate and up outputs
    )
    
    layer_activations = config.n_layers * (attention_activations_per_layer + ffn_activations_per_layer)
    
    # 3. 梯度内存 (反向传播)
    gradient_memory = model_memory  # 梯度与参数同样大小
    
    # 4. 优化器状态内存 (AdamW)
    # AdamW需要存储momentum和variance，每个参数需要额外8字节(两个float32)
    optimizer_memory = total_params * 8
    
    # 5. 数据加载内存
    # 输入数据 (X, Y)
    data_memory = batch_size * max_seq_len * dtype_bytes['int64'] * 2
    
    # 6. 其他缓存和临时内存 (估算为模型参数的20%)
    misc_memory = model_memory * 0.2
    
    # 总显存使用量
    total_gpu_memory = (
        model_memory + 
        input_activations + 
        layer_activations + 
        gradient_memory + 
        optimizer_memory + 
        data_memory + 
        misc_memory
    )
    
    # 系统内存主要用于数据加载和预处理
    system_memory = data_memory * 4  # 估算为数据内存的4倍
    
    def format_bytes(bytes_val):
        """格式化字节数为可读格式"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"
    
    # 构建格式化的字符串返回
    result_str = f"""内存使用量预估报告:
{'='*50}
模型参数总数: {format_bytes(total_params)}
模型参数内存: {format_bytes(model_memory)}
激活值内存: {format_bytes(input_activations + layer_activations)}
梯度内存: {format_bytes(gradient_memory)}
优化器内存: {format_bytes(optimizer_memory)}
数据内存: {format_bytes(data_memory)}
其他缓存: {format_bytes(misc_memory)}
总显存需求: {format_bytes(total_gpu_memory)}
系统内存需求: {format_bytes(system_memory)}
{'='*50}
原始数值 (字节):
- 模型内存: {model_memory:,}
- 总显存: {total_gpu_memory:,}
- 系统内存: {system_memory:,}"""
    
    return result_str
