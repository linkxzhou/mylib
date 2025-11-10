import os
import torch
from transformers import PretrainedConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MyPretrainConfig(PretrainedConfig):
    """自定义预训练配置类，包含模型架构和训练参数。"""
    model_type = "myllm"

    def __init__(
            self,
            dim: int = 1024,
            n_layers: int = 16,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 19200,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            # MOE 相关参数
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 8,
            n_shared_experts: int = None,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = False,
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
        
        # MOE 相关参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        
        print(f"kwargs: {kwargs}")
        super().__init__(**kwargs)
    
    def get_config_info(self) -> str:
        """返回配置参数的格式化字符串。
        
        Returns:
            str: 格式化的配置参数信息
        """
        config_str = f"""MyPretrainConfig 参数信息:
========================================
模型架构参数:
  dim: {self.dim} (模型隐藏层维度)
  n_layers: {self.n_layers} (Transformer层数)
  n_heads: {self.n_heads} (注意力头数)
  n_kv_heads: {self.n_kv_heads} (键值对注意力头数)
  vocab_size: {self.vocab_size} (词汇表大小)
  hidden_dim: {self.hidden_dim} (前馈网络隐藏层维度)
  multiple_of: {self.multiple_of} (隐藏层维度的倍数约束)
  norm_eps: {self.norm_eps} (层归一化的epsilon值)
  max_seq_len: {self.max_seq_len} (最大序列长度)
  dropout: {self.dropout} (Dropout概率)
  flash_attn: {self.flash_attn} (是否使用Flash Attention优化)

MOE 相关参数:
  use_moe: {self.use_moe} (是否使用混合专家模型)
  num_experts_per_tok: {self.num_experts_per_tok} (每个token选择的专家数量)
  n_routed_experts: {self.n_routed_experts} (路由专家总数)
  n_shared_experts: {self.n_shared_experts} (共享专家数量)
  scoring_func: {self.scoring_func} (门控评分函数)
  aux_loss_alpha: {self.aux_loss_alpha} (辅助损失权重)
  seq_aux: {self.seq_aux} (是否使用序列级辅助损失)
  norm_topk_prob: {self.norm_topk_prob} (是否归一化topk概率)

计算得出的参数:
  head_dim: {self.dim // self.n_heads} (每个注意力头的维度)
  effective_hidden_dim: {self.hidden_dim or int(2 * self.dim / 3)} (实际使用的前馈网络隐藏层维度)
========================================"""
        return config_str

def validate_config(config: MyPretrainConfig) -> None:
    """验证模型配置参数的有效性。
    
    Args:
        config: 模型配置对象
        
    Raises:
        ValueError: 当配置参数无效时抛出异常
    """
    # ... existing code ...
    
    # 验证序列长度的合理性
    if config.max_seq_len > 32768:
        print(f"[WARNING] 序列长度 ({config.max_seq_len}) 很大，可能导致内存不足")
    
    # 验证 MOE 相关参数
    if config.use_moe:
        if config.num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok 必须大于 0，当前值: {config.num_experts_per_tok}")
        
        if config.n_routed_experts <= 0:
            raise ValueError(f"n_routed_experts 必须大于 0，当前值: {config.n_routed_experts}")
        
        if config.num_experts_per_tok > config.n_routed_experts:
            raise ValueError(f"num_experts_per_tok ({config.num_experts_per_tok}) 不能大于 n_routed_experts ({config.n_routed_experts})")
        
        if config.n_shared_experts is not None and config.n_shared_experts < 0:
            raise ValueError(f"n_shared_experts 必须大于等于 0 或为 None，当前值: {config.n_shared_experts}")
        
        if config.scoring_func not in ['softmax']:
            raise ValueError(f"不支持的评分函数: {config.scoring_func}，支持的函数: ['softmax']")
        
        if not (0.0 <= config.aux_loss_alpha <= 1.0):
            raise ValueError(f"aux_loss_alpha 必须在 [0.0, 1.0] 范围内，当前值: {config.aux_loss_alpha}")
        
        # 验证专家数量的合理性
        if config.n_routed_experts > 64:
            print(f"[WARNING] 专家数量 ({config.n_routed_experts}) 很大，可能影响训练效率")
        
        if config.num_experts_per_tok == 1:
            print(f"[WARNING] 每个token只选择1个专家，MOE的效果可能有限")
    
    print(f"[INFO] 配置验证通过，模型架构: {'MOE' if config.use_moe else '标准'} Transformer")

def validate_model_inputs(tokens: torch.Tensor, config: MyPretrainConfig) -> None:
    """验证模型输入的有效性。
    
    Args:
        tokens: 输入的 token 张量
        config: 模型配置对象
        
    Raises:
        ValueError: 当输入无效时抛出异常
    """
    if tokens is None:
        raise ValueError("输入 tokens 不能为 None")
    
    if not isinstance(tokens, torch.Tensor):
        raise ValueError(f"tokens 必须是 torch.Tensor 类型，当前类型: {type(tokens)}")
    
    if tokens.dim() != 2:
        raise ValueError(f"tokens 必须是 2D 张量 (batch_size, seq_len)，当前维度: {tokens.dim()}")
    
    batch_size, seq_len = tokens.shape
    
    if seq_len > config.max_seq_len:
        raise ValueError(f"序列长度 ({seq_len}) 超过最大序列长度 ({config.max_seq_len})")
    
    if torch.any(tokens < 0) or torch.any(tokens >= config.vocab_size):
        raise ValueError(f"tokens 包含无效的词汇表索引，应在 [0, {config.vocab_size}) 范围内")

def validate_model_state(model: 'Transformer') -> None:
    """验证模型状态的有效性。
    
    Args:
        model: Transformer 模型实例
        
    Raises:
        ValueError: 当模型状态无效时抛出异常
    """
    if not hasattr(model, 'params'):
        raise ValueError("模型缺少 params 属性")
    
    if not hasattr(model, 'tok_embeddings'):
        raise ValueError("模型缺少 tok_embeddings 层")
    
    if not hasattr(model, 'layers'):
        raise ValueError("模型缺少 layers 属性")
    
    if len(model.layers) != model.params.n_layers:
        raise ValueError(f"模型层数 ({len(model.layers)}) 与配置不匹配 ({model.params.n_layers})")
    
    # 验证嵌入层维度
    if model.tok_embeddings.embedding_dim != model.params.dim:
        raise ValueError(f"嵌入层维度 ({model.tok_embeddings.embedding_dim}) 与配置不匹配 ({model.params.dim})")
    
    if model.tok_embeddings.num_embeddings != model.params.vocab_size:
        raise ValueError(f"词汇表大小 ({model.tok_embeddings.num_embeddings}) 与配置不匹配 ({model.params.vocab_size})")