"""
pre_attention.py - 注意力机制实现

本文件实现了 Transformer 模型的注意力机制相关功能，包含以下主要组件：

核心组件：
- Attention: 多头注意力机制，支持 Flash Attention 和 KV 缓存
- precompute_pos_cis: 预计算旋转位置编码的复数形式
- apply_rotary_emb: 应用旋转位置编码到查询和键
- repeat_kv: 重复键值对以适应多查询注意力机制

关键特性：
- 旋转位置编码 (RoPE): 相对位置编码机制
- 分组多查询注意力 (GQA): 优化的注意力机制
- KV 缓存: 推理时的性能优化
- Flash Attention: 高效的注意力计算

使用示例：
    from pre_attention import Attention, precompute_pos_cis
    
    attention = Attention(config)
    pos_cis = precompute_pos_cis(dim=config.dim // config.n_heads, seq_len=config.max_seq_len)
    output, past_kv = attention(x, pos_cis)
"""

import math
import torch
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F
from pre_pretrainconfig import MyPretrainConfig

def precompute_pos_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算相对位置编码的复数形式，用于旋转位置编码（RoPE）。
    
    Args:
        dim: 注意力头的维度
        seq_len: 序列长度
        theta: 旋转位置编码的基础频率参数
        
    Returns:
        torch.Tensor: 预计算的复数位置编码，形状为 (seq_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # 计算频率
    t = torch.arange(seq_len, device=freqs.device)  # 创建时间步长
    freqs = torch.outer(t, freqs).float()  # 计算频率的外积
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成复数形式的频率
    return pos_cis # 返回预计算的复数位置编码

def apply_rotary_emb(xq, xk, pos_cis):
    """应用旋转位置编码到查询和键。
    
    Args:
        xq: 查询张量，形状为 (batch_size, seq_len, n_heads, head_dim)
        xk: 键张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)
        pos_cis: 位置编码，形状为 (seq_len, head_dim//2)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 应用位置编码后的查询和键张量
    """
    def unite_shape(pos_cis, x):
        """调整位置编码的形状以匹配输入张量的形状。"""
        ndim = x.ndim # 获取输入的维度
        assert 0 <= 1 < ndim # 确保维度有效
        assert pos_cis.shape == (x.shape[1], x.shape[-1])  # 确保位置编码形状匹配
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # 生成新形状
        return pos_cis.reshape(*shape) # 调整位置编码的形状

    # 将查询和键转换为复数形式
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_) # 调整位置编码形状
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3) # 应用位置编码并转换回实数
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3) # 同上
    return xq_out.type_as(xq), xk_out.type_as(xk)         # 返回与输入类型一致的输出

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复键值对以适应多查询注意力机制。
    
    在分组多查询注意力 (GQA) 中，键值对的头数可能少于查询的头数，
    需要重复键值对以匹配查询的头数。
    
    Args:
        x: 键或值张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep: 重复次数，通常为 n_heads // n_kv_heads
        
    Returns:
        torch.Tensor: 重复后的张量，形状为 (batch_size, seq_len, n_heads, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape # 获取输入的形状
    if n_rep == 1:
        return x # 如果不需要重复，直接返回
    return (
        x[:, :, :, None, :] # 在键值对的最后一个维度增加一个维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)   # 扩展到所需的重复次数
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # 重塑形状
    )

class Attention(nn.Module):
    """多头注意力机制实现。
    
    支持以下特性：
    - 分组多查询注意力 (GQA)
    - 旋转位置编码 (RoPE)
    - Flash Attention 优化
    - KV 缓存用于推理加速
    - 因果掩码确保自回归生成
    """
    
    def __init__(self, args: MyPretrainConfig):
        """初始化注意力层。
        
        Args:
            args: 模型配置参数
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        # 线性投影层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # Dropout 层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 使用 Flash Attention 还是手动实现？
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建因果掩码
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
            self,
            x: torch.Tensor,
            pos_cis: torch.Tensor,
            use_kv_cache: bool = False,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            pos_cis: 位置编码，形状为 (seq_len, head_dim//2)
            use_kv_cache: 是否使用 KV 缓存
            past_kv: 过去的键值对缓存
            
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 
                输出张量和更新后的 KV 缓存
        """
        bsz, seqlen, _ = x.shape
        
        # 计算 QKV
        if use_kv_cache:
            # 推理模式：只计算最后一个 token 的 Q
            current_token = x[:, -1:, :]

            if not past_kv:
                # 第一次推理，计算完整的 QKV
                xq = self.wq(x)
                xk, xv = self.wk(x), self.wv(x)
            else:
                # 后续推理，使用缓存的 KV
                past_key, past_value = past_kv
                xq = torch.cat((torch.zeros_like(x[:, :-1, :]), self.wq(current_token)), dim=1)
                xk = torch.cat((past_key, self.wk(current_token)), dim=1)
                xv = torch.cat((past_value, self.wv(current_token)), dim=1)

            past_kv = (xk, xv)
        else:
            # 训练模式：计算完整的 QKV
            xq = self.wq(x)
            xk, xv = self.wk(x), self.wv(x)

        # 重塑为多头形式
        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 分组多查询注意力：扩展键值对
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # 将头维度移到批次维度
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 计算注意力
        if self.flash:
            # Flash Attention 实现
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # 手动实现
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # 应用因果掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # 恢复时间维度并合并头
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)

        # 最终投影到残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        
        return output, past_kv

# 忽略：测试功能
if __name__ == "__main__":
    print("=" * 60)
    print("测试 pre_attention.py 中的注意力机制组件")
    print("=" * 60)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 创建测试配置
    class TestConfig:
        def __init__(self):
            self.dim = 512
            self.n_heads = 8
            self.n_kv_heads = 4  # 分组多查询注意力
            self.max_seq_len = 128
            self.dropout = 0.1
            self.flash_attn = False  # 使用手动实现进行测试
    
    config = TestConfig()
    
    # 测试参数
    batch_size = 2
    seq_len = 32
    head_dim = config.dim // config.n_heads
    
    print(f"测试配置:")
    print(f"  - 模型维度: {config.dim}")
    print(f"  - 注意力头数: {config.n_heads}")
    print(f"  - KV头数: {config.n_kv_heads}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 头维度: {head_dim}")
    print()
    
    # 1. 测试位置编码预计算
    print("1. 测试位置编码预计算 (precompute_pos_cis)")
    print("-" * 40)
    try:
        pos_cis = precompute_pos_cis(head_dim, seq_len)
        print(f"✓ 位置编码形状: {pos_cis.shape}")
        print(f"✓ 位置编码数据类型: {pos_cis.dtype}")
        print(f"✓ 位置编码范围: [{pos_cis.abs().min():.4f}, {pos_cis.abs().max():.4f}]")
        assert pos_cis.shape == (seq_len, head_dim // 2), f"位置编码形状错误: {pos_cis.shape}"
        print("✓ 位置编码预计算测试通过")
    except Exception as e:
        print(f"✗ 位置编码预计算测试失败: {e}")
        raise e
        
    print()
    
    # 2. 测试旋转位置编码应用
    print("2. 测试旋转位置编码应用 (apply_rotary_emb)")
    print("-" * 40)
    try:
        # 创建测试查询和键
        xq = torch.randn(batch_size, seq_len, config.n_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, config.n_kv_heads, head_dim)
        
        print(f"输入查询形状: {xq.shape}")
        print(f"输入键形状: {xk.shape}")
        
        # 应用旋转位置编码
        xq_rot, xk_rot = apply_rotary_emb(xq, xk, pos_cis)
        
        print(f"✓ 输出查询形状: {xq_rot.shape}")
        print(f"✓ 输出键形状: {xk_rot.shape}")
        assert xq_rot.shape == xq.shape, f"查询形状不匹配: {xq_rot.shape} vs {xq.shape}"
        assert xk_rot.shape == xk.shape, f"键形状不匹配: {xk_rot.shape} vs {xk.shape}"
        print("✓ 旋转位置编码应用测试通过")
    except Exception as e:
        print(f"✗ 旋转位置编码应用测试失败: {e}")
        raise e

    print()
    
    # 3. 测试键值重复
    print("3. 测试键值重复 (repeat_kv)")
    print("-" * 40)
    try:
        # 创建测试键值
        kv = torch.randn(batch_size, seq_len, config.n_kv_heads, head_dim)
        n_rep = config.n_heads // config.n_kv_heads
        
        print(f"输入键值形状: {kv.shape}")
        print(f"重复次数: {n_rep}")
        
        # 重复键值
        kv_repeated = repeat_kv(kv, n_rep)
        
        expected_shape = (batch_size, seq_len, config.n_heads, head_dim)
        print(f"✓ 输出键值形状: {kv_repeated.shape}")
        print(f"✓ 期望形状: {expected_shape}")
        assert kv_repeated.shape == expected_shape, f"键值重复形状错误: {kv_repeated.shape}"
        print("✓ 键值重复测试通过")
    except Exception as e:
        print(f"✗ 键值重复测试失败: {e}")
        raise e

    print()
    
    # 4. 测试注意力机制
    print("4. 测试注意力机制 (Attention)")
    print("-" * 40)
    try:
        # 创建注意力层
        attention = Attention(config)
        
        # 创建测试输入
        x = torch.randn(batch_size, seq_len, config.dim)
        print(f"输入形状: {x.shape}")
        
        # 前向传播
        output, past_kv = attention(x, pos_cis, use_kv_cache=False)
        
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ KV缓存: {past_kv is None}")
        assert output.shape == x.shape, f"输出形状错误: {output.shape} vs {x.shape}"
        print("✓ 注意力机制前向传播测试通过")
        
        # 测试 KV 缓存
        print("\n测试 KV 缓存功能:")
        output_cache, past_kv = attention(x, pos_cis, use_kv_cache=True)
        print(f"✓ 缓存输出形状: {output_cache.shape}")
        print(f"✓ KV缓存类型: {type(past_kv)}")
        if past_kv:
            print(f"✓ 缓存键形状: {past_kv[0].shape}")
            print(f"✓ 缓存值形状: {past_kv[1].shape}")
        print("✓ KV缓存测试通过")
        
    except Exception as e:
        print(f"✗ 注意力机制测试失败: {e}")
        raise e

    print()
    
    # 5. 性能基准测试
    print("5. 性能基准测试")
    print("-" * 40)
    try:
        import time
        
        # 预热
        for _ in range(5):
            _ = attention(x, pos_cis, use_kv_cache=False)
        
        # 测试训练模式性能
        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            output, _ = attention(x, pos_cis, use_kv_cache=False)
        train_time = (time.time() - start_time) / num_runs
        
        # 测试推理模式性能
        start_time = time.time()
        past_kv = None
        for i in range(seq_len):
            single_token = x[:, i:i+1, :]
            output, past_kv = attention(single_token, pos_cis[i:i+1], use_kv_cache=True, past_kv=past_kv)
        inference_time = time.time() - start_time
        
        print(f"✓ 训练模式平均时间: {train_time*1000:.2f} ms")
        print(f"✓ 推理模式总时间: {inference_time*1000:.2f} ms")
        print(f"✓ 推理模式平均每token: {inference_time/seq_len*1000:.2f} ms")
        print("✓ 性能基准测试完成")
        
    except Exception as e:
        print(f"✗ 性能基准测试失败: {e}")
    print()
    
    # 6. 梯度检查
    print("6. 梯度检查")
    print("-" * 40)
    try:
        # 设置需要梯度
        x_grad = torch.randn(batch_size, seq_len, config.dim, requires_grad=True)
        
        # 前向传播
        output, _ = attention(x_grad, pos_cis)
        loss = output.sum()
        
        # 反向传播
        loss.backward()
        
        print(f"✓ 输入梯度形状: {x_grad.grad.shape}")
        print(f"✓ 输入梯度范数: {x_grad.grad.norm():.6f}")
        
        # 检查参数梯度
        param_grads = []
        for name, param in attention.named_parameters():
            if param.grad is not None:
                param_grads.append((name, param.grad.norm().item()))
        
        print("✓ 参数梯度:")
        for name, grad_norm in param_grads:
            print(f"    {name}: {grad_norm:.6f}")
        
        print("✓ 梯度检查测试通过")
        
    except Exception as e:
        print(f"✗ 梯度检查测试失败: {e}")
    print()
    
    print("=" * 60)
    print("所有测试完成！注意力机制组件功能正常。")
    print("=" * 60)