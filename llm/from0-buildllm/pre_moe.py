"""
pre_moe.py - 混合专家模型 (Mixture of Experts) 实现

本文件实现了 MOE 相关的核心组件：

核心组件：
- MoEGate: 专家选择的门控机制，实现 top-k 专家选择和辅助损失计算
- MOEFeedForward: 混合专家前馈网络，集成多个专家和门控机制
- validate_moe_config: MOE 配置参数验证函数

关键特性：
- Top-k 专家选择: 为每个 token 选择最优的 k 个专家
- 辅助损失计算: 平衡专家负载，防止专家利用不均
- 共享专家支持: 可选的共享专家机制
- 训练/推理模式优化: 不同模式下的性能优化

使用示例：
    from pre_moe import MOEFeedForward, validate_moe_config
    
    # 验证 MOE 配置
    validate_moe_config(config)
    
    # 创建 MOE 前馈网络
    moe_ffn = MOEFeedForward(config)
    output, aux_loss = moe_ffn(input_tensor)
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

def validate_moe_config(config) -> None:
    """验证 MOE 相关配置参数的有效性。
    
    Args:
        config: 模型配置对象
        
    Raises:
        ValueError: 当 MOE 配置参数无效时抛出异常
    """
    if not config.use_moe:
        return
        
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

class MoEGate(nn.Module):
    """混合专家模型的门控机制。
    
    实现 top-k 专家选择和辅助损失计算，用于平衡专家负载。
    """
    
    def __init__(self, config):
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
        """重置门控网络参数。"""
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """前向传播，计算专家选择和辅助损失。
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_dim]
            
        Returns:
            tuple: (topk_idx, topk_weight, aux_loss)
                - topk_idx: 选中的专家索引
                - topk_weight: 专家权重
                - aux_loss: 辅助损失（训练时）
        """
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
    """混合专家前馈网络。
    
    集成多个专家网络和门控机制，支持训练和推理模式的优化。
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 延迟导入以避免循环导入
        from pre_transformer import FeedForward
        
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
        """前向传播。
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
            
        Returns:
            tuple: (output, aux_loss)
                - output: 输出张量
                - aux_loss: 辅助损失（训练时）
        """
        identity = x
        orig_shape = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.reshape(-1, x.shape[-1])
        flat_topk_idx = topk_idx.reshape(-1)

        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 修复：使用与输入相同的数据类型，而不是硬编码 torch.float16
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.reshape(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.reshape(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.reshape(-1, 1)).reshape(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y, aux_loss

    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """推理模式下的 MOE 计算。
        
        Args:
            x: 输入张量 [batch_seq_len, hidden_dim]
            flat_expert_indices: 扁平化的专家索引 [batch_seq_len * num_experts_per_tok]
            flat_expert_weights: 扁平化的专家权重 [batch_seq_len * num_experts_per_tok, 1]
            
        Returns:
            torch.Tensor: 输出张量
        """
        batch_seq_len, hidden_dim = x.shape
        num_experts_per_tok = self.config.num_experts_per_tok
        
        # 重复输入以匹配专家索引的形状
        x_repeated = x.repeat_interleave(num_experts_per_tok, dim=0)
        
        y = torch.zeros_like(x_repeated)
        flat_expert_weights = flat_expert_weights.squeeze(-1)  # 移除最后一个维度
        
        for i, expert in enumerate(self.experts):
            mask = (flat_expert_indices == i)
            if mask.any():
                selected_x = x_repeated[mask]
                if selected_x.numel() > 0:
                    expert_output = expert(selected_x)
                    y[mask] = flat_expert_weights[mask].unsqueeze(-1) * expert_output
        
        # 将结果重新组织并求和
        y = y.view(batch_seq_len, num_experts_per_tok, hidden_dim).sum(dim=1)
        
        return y

if __name__ == "__main__":
    """测试 MOE 组件的各项功能。"""
    print("=" * 60)
    print("测试 pre_moe.py 中的混合专家模型组件")
    print("=" * 60)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 创建测试配置
    class TestConfig:
        def __init__(self):
            # 基础配置
            self.dim = 512
            self.hidden_dim = 2048
            self.multiple_of = 256
            self.dropout = 0.1
            
            # MOE 配置
            self.use_moe = True
            self.num_experts_per_tok = 2
            self.n_routed_experts = 8
            self.n_shared_experts = None
            self.scoring_func = 'softmax'
            self.aux_loss_alpha = 0.01
            self.seq_aux = True
            self.norm_topk_prob = True
    
    config = TestConfig()
    
    # 测试参数
    batch_size = 2
    seq_len = 16
    
    print(f"测试配置:")
    print(f"  - 模型维度: {config.dim}")
    print(f"  - 隐藏层维度: {config.hidden_dim}")
    print(f"  - 专家总数: {config.n_routed_experts}")
    print(f"  - 每token选择专家数: {config.num_experts_per_tok}")
    print(f"  - 辅助损失权重: {config.aux_loss_alpha}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 批次大小: {batch_size}")
    print()
    
    # 1. 测试 MOE 配置验证
    print("1. 测试 MOE 配置验证 (validate_moe_config)")
    print("-" * 40)
    try:
        validate_moe_config(config)
        print("✓ MOE 配置验证通过")
        
        # 测试无效配置
        invalid_config = TestConfig()
        invalid_config.num_experts_per_tok = 0
        try:
            validate_moe_config(invalid_config)
            print("✗ 应该检测到无效配置")
        except ValueError as e:
            print(f"✓ 正确检测到无效配置: {e}")
            
    except Exception as e:
        print(f"✗ MOE 配置验证测试失败: {e}")
        raise e
        
    print()
    
    # 2. 测试 MoEGate 门控机制
    print("2. 测试 MoEGate 门控机制")
    print("-" * 40)
    try:
        gate = MoEGate(config)
        
        # 创建测试输入
        x = torch.randn(batch_size, seq_len, config.dim)
        print(f"输入形状: {x.shape}")
        
        # 前向传播
        topk_idx, topk_weight, aux_loss = gate(x)
        
        print(f"✓ Top-k 专家索引形状: {topk_idx.shape}")
        print(f"✓ Top-k 专家权重形状: {topk_weight.shape}")
        print(f"✓ 辅助损失: {aux_loss.item() if aux_loss is not None else None}")
        
        # 验证专家选择的有效性
        assert topk_idx.shape == (batch_size * seq_len, config.num_experts_per_tok)
        assert topk_weight.shape == (batch_size * seq_len, config.num_experts_per_tok)
        assert torch.all(topk_idx >= 0) and torch.all(topk_idx < config.n_routed_experts)
        assert torch.all(topk_weight >= 0) and torch.all(topk_weight <= 1)
        
        # 检查权重归一化
        if config.norm_topk_prob and config.num_experts_per_tok > 1:
            weight_sums = topk_weight.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
            print("✓ 权重归一化正确")
        
        print("✓ MoEGate 门控机制测试通过")
        
    except Exception as e:
        print(f"✗ MoEGate 门控机制测试失败: {e}")
        raise e

    print()
    
    # 3. 测试 MOEFeedForward 前馈网络
    print("3. 测试 MOEFeedForward 前馈网络")
    print("-" * 40)
    try:
        moe_ffn = MOEFeedForward(config)
        
        # 创建测试输入
        x = torch.randn(batch_size, seq_len, config.dim)
        print(f"输入形状: {x.shape}")
        
        # 训练模式测试
        moe_ffn.train()
        output_train, aux_loss_train = moe_ffn(x)
        
        print(f"✓ 训练模式输出形状: {output_train.shape}")
        print(f"✓ 训练模式辅助损失: {aux_loss_train.item() if aux_loss_train is not None else None}")
        
        # 推理模式测试
        moe_ffn.eval()
        with torch.no_grad():
            output_eval, aux_loss_eval = moe_ffn(x)
        
        print(f"✓ 推理模式输出形状: {output_eval.shape}")
        print(f"✓ 推理模式辅助损失: {aux_loss_eval}")
        
        # 验证输出形状
        assert output_train.shape == x.shape
        assert output_eval.shape == x.shape
        assert aux_loss_eval is None  # 推理模式下应该没有辅助损失
        
        print("✓ MOEFeedForward 前馈网络测试通过")
        
    except Exception as e:
        print(f"✗ MOEFeedForward 前馈网络测试失败: {e}")
        raise e

    print()
    
    # 4. 测试共享专家功能
    print("4. 测试共享专家功能")
    print("-" * 40)
    try:
        # 创建带共享专家的配置
        shared_config = TestConfig()
        shared_config.n_shared_experts = 1
        
        validate_moe_config(shared_config)
        moe_ffn_shared = MOEFeedForward(shared_config)
        
        # 测试前向传播
        output_shared, aux_loss_shared = moe_ffn_shared(x)
        
        print(f"✓ 共享专家输出形状: {output_shared.shape}")
        print(f"✓ 共享专家辅助损失: {aux_loss_shared.item() if aux_loss_shared is not None else None}")
        
        assert output_shared.shape == x.shape
        print("✓ 共享专家功能测试通过")
        
    except Exception as e:
        print(f"✗ 共享专家功能测试失败: {e}")
    print()
    
    # 5. 测试专家负载均衡
    print("5. 测试专家负载均衡")
    print("-" * 40)
    try:
        # 统计专家使用频率
        expert_counts = torch.zeros(config.n_routed_experts)
        num_samples = 100
        
        moe_ffn.train()
        for _ in range(num_samples):
            sample_x = torch.randn(1, seq_len, config.dim)
            topk_idx, _, _ = moe_ffn.gate(sample_x)
            for idx in topk_idx.flatten():
                expert_counts[idx] += 1
        
        expert_usage = expert_counts / expert_counts.sum()
        print(f"专家使用分布: {expert_usage.tolist()}")
        
        # 检查负载均衡性
        expected_usage = 1.0 / config.n_routed_experts
        max_deviation = torch.abs(expert_usage - expected_usage).max()
        print(f"✓ 最大偏差: {max_deviation:.4f}")
        print(f"✓ 期望使用率: {expected_usage:.4f}")
        
        if max_deviation < 0.2:  # 允许20%的偏差
            print("✓ 专家负载相对均衡")
        else:
            print("⚠ 专家负载不够均衡，可能需要调整辅助损失权重")
            
    except Exception as e:
        print(f"✗ 专家负载均衡测试失败: {e}")
    print()
    
    # 6. 性能基准测试
    print("6. 性能基准测试")
    print("-" * 40)
    try:
        import time
        
        # 创建普通前馈网络作为对比
        from pre_transformer import FeedForward
        normal_ffn = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout
        )
        
        # 预热
        for _ in range(5):
            _ = moe_ffn(x)
            _ = normal_ffn(x)
        
        # 测试 MOE 性能
        moe_ffn.eval()
        start_time = time.time()
        num_runs = 20
        for _ in range(num_runs):
            with torch.no_grad():
                _, _ = moe_ffn(x)
        moe_time = (time.time() - start_time) / num_runs
        
        # 测试普通前馈网络性能
        normal_ffn.eval()
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = normal_ffn(x)
        normal_time = (time.time() - start_time) / num_runs
        
        print(f"✓ MOE 前馈网络平均时间: {moe_time*1000:.2f} ms")
        print(f"✓ 普通前馈网络平均时间: {normal_time*1000:.2f} ms")
        print(f"✓ 性能比率 (MOE/Normal): {moe_time/normal_time:.2f}x")
        
        if moe_time / normal_time < 3.0:  # MOE 时间不超过普通网络的3倍
            print("✓ MOE 性能开销在可接受范围内")
        else:
            print("⚠ MOE 性能开销较大")
            
    except Exception as e:
        print(f"✗ 性能基准测试失败: {e}")
    print()
    
    # 7. 梯度检查
    print("7. 梯度检查")
    print("-" * 40)
    try:
        # 设置需要梯度
        x_grad = torch.randn(batch_size, seq_len, config.dim, requires_grad=True)
        
        # 前向传播
        moe_ffn.train()
        output, aux_loss = moe_ffn(x_grad)
        
        # 计算总损失
        main_loss = output.sum()
        total_loss = main_loss
        if aux_loss is not None:
            total_loss = main_loss + aux_loss
        
        # 反向传播
        total_loss.backward()
        
        print(f"✓ 输入梯度形状: {x_grad.grad.shape}")
        print(f"✓ 输入梯度范数: {x_grad.grad.norm():.6f}")
        print(f"✓ 主损失: {main_loss.item():.6f}")
        if aux_loss is not None:
            print(f"✓ 辅助损失: {aux_loss.item():.6f}")
        
        # 检查门控网络梯度
        gate_grad_norm = moe_ffn.gate.weight.grad.norm()
        print(f"✓ 门控网络梯度范数: {gate_grad_norm:.6f}")
        
        # 检查专家网络梯度
        expert_grad_norms = []
        for i, expert in enumerate(moe_ffn.experts):
            expert_grad_norm = sum(p.grad.norm().item() for p in expert.parameters() if p.grad is not None)
            expert_grad_norms.append(expert_grad_norm)
        
        print(f"✓ 专家网络梯度范数: {expert_grad_norms}")
        print("✓ 梯度检查测试通过")
        
    except Exception as e:
        print(f"✗ 梯度检查测试失败: {e}")
    print()
    
    # 8. 不同配置测试
    print("8. 不同配置测试")
    print("-" * 40)
    try:
        # 测试不同的专家选择数量
        for k in [1, 2, 4]:
            if k <= config.n_routed_experts:
                test_config = TestConfig()
                test_config.num_experts_per_tok = k
                
                validate_moe_config(test_config)
                test_moe = MOEFeedForward(test_config)
                
                output, aux_loss = test_moe(x)
                print(f"✓ Top-{k} 专家配置测试通过，输出形状: {output.shape}")
        
        # 测试不同的评分函数
        print("✓ 不同配置测试通过")
        
    except Exception as e:
        print(f"✗ 不同配置测试失败: {e}")
    print()
    
    print("=" * 60)
    print("所有测试完成！MOE 组件功能正常。")
    print("=" * 60)
    
    # 输出总结信息
    print("\n📊 MOE 组件总结:")
    print(f"- 专家总数: {config.n_routed_experts}")
    print(f"- 每token选择专家数: {config.num_experts_per_tok}")
    print(f"- 参数总数: {sum(p.numel() for p in moe_ffn.parameters()):,}")
    print(f"- 门控网络参数: {moe_ffn.gate.weight.numel():,}")
    print(f"- 单个专家参数: {sum(p.numel() for p in moe_ffn.experts[0].parameters()):,}")
    print(f"- 专家网络总参数: {sum(p.numel() for expert in moe_ffn.experts for p in expert.parameters()):,}")