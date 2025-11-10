import numpy as np
import torch

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

def precompute_pos_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算相对位置编码的复数形式，用于旋转位置编码（RoPE）。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # 计算频率
    t = torch.arange(seq_len, device=freqs.device)  # 创建时间步长
    freqs = torch.outer(t, freqs).float()  # 计算频率的外积
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成复数形式的频率
    return pos_cis # 返回预计算的复数位置编码

def apply_rotary_emb(xq, xk, pos_cis):
    """应用旋转位置编码到查询和键。"""
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

# 忽略：测试功能
if __name__ == "__main__":
    P = getPositionEncoding(seq_len=4, d=4, n=100)
    print(P)
    
    pos_cis = precompute_pos_cis(16, 8)
    print(pos_cis)
