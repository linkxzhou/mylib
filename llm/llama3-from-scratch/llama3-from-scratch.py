import torch
import torch.nn.functional as F
import json
import tiktoken
from pathlib import Path
from tiktoken.load import load_tiktoken_bpe
from typing import Optional
import math

class LlamaConfig:
    """Llama模型配置类"""
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.dim = config["dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.vocab_size = config["vocab_size"]
        self.multiple_of = config["multiple_of"]
        self.ffn_dim_multiplier = config["ffn_dim_multiplier"]
        self.norm_eps = config["norm_eps"]
        self.rope_theta = torch.tensor(config["rope_theta"])
        self.head_dim = self.dim // self.n_heads

class LlamaTokenizer:
    """Llama分词器类"""
    def __init__(self, tokenizer_path: str):
        self.special_tokens = [
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>", "<|reserved_special_token_3|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|reserved_special_token_4|>", "<|eot_id|>"
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
        
        mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
        self.tokenizer = tiktoken.Encoding(
            name=Path(tokenizer_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(self.special_tokens)},
        )
    
    def encode(self, text: str) -> list:
        return [128000] + self.tokenizer.encode(text)
    
    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)

class RoPECache:
    """RoPE位置编码缓存类"""
    def __init__(self, head_dim: int, rope_theta: torch.Tensor, max_seq_len: int = 2048):
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self._cache = {}
    
    def get_freqs_cis(self, seq_len: int) -> torch.Tensor:
        if seq_len not in self._cache:
            zero_to_one_split = torch.tensor(range(self.head_dim // 2)) / (self.head_dim // 2)
            freqs = 1.0 / (self.rope_theta ** zero_to_one_split)
            freqs_for_each_token = torch.outer(torch.arange(seq_len), freqs)
            self._cache[seq_len] = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
        return self._cache[seq_len]

def rms_norm(tensor: torch.Tensor, norm_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS归一化函数"""
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + eps)) * norm_weights

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """应用RoPE位置编码"""
    x_pairs = x.float().view(x.shape[0], -1, 2)
    x_complex = torch.view_as_complex(x_pairs)
    x_rotated = x_complex * freqs_cis[:x.shape[0]]
    return torch.view_as_real(x_rotated).view(x.shape)

def compute_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """计算注意力机制"""
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    
    if mask is not None:
        scores = scores + mask
    
    attn_weights = F.softmax(scores, dim=-1).to(torch.bfloat16)
    return torch.matmul(attn_weights, v)

class LlamaAttention:
    """Llama注意力层类"""
    def __init__(self, config: LlamaConfig, layer_idx: int, model_weights: dict):
        self.config = config
        self.layer_idx = layer_idx
        
        # 加载权重
        self.wq = model_weights[f"layers.{layer_idx}.attention.wq.weight"]
        self.wk = model_weights[f"layers.{layer_idx}.attention.wk.weight"]
        self.wv = model_weights[f"layers.{layer_idx}.attention.wv.weight"]
        self.wo = model_weights[f"layers.{layer_idx}.attention.wo.weight"]
        
        # 重塑权重
        self.wq = self.wq.view(config.n_heads, config.head_dim, config.dim)
        self.wk = self.wk.view(config.n_kv_heads, config.head_dim, config.dim)
        self.wv = self.wv.view(config.n_kv_heads, config.head_dim, config.dim)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        seq_len = x.shape[0]
        attention_outputs = []
        
        for head in range(self.config.n_heads):
            # 计算Q, K, V
            q = torch.matmul(x, self.wq[head].T)
            k = torch.matmul(x, self.wk[head // 4].T)  # KV权重共享
            v = torch.matmul(x, self.wv[head // 4].T)
            
            # 应用RoPE
            q_rotated = apply_rope(q, freqs_cis)
            k_rotated = apply_rope(k, freqs_cis)
            
            # 计算注意力
            attn_output = compute_attention(q_rotated, k_rotated, v, mask)
            attention_outputs.append(attn_output)
        
        # 拼接多头输出
        multi_head_output = torch.cat(attention_outputs, dim=-1)
        return torch.matmul(multi_head_output, self.wo.T)

class LlamaFeedForward:
    """Llama前馈网络类"""
    def __init__(self, config: LlamaConfig, layer_idx: int, model_weights: dict):
        self.w1 = model_weights[f"layers.{layer_idx}.feed_forward.w1.weight"]
        self.w2 = model_weights[f"layers.{layer_idx}.feed_forward.w2.weight"]
        self.w3 = model_weights[f"layers.{layer_idx}.feed_forward.w3.weight"]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        gate = F.silu(torch.matmul(x, self.w1.T))
        up = torch.matmul(x, self.w3.T)
        return torch.matmul(gate * up, self.w2.T)

class LlamaLayer:
    """Llama Transformer层类"""
    def __init__(self, config: LlamaConfig, layer_idx: int, model_weights: dict):
        self.attention = LlamaAttention(config, layer_idx, model_weights)
        self.feed_forward = LlamaFeedForward(config, layer_idx, model_weights)
        self.attention_norm = model_weights[f"layers.{layer_idx}.attention_norm.weight"]
        self.ffn_norm = model_weights[f"layers.{layer_idx}.ffn_norm.weight"]
        self.config = config
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 注意力机制
        h = x + self.attention.forward(
            rms_norm(x, self.attention_norm, self.config.norm_eps), 
            freqs_cis, 
            mask
        )
        
        # 前馈网络
        out = h + self.feed_forward.forward(
            rms_norm(h, self.ffn_norm, self.config.norm_eps)
        )
        
        return out

class LlamaModel:
    """Llama模型主类"""
    def __init__(self, model_path: str, config_path: str, tokenizer_path: str):
        print("正在加载模型...")
        self.config = LlamaConfig(config_path)
        self.tokenizer = LlamaTokenizer(tokenizer_path)
        self.model_weights = torch.load(model_path)
        
        # 初始化embedding层
        self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.dim)
        self.embedding.weight.data.copy_(self.model_weights["tok_embeddings.weight"])
        
        # 初始化RoPE缓存
        self.rope_cache = RoPECache(self.config.head_dim, self.config.rope_theta)
        
        # 初始化所有层
        self.layers = []
        for i in range(self.config.n_layers):
            self.layers.append(LlamaLayer(self.config, i, self.model_weights))
        
        # 输出层权重
        self.norm = self.model_weights["norm.weight"]
        self.output = self.model_weights["output.weight"]
        
        print(f"模型加载完成: {self.config.n_layers}层, {self.config.n_heads}头")
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果掩码"""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)
    
    @torch.no_grad()
    def forward(self, prompt: str) -> str:
        """模型前向传播"""
        print(f"\n处理提示: {prompt}")
        
        # 分词
        tokens = self.tokenizer.encode(prompt)
        tokens_tensor = torch.tensor(tokens)
        seq_len = len(tokens)
        
        print(f"Token数量: {seq_len}")
        
        # 获取embedding
        x = self.embedding(tokens_tensor).to(torch.bfloat16)
        
        # 获取位置编码和掩码
        freqs_cis = self.rope_cache.get_freqs_cis(seq_len)
        mask = self.create_causal_mask(seq_len, tokens_tensor.device)
        
        # 通过所有层
        for i, layer in enumerate(self.layers):
            if i % 8 == 0:  # 每8层打印一次进度
                print(f"处理第 {i+1}/{self.config.n_layers} 层")
            x = layer.forward(x, freqs_cis, mask)
        
        # 最终归一化和输出
        x = rms_norm(x, self.norm, self.config.norm_eps)
        logits = torch.matmul(x[-1], self.output.T)
        
        # 获取下一个token
        next_token = torch.argmax(logits, dim=-1)
        result = self.tokenizer.decode([next_token.item()])
        
        print(f"预测的下一个token: {result}")
        return result
    
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """生成文本"""
        print(f"\n开始生成文本，最大token数: {max_tokens}")
        generated_text = prompt
        
        for i in range(max_tokens):
            next_token = self.forward(generated_text)
            if next_token.strip() == "":
                break
            generated_text += next_token
            print(f"第{i+1}步: {next_token}")
        
        return generated_text

def main():
    """主函数"""
    print("=" * 50)
    print("Llama3 从零实现 - 优化版本")
    print("=" * 50)
    
    try:
        # 初始化模型
        model = LlamaModel(
            model_path="Meta-Llama-3-8B/consolidated.00.pth",
            config_path="Meta-Llama-3-8B/params.json",
            tokenizer_path="Meta-Llama-3-8B/tokenizer.model"
        )
        
        # 测试单次推理
        prompt = "the answer to the ultimate question of life, the universe, and everything is "
        result = model.forward(prompt)
        print(f"\n单次推理结果: {prompt}{result}")
        
        # 测试文本生成（可选，注释掉以节省时间）
        # generated = model.generate("Hello, I am", max_tokens=10)
        # print(f"\n生成的文本: {generated}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保模型文件路径正确")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()