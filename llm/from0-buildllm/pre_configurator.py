import logging
import os
import platform
import torch
from pathlib import Path

class ConfigAccessor:
    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):
        if isinstance(key, str) and '.' in key:
            # 复用已有的 __get_config_attribute 实现
            try:
                # 按点号分割路径
                keys = key.split('.')    
                current = self.config
                # 逐级访问属性
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return default
                
                return current
            except (KeyError, TypeError, AttributeError):
                return default

        return self.config.get(key, default)

def detect_device_type():
    """检测当前机器的设备类型和硬件配置"""
    device_info = {
        'device_type': 'cpu',
        'device_count': 1,
        'memory_gb': 8,
        'is_apple_silicon': False
    }
    
    # 检测是否为Apple Silicon Mac
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        device_info['is_apple_silicon'] = True
        if torch.backends.mps.is_available():
            device_info['device_type'] = 'mps'
            # Apple Silicon通常有统一内存架构
            import psutil
            device_info['memory_gb'] = psutil.virtual_memory().total // (1024**3)
    
    # 检测CUDA
    elif torch.cuda.is_available():
        device_info['device_type'] = 'cuda'
        device_info['device_count'] = torch.cuda.device_count()
        # 获取GPU内存信息
        if device_info['device_count'] > 0:
            device_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    
    # CPU fallback
    else:
        import psutil
        device_info['memory_gb'] = psutil.virtual_memory().total // (1024**3)
        device_info['device_count'] = psutil.cpu_count()
    
    return device_info

def optimize_config_for_device(base_config):
    """根据设备类型优化配置参数，目标6小时内训练完成"""
    device_info = detect_device_type()
    config = base_config.copy()
    device_type = device_info['device_type']
    memory_gb = device_info['memory_gb']
    device_count = device_info['device_count']
    
    # 基础优化参数 - 针对6小时训练目标
    if device_type == 'mps':  # Apple Silicon Mac
        # MPS优化配置
        config['model_config']['dim'] = 256 if memory_gb >= 16 else 128
        config['model_config']['n_layers'] = 6 if memory_gb >= 16 else 4
        config['model_config']['n_heads'] = 8 if memory_gb >= 16 else 4
        config['model_config']['n_kv_heads'] = 8 if memory_gb >= 16 else 4
        config['model_config']['intermediate_size'] = config['model_config']['dim'] * 4
        config['model_config']['hidden_dim'] = config['model_config']['intermediate_size']
        
        # 训练参数优化
        config['model_config']['train']['device_type'] = 'mps'
        config['model_config']['train']['batch_size'] = 8 if memory_gb >= 16 else 4
        config['model_config']['train']['accumulation_steps'] = 2 if memory_gb >= 16 else 4
        config['model_config']['train']['learning_rate'] = 2e-4  # 提高学习率加速训练
        config['model_config']['train']['num_workers'] = 4 if memory_gb >= 16 else 2
        config['model_config']['train']['sample_ratio'] = 0.3 if memory_gb >= 16 else 0.2  # 增加数据采样
        
        # 禁用DeepSpeed（MPS不支持）
        config['distributed_config']['use_deepspeed'] = False
        config['model_config']['torch_dtype'] = 'float16'
        config['model_config']['dtype'] = 'float16'
        
    elif device_type == 'cuda':  # NVIDIA GPU
        # CUDA优化配置
        if memory_gb >= 24:  # 高端GPU
            config['model_config']['dim'] = 512
            config['model_config']['n_layers'] = 8
            config['model_config']['n_heads'] = 16
            config['model_config']['n_kv_heads'] = 16
            config['model_config']['max_seq_len'] = 1024
            config['model_config']['train']['batch_size'] = 16
            config['model_config']['train']['accumulation_steps'] = 1
        elif memory_gb >= 12:  # 中端GPU
            config['model_config']['dim'] = 384
            config['model_config']['n_layers'] = 6
            config['model_config']['n_heads'] = 12
            config['model_config']['n_kv_heads'] = 12
            config['model_config']['max_seq_len'] = 768
            config['model_config']['train']['batch_size'] = 12
            config['model_config']['train']['accumulation_steps'] = 2
        else:  # 低端GPU
            config['model_config']['dim'] = 256
            config['model_config']['n_layers'] = 4
            config['model_config']['n_heads'] = 8
            config['model_config']['n_kv_heads'] = 8
            config['model_config']['max_seq_len'] = 512
            config['model_config']['train']['batch_size'] = 8
            config['model_config']['train']['accumulation_steps'] = 4
        
        config['model_config']['intermediate_size'] = config['model_config']['dim'] * 4
        config['model_config']['hidden_dim'] = config['model_config']['intermediate_size']
        
        # CUDA训练参数
        config['model_config']['train']['device_type'] = 'cuda'
        config['model_config']['train']['learning_rate'] = 3e-4  # CUDA可以用更高学习率
        config['model_config']['train']['num_workers'] = min(8, device_count * 2)
        config['model_config']['train']['sample_ratio'] = 0.4 if memory_gb >= 16 else 0.3
        
        # 启用DeepSpeed和混合精度
        config['distributed_config']['use_deepspeed'] = device_count > 1
        config['model_config']['torch_dtype'] = 'bfloat16'
        config['model_config']['dtype'] = 'bfloat16'
        config['model_config']['flash_attn'] = True
        
        # 多GPU环境配置
        if device_count > 1:
            config['environment']['world_size'] = device_count
            config['environment']['cuda_visible_devices'] = ','.join(map(str, range(device_count)))
        
    else: # CPU
        # CPU优化配置 - 最小模型
        config['model_config']['dim'] = 128
        config['model_config']['n_layers'] = 3
        config['model_config']['n_heads'] = 4
        config['model_config']['n_kv_heads'] = 4
        config['model_config']['max_seq_len'] = 256
        config['model_config']['intermediate_size'] = 512
        config['model_config']['hidden_dim'] = 512
        
        # CPU训练参数
        config['model_config']['train']['device_type'] = 'cpu'
        config['model_config']['train']['batch_size'] = 2
        config['model_config']['train']['accumulation_steps'] = 8
        config['model_config']['train']['learning_rate'] = 1e-4
        config['model_config']['train']['num_workers'] = min(4, device_count)
        config['model_config']['train']['sample_ratio'] = 0.1  # CPU训练用最少数据
        
        # 禁用高级特性
        config['distributed_config']['use_deepspeed'] = False
        config['model_config']['torch_dtype'] = 'float32'
        config['model_config']['dtype'] = 'float32'
        config['model_config']['flash_attn'] = False
    
    return config

# 配置验证函数
def validate_config(cfg):
    """验证配置的有效性"""
    errors = []
    
    # 检查必要的路径
    required_paths = ['base_dir', 'tokenizer_dir', 'model_dir']
    for path_key in required_paths:
        if not cfg.get('paths', {}).get(path_key):
            errors.append(f"Missing required path: {path_key}")
    
    # 检查模型配置
    if cfg.get('model_config.vocab_size', 0) != cfg.get('tokenizer_config.vocab_size', 0):
        errors.append("Model vocab_size and tokenizer vocab_size must match")
    
    # 检查训练配置
    if cfg.get('model_config.train.batch_size', 0) <= 0:
        errors.append("Batch size must be positive")
    
    if cfg.get('model_config.train.learning_rate', 0) <= 0:
        errors.append("Learning rate must be positive")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True

# 配置日志（优化版）
def setup_logging(log_level=logging.INFO, log_dir="./logs"):
    """设置日志配置"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'training.log'), 
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 错误日志处理器
    error_handler = logging.FileHandler(
        os.path.join(log_dir, 'error.log'), 
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

default_llmconfig = {
    # 基本信息配置
    "name": "myllm",  # 模型名称
    "version": "1.0.0",  # 版本号
    "description": "从零构建的大语言模型训练配置",  # 项目描述
    
    # 路径配置 - 定义各种文件和目录的存储位置
    "paths": {
        "base_dir": "./datasets",  # 数据集根目录
        "tokenizer_dir": "./my_tokenizer",  # 分词器保存目录
        "model_dir": "./models",  # 模型文件保存目录
        "checkpoint_dir": "./outputs",  # 训练检查点保存目录
        "log_dir": "./logs",  # 日志文件保存目录
        "cache_dir": "./cache",  # 缓存文件保存目录
        "output_dir": "./outputs"  # 输出结果保存目录
    },
    
    # 数据集配置 - 定义训练、验证和测试数据的处理方式
    "dataset": {
        "tokenizer": {
            "content_field": ['completion'],  # 训练数据文件字段列表
            "files": ["datasets/tokenizer_wikipedia-cn-20230720-filtered.json"],  # 训练数据文件函数表
        },
        "preprocessing": {  # 数据预处理配置
            "content_field": ['content', "text", "completion"],  # 训练数据文件字段列表
            "files": [
                "datasets/pretrain_data_preprocessing.json", 
                "datasets/pretrain_data_preprocessing_hq.jsonl", 
                "datasets/pretrain_web_text_zh_train.jsonl"
            ],  # 训练数据文件列表
            "remove_duplicates": True,  # 是否移除重复数据
            "min_length": 10,  # 最小文本长度阈值
            "max_length": 2048,  # 最大文本长度阈值
            "filter_languages": ["zh", "en"],  # 保留的语言类型
            "quality_threshold": 0.7  # 数据质量阈值
        },
        "sft_preprocessing": {  # sft 数据预处理配置
            "content_field": ['instruction', 'input', 'output'],  # sft 数据文件字段列表
            "files": ["./datasets/sft_data_preprocessing.jsonl"],  # sft 数据文件列表
            "remove_duplicates": True,  # 是否移除重复数据
            "min_length": 10,  # 最小文本长度阈值
            "max_length": 2048,  # 最大文本长度阈值
            "filter_languages": ["zh", "en"],  # 保留的语言类型
            "quality_threshold": 0.7  # 数据质量阈值
        },
        "valid_files": ["valid.txt", "valid.jsonl"],  # 验证数据文件列表
        "test_files": ["test.txt", "test.jsonl"],  # 测试数据文件列表
        "max_seq_length": 2048  # 最大序列长度
    },
    
    # Tokenizer配置（优化版） - 分词器的详细配置
    "tokenizer_config": {
        "vocab_size": 3200,  # 词汇表大小，增加以提高表达能力
        "model_type": "BPE",  # 分词算法类型，使用字节对编码
        "add_bos_token": True,  # 是否添加句子开始标记
        "add_eos_token": True,  # 是否添加句子结束标记
        "add_prefix_space": True,  # 是否在词前添加空格
        "added_tokens_decoder": {  # 特殊标记的解码配置
            "0": {  # 未知标记配置
                "content": "<unk>",  # 标记内容
                "lstrip": False,  # 是否去除左侧空格
                "normalized": False,  # 是否标准化
                "rstrip": False,  # 是否去除右侧空格
                "single_word": False,  # 是否为单词
                "special": True  # 是否为特殊标记
            },
            "1": {  # 句子开始标记配置
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {  # 句子结束标记配置
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "3": {  # 填充标记配置
                "content": "<pad>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": ["<mask>", "<sep>", "<cls>"],  # 额外的特殊标记
        "bos_token": "<s>",  # 句子开始标记
        "eos_token": "</s>",  # 句子结束标记
        "pad_token": "<pad>",  # 填充标记
        "unk_token": "<unk>",  # 未知标记
        "mask_token": "<mask>",  # 掩码标记
        "sep_token": "<sep>",  # 分隔标记
        "cls_token": "<cls>",  # 分类标记
        "clean_up_tokenization_spaces": False,  # 是否清理分词空格
        "legacy": False,  # 是否使用旧版本特性
        "model_max_length": 2048,  # 模型最大输入长度
        "sp_model_kwargs": {},  # SentencePiece模型参数
        "spaces_between_special_tokens": False,  # 特殊标记间是否添加空格
        "tokenizer_class": "PreTrainedTokenizerFast",  # 分词器类名
        "use_default_system_prompt": False,  # 是否使用默认系统提示
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"  # 对话模板
    },
    
    # 模型配置 - 包含模型架构和训练参数
    "model_config": {
        "vocab_size": 3200,  # 词汇表大小
        "dim": 128,  # 隐藏层维度 
        "intermediate_size": 11008,  # 前馈网络中间层维度
        "hidden_dim": 11008,  # 前馈网络中间层维度
        "n_layers": 8,  # Transformer层数
        "n_heads": 8,  # 多头注意力头数
        "n_kv_heads": 8,  # 键值对头数
        "max_seq_len": 256,  # 最大序列长度
        "norm_eps": 1e-6,  # RMS归一化的epsilon值
        "rope_theta": 10000.0,  # 旋转位置编码的theta参数
        "dropout": 0.0,  # dropout率
        "flash_attn": True,  # 是否使用flash attention
        "activation_function": "silu",  # 激活函数类型
        "initializer_range": 0.02,  # 参数初始化范围
        "use_cache": True,  # 是否启用键值缓存
        "tie_word_embeddings": False,  # 是否共享输入输出词嵌入权重
        "torch_dtype": "bfloat16",  # PyTorch数据类型
        "dtype": "bfloat16",  # PyTorch数据类型
        "attn_implementation": "eager",  # 注意力机制实现方式
        "multiple_of": 64,  # 隐藏层维度的倍数要求
        
        # train 参数
        "train": {
            "weight_decay": 0.01,  # 权重衰减系数
            "save_steps": 500,  # 模型保存间隔步数
            "seed": 42,  # 随机种子
            "epochs": 1,  # 训练轮数
            "batch_size": 16,  # 每个设备的训练批次大小
            "learning_rate": 1e-5,  # 学习率
            "accumulation_steps": 4,  # 梯度累积步数
            "max_grad_norm": 1.0,  # 梯度裁剪的最大范数
            "num_workers": 2,  # 数据加载器工作进程数
            "device_type": "mps",  # 训练设备,支持：cpu,cuda,mps
            "resume_checkpoint": "",  # 是否从检查点恢复训练，为空或者不设置从新开始训练
            "sample_ratio": 1  # 数据采样比例
        }
    },
    
    # 优化器配置 - 定义优化算法的具体参数
    "optimizer_config": {
        "optimizer_type": "adamw_torch",  # 优化器类型
        "lr_scheduler_kwargs": {  # 学习率调度器参数
            "num_cycles": 0.5  # 余弦调度的周期数
        },
        "optim_args": {  # 优化器参数
            "weight_decay": 0.01,  # 权重衰减
            "betas": [0.9, 0.999],  # Adam的beta参数
            "eps": 1e-8  # 数值稳定性参数
        }
    },
    
    # 分布式训练配置 - DeepSpeed分布式训练的详细设置
    "distributed_config": {
        "use_deepspeed": True,  # 是否使用DeepSpeed
        "deepspeed_config": {  # DeepSpeed配置
            "zero_optimization": {  # ZeRO优化配置
                "stage": 2,  # ZeRO优化阶段
                "offload_optimizer": {  # 优化器状态卸载配置
                    "device": "cpu",  # 卸载到CPU
                    "pin_memory": False  # 是否固定内存
                },
                "offload_param": {  # 参数卸载配置
                    "device": "cpu",  # 卸载到CPU
                    "pin_memory": False  # 是否固定内存
                },
                "overlap_comm": True,  # 是否重叠通信
                "contiguous_gradients": True,  # 是否使用连续梯度
                "sub_group_size": 1e9,  # 子组大小
                "reduce_bucket_size": "auto",  # 归约桶大小
                "stage3_prefetch_bucket_size": "auto",  # 阶段3预取桶大小
                "stage3_param_persistence_threshold": "auto",  # 阶段3参数持久化阈值
                "stage3_max_live_parameters": 1e9,  # 阶段3最大活跃参数数
                "stage3_max_reuse_distance": 1e9,  # 阶段3最大重用距离
                "gather_16bit_weights_on_model_save": True  # 保存模型时是否收集16位权重
            },
            "fp16": {  # 16位浮点数配置
                "enabled": True,  # 是否启用
                "loss_scale": 0,  # 损失缩放
                "loss_scale_window": 1000,  # 损失缩放窗口
                "initial_scale_power": 16,  # 初始缩放幂次
                "hysteresis": 2,  # 滞后参数
                "min_loss_scale": 1  # 最小损失缩放
            },
            "train_batch_size": "auto",  # 训练批次大小（自动）
            "train_micro_batch_size_per_gpu": "auto",  # 每GPU微批次大小（自动）
            "gradient_accumulation_steps": "auto",  # 梯度累积步数（自动）
            "gradient_clipping": "auto",  # 梯度裁剪（自动）
            "steps_per_print": 10,  # 打印间隔步数
            "wall_clock_breakdown": False  # 是否显示时间分解
        }
    },
    
    # 评估配置 - 模型性能评估的相关设置
    "evaluation_config": {
        "metrics": ["perplexity", "bleu", "rouge"],  # 评估指标列表
        "eval_datasets": ["validation", "test"],  # 评估数据集
        "generation_config": {  # 文本生成配置
            "max_new_tokens": 512,  # 最大生成token数
            "temperature": 0.7,  # 生成温度
            "top_p": 0.9,  # 核采样概率阈值
            "top_k": 50,  # top-k采样参数
            "do_sample": True,  # 是否启用采样
            "repetition_penalty": 1.1,  # 重复惩罚系数
            "length_penalty": 1.0,  # 长度惩罚系数
            "early_stopping": True  # 是否启用早停
        }
    },
    
    # 推理配置 - 模型推理时的资源和参数设置
    "inference_config": {
        "max_memory": {  # 最大内存限制
            0: "20GB",  # GPU 0的内存限制
            1: "20GB"   # GPU 1的内存限制
        },
        "device_map": "auto",  # 设备映射策略
        "torch_dtype": "float16",  # 推理时的数据类型
        "trust_remote_code": True,  # 是否信任远程代码
        "use_flash_attention_2": True  # 是否使用Flash Attention 2
    },
    
    # 特殊标记 - 模型使用的特殊token列表
    "special_tokens": ["<unk>", "<s>", "</s>", "<pad>", "<mask>", "<sep>", "<cls>"],
    
    # 环境配置 - 训练环境的系统级设置
    "environment": {
        "cuda_visible_devices": "0,1",  # 可见的CUDA设备
        "world_size": 2,  # 分布式训练的总进程数
        "local_rank": -1,  # 本地进程排名
        "master_port": "29500",  # 主节点端口
        "nccl_debug": "INFO"  # NCCL调试级别
    }
}

# 根据机器自动优化config
is_auto_optimize_llmconfig = True
llmconfig = ConfigAccessor(default_llmconfig)
if is_auto_optimize_llmconfig:
    llmconfig = ConfigAccessor(optimize_config_for_device(default_llmconfig))

logger = setup_logging(log_dir=llmconfig.get('paths.log_dir', './logs'))

device_info = detect_device_type()
device_type = device_info['device_type']
memory_gb = device_info['memory_gb']
device_count = device_info['device_count']

logger.info(f"检测到设备类型: {device_type}, 内存: {memory_gb}GB, 设备数量: {device_count}")

if __name__ == "__main__":
    # 使用优化后的配置
    cfg = llmconfig
    
    # 验证配置
    validate_config(cfg)
    logger.info(f"Configuration loaded successfully for model: {cfg.get('name', 'unknown')}")
    logger.info(f"Model architecture: dim={cfg.get('model_config.dim', 0)}, layers={cfg.get('model_config.n_layers', 0)}")
    logger.info(f"Training batch size: {cfg.get('model_config.train.batch_size', 0)} per device")
    logger.info(f"Max sequence length: {cfg.get('model_config.max_seq_len', 0)}")
    logger.info(f"Sample ratio: {cfg.get('model_config.train.sample_ratio', 0)}")
