import os
import time
import numpy as np
import torch
from torch import optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
from pre_transformer import MyPretrainConfig, get_lr, validate_config, Transformer, count_parameters

try:
    from pre_configurator import llmconfig, logger
    logger.info("已加载pre_configurator配置")
except ImportError as e:
    print("未找到pre_configurator.py，将使用默认配置")
    raise e

class SFTDataset(Dataset):
    def __init__(self, data_path_lst, max_length=512, sample_ratio=0.1):
        super().__init__()
        # 从配置获取参数
        self.max_length = max_length or llmconfig.get('model_config.max_seq_len', 1024)
        data_list = []

        for data_name in data_path_lst:
            data_path = Path(data_name)
            if not data_path.exists():
                logger.warning(f"数据文件不存在: {data_path}")
                continue

            try:
                if data_path.suffix == '.bin':
                    data = np.fromfile(data_path, dtype=np.uint16)
                else:
                    with open(data_path, 'rb') as f:
                        data = np.loadtxt(f, dtype=np.uint16)
                
                # 数据采样
                if sample_ratio < 1.0:
                    sample_size = int(len(data) * sample_ratio)
                    indices = np.random.choice(len(data), sample_size, replace=False)
                    data = data[indices]
                    logger.info(f"采样数据: {len(data)} / {len(data) / sample_ratio:.0f} ({sample_ratio*100:.1f}%)")
                
                data_list.append(data)
                logger.info(f"加载数据文件: {data_path}, 大小: {len(data)}")
            except Exception as e:
                logger.error(f"加载数据文件失败: {data_path}, 错误: {e}")
                continue
        
        if not data_list:
            raise FileNotFoundError("未找到有效的训练数据文件")
            
        data = np.concatenate(data_list)
        data = data[:self.max_length * int(len(data) / self.max_length)]
        self.data = data.reshape(-1, self.max_length)
        logger.info(f"训练数据形状: {self.data.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)

def create_config_from_llmconfig():
    """从 llmconfig 创建 MyPretrainConfig 实例"""
    return MyPretrainConfig(
        dim=llmconfig.get('model_config.dim', 1024),
        n_layers=llmconfig.get('model_config.n_layers', 16),
        n_heads=llmconfig.get('model_config.n_heads', 16),
        n_kv_heads=llmconfig.get('model_config.n_kv_heads', 8),
        vocab_size=llmconfig.get('model_config.vocab_size', 19200),
        hidden_dim=llmconfig.get('model_config.hidden_dim', None),
        multiple_of=llmconfig.get('model_config.multiple_of', 64),
        norm_eps=llmconfig.get('model_config.norm_eps', 1e-5),
        max_seq_len=llmconfig.get('model_config.max_seq_len', 512),
        dropout=llmconfig.get('model_config.dropout', 0.0),
        flash_attn=llmconfig.get('model_config.flash_attn', True),
        rope_theta=llmconfig.get('model_config.rope_theta', 10000.0),
        activation_function=llmconfig.get('model_config.activation_function', 'silu'),
        initializer_range=llmconfig.get('model_config.initializer_range', 0.02),
        tie_word_embeddings=llmconfig.get('model_config.tie_word_embeddings', False),
        torch_dtype=llmconfig.get('model_config.torch_dtype', 'float16'),
        attn_implementation=llmconfig.get('model_config.attn_implementation', 'flash_attention_2'),
    )

def optimize_for_mps():
    """针对 MPS 的优化设置"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # 设置 MPS 内存分配策略
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 禁用高水位标记
        
        # 启用 MPS 优化
        torch.backends.mps.enable_fallback = True  # 启用回退机制
        logger.info("已启用 MPS 优化设置")

def load_model(config, device_type, batch_size):
    """初始化模型并加载预训练权重"""
    model = Transformer(config)
    
    # 构建检查点路径
    checkpoint_dir = llmconfig.get('paths.checkpoint_dir', 'outputs')
    ckp_path = os.path.join(checkpoint_dir, f'pretrain_model_{config.dim}_{batch_size}.pth')
    
    if os.path.exists(ckp_path):
        logger.info(f"加载预训练模型: {ckp_path}")
        logger.info(f"模型加载成功，参数量: {count_parameters(model)}")
    else:
        logger.warning(f"预训练模型不存在: {ckp_path}，使用随机初始化")
    
    model = model.to(device_type)
    return model

def main():
    """主函数"""
    optimize_for_mps()
    
    # 训练参数
    epochs = llmconfig.get('model_config.train.epochs', 20)
    batch_size = llmconfig.get('model_config.train.batch_size', 8)
    learning_rate = llmconfig.get('model_config.train.learning_rate', 1e-5)
    accumulation_steps = llmconfig.get('model_config.train.accumulation_steps', 8)
    max_grad_norm = llmconfig.get('model_config.train.max_grad_norm', 1.0)
    num_workers = llmconfig.get('model_config.train.num_workers', 0)
    save_steps = llmconfig.get('model_config.train.save_steps', 1000)
    device_type = llmconfig.get('model_config.train.device_type', 'cuda')
    # 使用优化的设备检测
    dtype = llmconfig.get('model_config.torch_dtype', 'bfloat16')
    sample_ratio = llmconfig.get('model_config.train.sample_ratio', 0.1)
    
    # MPS 特殊处理：调整批次大小和数据类型
    if device_type == 'mps':
        # MPS 对某些数据类型支持有限，建议使用 float32
        if dtype == 'bfloat16':
            dtype = 'float32'
            logger.info("MPS 设备：将数据类型从 bfloat16 调整为 float32")
        
        # 根据内存情况调整批次大小
        if batch_size > 4:
            batch_size = 4
            logger.info(f"MPS 设备：调整批次大小为 {batch_size}")
        
        # MPS 优化：减少 num_workers
        if num_workers > 0:
            num_workers = 0
            logger.info("MPS 设备：设置 num_workers=0 以避免多进程问题")
    
    # 输出目录
    out_dir = llmconfig.get('paths.output_dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    save_dir = Path(out_dir)
    save_dir.mkdir(exist_ok=True)
    
    logger.info(f"设备: {device_type}")
    logger.info(f"数据类型: {dtype}")
    logger.info(f"批次大小: {batch_size}")
    
    # 设置随机种子
    torch.manual_seed(llmconfig.get('model_config.train.seed', 1337))
    
    # 根据设备类型设置上下文
    if device_type == "cpu":
        ctx = nullcontext()
    elif device_type == "mps":
        # MPS 使用 float32 精度
        ctx = torch.amp.autocast(device_type='cpu', dtype=torch.float32) if dtype == 'float32' else nullcontext()
    else:
        ctx = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, dtype, torch.bfloat16))
    
    # 初始化配置和模型
    lm_config = create_config_from_llmconfig()
    logger.info(f"模型配置: {lm_config.get_config_info()}")
    validate_config(lm_config)  # 验证配置参数
    max_seq_len = lm_config.max_seq_len
    # 初始化模型
    model = load_model(lm_config, device_type, batch_size)
    
    # 数据加载
    data_path_list = [f'{out_dir}/sft_data.bin']
    
    try:
        train_ds = SFTDataset(data_path_list, max_length=max_seq_len, sample_ratio=sample_ratio)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            pin_memory=(device_type == 'cuda'),  # 只在 CUDA 上启用 pin_memory
            drop_last=False,
            shuffle=False,
            num_workers=num_workers
        )
        logger.info(f"数据加载器创建成功，批次数: {len(train_loader)}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    
    # 优化器和缩放器 - MPS 不支持 GradScaler
    use_scaler = (dtype in ['float16', 'bfloat16'] and device_type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler) if use_scaler else None
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 检查是否有检查点需要恢复
    resume_epoch = 0
    resume_step = 0
    resume_checkpoint_path = llmconfig.get('model_config.train.resume_checkpoint', None)
    if resume_checkpoint_path and len(resume_checkpoint_path) > 0:
        resume_epoch, resume_step, _ = load_checkpoint(model, optimizer, resume_checkpoint_path, device_type)
        logger.info(f"从检查点 {resume_checkpoint_path} 恢复训练，epoch={resume_epoch}, step={resume_step}")

    # 训练循环
    iter_per_epoch = len(train_loader)
    total_steps = epochs * iter_per_epoch
    
    logger.info(f"开始训练: {epochs} epochs, {iter_per_epoch} steps/epoch")
    
    for epoch in range(resume_epoch, epochs):
        start_time = time.time()
        model.train()
        
        start_step = resume_step if epoch == resume_epoch else 0  # 如果是恢复的epoch，从恢复的step开始
        
        for step, (X, Y) in enumerate(train_loader):
            if step < start_step:  # 跳过已经训练过的步骤
                continue

            # MPS 优化：使用 non_blocking=False
            non_blocking = (device_type == 'cuda')
            X = X.to(device_type, non_blocking=non_blocking)
            Y = Y.to(device_type, non_blocking=non_blocking)
            
            # 动态学习率
            current_step = epoch * iter_per_epoch + step
            lr = get_lr(current_step, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 前向传播
            with ctx:
                out = model(X, Y)
                loss = out.last_loss
            
            # 反向传播 - 根据设备类型选择策略
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度更新
            if (step + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                
                # MPS 内存管理
                if device_type == 'mps':
                    torch.mps.empty_cache()
            
            # 定期保存和日志
            if step % save_steps == 0:
                elapsed_time = time.time() - start_time
                eta = elapsed_time / (step + 1) * (iter_per_epoch - step - 1) / 60
                
                logger.info(
                    f'Epoch [{epoch+1}/{epochs}] Step [{step}/{iter_per_epoch}] '
                    f'Loss: {loss.item():.4f} LR: {lr:.2e} ETA: {eta:.1f}min'
                )
                
                # 保存模型
                model.eval()
                ckp_path = save_dir / f'full_sft_epoch{epoch+1}_step{step}.pth'
                
                # 删除之前的检查点文件（保留最近的2个）
                existing_checkpoints = sorted(save_dir.glob('full_sft_epoch*.pth'))
                if len(existing_checkpoints) >= 2:  # 保留最近的2个检查点
                    for old_ckp in existing_checkpoints[:-1]:  # 删除除了最新的之外的所有检查点
                        try:
                            old_ckp.unlink()
                            logger.info(f"已删除旧检查点: {old_ckp}")
                        except Exception as e:
                            logger.warning(f"删除旧检查点失败: {old_ckp}, 错误: {e}")
                
                # 保存完整的状态字典
                state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'loss': loss.item(),
                    'config': lm_config,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
                
                try:
                    torch.save(state_dict, ckp_path)
                    logger.info(f"模型已保存到: {ckp_path}")
                except Exception as e:
                    logger.error(f"保存模型失败: {e}")
                    
                model.train()
        
        # MPS 内存清理
        if device_type == 'mps':
            torch.mps.empty_cache()
    
    # 训练完成后保存最终模型
    logger.info("训练完成！开始保存最终模型...")
    model.eval()
    
    # 按照 {config.dim}_{config.batch_size} 格式命名
    final_model_name = f'full_sft_model_{lm_config.dim}_{batch_size}.pth'
    final_ckp_path = save_dir / final_model_name
    
    final_state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epochs,
        'config': lm_config,
        'training_completed': True
    }
    
    torch.save(final_state_dict, final_ckp_path)
    logger.info(f"最终模型已保存到: {final_ckp_path}")
    logger.info("训练完成！")

if __name__ == "__main__":
    main()