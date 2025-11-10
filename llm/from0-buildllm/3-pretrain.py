import time
import numpy as np
import torch
import os
from torch import optim
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
from pathlib import Path
from pre_transformer import (
    MyPretrainConfig, load_checkpoint, init_model, estimate_memory_usage
)
from pre_pretrainconfig import (
    validate_config
)

try:
    from pre_configurator import llmconfig, logger
    logger.info("已加载pre_configurator配置")
except ImportError:
    print("未找到pre_configurator.py，将使用默认配置")
    raise e

try:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    logger.warning("警告: 未安装accelerate库，将使用标准训练模式")
    ACCELERATE_AVAILABLE = False

class PretrainDataset(Dataset):
    """优化的预训练数据集，支持更高效的数据加载"""
    def __init__(self, data_path_list, max_length=512, sample_ratio=0.1, prefetch_factor=2):
        super().__init__()
        self.max_length = max_length
        self.prefetch_factor = prefetch_factor
        data_list = []
        
        for data_path in data_path_list:
            data_path = Path(data_path)
            if not data_path.exists():
                logger.warning(f"数据文件不存在: {data_path}")
                continue
                
            try:
                if data_path.suffix == '.bin':
                    data = np.fromfile(data_path, dtype=np.uint16)
                else:
                    with open(data_path, 'rb') as f:
                        data = np.loadtxt(f, dtype=np.uint16)
                
                # 数据采样优化
                if sample_ratio < 1.0:
                    sample_size = int(len(data) * sample_ratio)
                    # 使用更高效的随机采样
                    np.random.seed(42)  # 确保可重现性
                    indices = np.random.choice(len(data), sample_size, replace=False)
                    data = data[indices]
                    logger.info(f"采样数据: {len(data)} / {len(data) / sample_ratio:.0f} ({sample_ratio*100:.1f}%)")
                
                data_list.append(data)
                logger.info(f"加载数据文件: {data_path}, 大小: {len(data)}")
            except Exception as e:
                logger.error(f"加载数据文件失败: {data_path}, 错误: {e}")
                continue
        
        if not data_list:
            raise ValueError("没有成功加载任何数据文件")
            
        data = np.concatenate(data_list)
        # 确保数据长度是max_length的整数倍
        data = data[:max_length * (len(data) // max_length)]
        self.data = data.reshape(-1, max_length)
        logger.info(f"训练数据形状: {self.data.shape}")
        logger.info(f"数据加载完成，共 {len(self.data)} 个样本")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = torch.from_numpy(sample[:-1].astype(np.int64))
        Y = torch.from_numpy(sample[1:].astype(np.int64))
        return X, Y

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

def setup_accelerator(mixed_precision='fp16', gradient_accumulation_steps=1):
    """设置Accelerate加速器"""
    if not ACCELERATE_AVAILABLE:
        return None
    
    # Mac设备兼容性检查
    import platform
    if platform.system() == 'Darwin':  # Mac系统
        logger.warning("Mac设备不支持bf16，自动切换到no")
        mixed_precision = 'no'
    
    # 配置DDP参数
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    try:
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
            log_with="tensorboard",  # 可选：添加tensorboard日志
            project_dir="./logs"
        )
        logger.info(f"Accelerator初始化成功，混合精度: {mixed_precision}")
    except ValueError as e:
        if "fp16 mixed precision" in str(e) or "bf16 mixed precision" in str(e):
            logger.warning(f"混合精度不支持: {e}，回退到无混合精度")
            mixed_precision = 'no'
            accelerator = Accelerator(
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
                kwargs_handlers=[ddp_kwargs],
                log_with="tensorboard",
                project_dir="./logs"
            )
        else:
            raise e
    
    return accelerator

def auto_optimize_for_device(device_type):
    """针对不同设备的优化设置"""
    if device_type == 'cuda':
        # CUDA优化
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("已启用CUDA优化设置")
        
    elif device_type == 'mps':
        # MPS优化
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.enable_fallback = True
        logger.info("已启用MPS优化设置")

def create_dataloader(dataset, batch_size, device_type, num_workers=None):
    """创建优化的数据加载器"""
    # 自动确定num_workers
    if num_workers is None:
        if device_type == 'mps':
            num_workers = 0  # MPS避免多进程问题
        elif device_type == 'cuda':
            num_workers = min(4, torch.cuda.device_count() * 2)
        else:
            num_workers = min(4, torch.get_num_threads())
    
    # 数据加载器配置
    dataloader_config = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': (device_type == 'cuda'),
        'drop_last': True,  # 确保批次大小一致
        'persistent_workers': (num_workers > 0),  # 保持worker进程
    }
    
    # 如果使用CUDA，添加预取
    if device_type == 'cuda' and num_workers > 0:
        dataloader_config['prefetch_factor'] = 2
    
    dataloader = DataLoader(dataset, **dataloader_config)
    logger.info(f"数据加载器配置: batch_size={batch_size}, num_workers={num_workers}, pin_memory={dataloader_config['pin_memory']}")
    
    return dataloader

def main():
    """主函数 - 修复版本"""
    # 获取配置参数
    out_dir = llmconfig.get('paths.output_dir', 'outputs')
    dtype = llmconfig.get('model_config.dtype', 'bfloat16')
    epochs = llmconfig.get('model_config.train.epochs', 20)
    batch_size = llmconfig.get('model_config.train.batch_size', 8)
    learning_rate = llmconfig.get('model_config.train.learning_rate', 1e-6)
    device_type = llmconfig.get('model_config.train.device_type', 'cpu')
    
    accumulation_steps = llmconfig.get('model_config.train.accumulation_steps', 8)
    num_workers = llmconfig.get('model_config.train.num_workers', None)
    save_interval = llmconfig.get('model_config.train.save_interval', 1000)
    max_grad_norm = llmconfig.get('model_config.train.max_grad_norm', 1.0)
    sample_ratio = llmconfig.get('model_config.train.sample_ratio', 0.1)
    use_compile = llmconfig.get('model_config.train.use_compile', False)
    
    # 验证和修复数据类型
    valid_dtypes = ['float32', 'float16', 'bfloat16']
    if dtype not in valid_dtypes:
        logger.warning(f"无效的数据类型 {dtype}，使用默认值 bfloat16")
        dtype = 'bfloat16'
    
    # 设备优化
    auto_optimize_for_device(device_type)

    # 修复混合精度设置
    if dtype == 'float16':
        mixed_precision = 'fp16'
    elif dtype == 'bfloat16':
        mixed_precision = 'bf16'
    else:
        mixed_precision = 'no'
    
    # CPU不支持混合精度
    if mixed_precision != 'no' and device_type == 'cpu':
        logger.warning("CPU不支持混合精度，切换到fp32")
        mixed_precision = 'no'
        dtype = 'float32'
    
    accelerator = setup_accelerator(mixed_precision, accumulation_steps)
    
    # 初始化配置和模型
    lm_config = create_config_from_llmconfig()
    logger.info(f"模型配置: {lm_config.get_config_info()}")
    validate_config(lm_config)
    max_seq_len = lm_config.max_seq_len
    save_dir = Path(out_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 预估内存使用量
    memory_estimate = estimate_memory_usage(lm_config, batch_size, max_seq_len, dtype)
    logger.info(memory_estimate)
    
    # 设置随机种子
    if accelerator:
        set_seed(1337)
    else:
        torch.manual_seed(1337)
    
    logger.info(f"设备: {device_type}")
    logger.info(f"训练配置: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, dtype={dtype}")
    logger.info(f"Accelerate状态: {'启用' if accelerator else '未启用'}")
    
    # 初始化模型
    if accelerator:
        device = accelerator.device
    else:
        device = device_type
    
    model = init_model(lm_config, device)
    
    # torch.compile优化（PyTorch 2.0+）
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("已启用torch.compile优化")
        except Exception as e:
            logger.warning(f"torch.compile优化失败: {e}")
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 初始化数据加载
    data_path_list = [f'{out_dir}/pretrain_data.bin']
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len, sample_ratio=sample_ratio)
    
    train_loader = create_dataloader(
        train_ds, batch_size, device_type, num_workers
    )
    
    # 初始化优化器
    # 使用更优的优化器参数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),  # 更适合大模型的beta值
        eps=1e-8
    )
    
    # 学习率调度器
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10%的步数用于warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='cos'
    )
    
    # 使用Accelerate准备训练组件
    if accelerator:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
        logger.info("已使用Accelerate准备训练组件")
    else:
        # 传统混合精度设置
        if device_type == "cpu":
            ctx = nullcontext()
        elif device_type == "mps":
            ctx = torch.amp.autocast(device_type='cpu', dtype=torch.float32) if dtype == 'float32' else nullcontext()
        else:
            ctx = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, dtype, torch.bfloat16))
        
        use_scaler = (dtype in ['float16', 'bfloat16'] and device_type == 'cuda')
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if use_scaler else None
    
    # 检查是否有检查点需要恢复
    resume_epoch = 0
    resume_step = 0
    resume_checkpoint_path = llmconfig.get('model_config.train.resume_checkpoint', None)
    if resume_checkpoint_path and len(resume_checkpoint_path) > 0:
        if accelerator:
            # Accelerate方式加载检查点
            accelerator.load_state(resume_checkpoint_path)
            logger.info(f"使用Accelerate从检查点 {resume_checkpoint_path} 恢复训练")
            
            # 读取额外的训练状态信息
            training_state_path = os.path.join(resume_checkpoint_path, 'training_state.pt')
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location='cpu')
                resume_epoch = training_state.get('epoch', 0)
                resume_step = training_state.get('step', 0)
                logger.info(f"从训练状态文件恢复: epoch={resume_epoch}, step={resume_step}")
            else:
                logger.warning("未找到训练状态文件，将从epoch=0, step=0开始")
        else:
            resume_epoch, resume_step, _ = load_checkpoint(model, optimizer, resume_checkpoint_path, device_type)
            logger.info(f"从检查点 {resume_checkpoint_path} 恢复训练，epoch={resume_epoch}, step={resume_step}")
    
    logger.info(f"开始训练，共 {epochs} 个epoch，每个epoch {len(train_loader)} 个batch")
    if resume_epoch > 0:
        logger.info(f"从第 {resume_epoch+1} 个epoch，第 {resume_step} 步恢复训练")
    
    # 训练循环
    iter_per_epoch = len(train_loader)
    global_step = resume_epoch * iter_per_epoch + resume_step
    best_loss = float('inf')
    
    for epoch in range(resume_epoch, epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        
        # 计算当前epoch需要跳过的步骤数
        start_step = resume_step if epoch == resume_epoch else 0
        
        for step, (X, Y) in enumerate(train_loader):
            # 如果是恢复的epoch，跳过已经训练过的步骤
            if step < start_step:
                continue
            
            # 使用Accelerate的梯度累积
            if accelerator:
                with accelerator.accumulate(model):
                    outputs = model(X, Y)
                    loss = outputs.last_loss
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # 传统训练方式
                if not hasattr(locals(), 'ctx'):
                    ctx = nullcontext()
                
                # 数据移动到设备
                non_blocking = (device_type == 'cuda')
                X = X.to(device_type, non_blocking=non_blocking)
                Y = Y.to(device_type, non_blocking=non_blocking)
                
                # 前向传播
                with ctx:
                    out = model(X, Y)
                    loss = out.last_loss / accumulation_steps
                
                # 反向传播
                if 'scaler' in locals() and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 梯度累积和更新
                if (step + 1) % accumulation_steps == 0:
                    if 'scaler' in locals() and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # 设备特定的内存管理
                    if device_type == 'mps':
                        torch.mps.empty_cache()
                    elif device_type == 'cuda':
                        torch.cuda.empty_cache()
            
            epoch_loss += loss.item() if accelerator else (loss.item() * accumulation_steps)
            global_step += 1
            
            # 定期保存和日志记录
            if step % save_interval == 0 and step > 0:
                spend_time = time.time() - start_time
                avg_loss = epoch_loss / (step + 1)
                estimated_time = spend_time / (step + 1) * iter_per_epoch
                current_lr = scheduler.get_last_lr()[0] if scheduler else learning_rate
                
                logger.info(
                    f'Epoch:[{epoch+1}/{epochs}]({step}/{iter_per_epoch}) '
                    f'loss:{avg_loss:.4f} lr:{current_lr:.7f} '
                    f'time:{spend_time/60:.1f}min eta:{(estimated_time-spend_time)/60:.1f}min'
                )
                
                # 保存模型
                if accelerator:
                    # 使用Accelerate保存
                    if accelerator.is_main_process:
                        ckp_path = save_dir / f'accelerate_pretrain_epoch{epoch+1}_step{step}'
                        accelerator.save_state(ckp_path)
                        
                        # 额外保存训练状态信息
                        training_state = {
                            'epoch': epoch,
                            'step': step,
                            'global_step': global_step,
                            'loss': avg_loss
                        }
                        torch.save(training_state, ckp_path / 'training_state.pt')
                        logger.info(f"模型已保存到: {ckp_path}")
                        
                        # 删除旧的Accelerate检查点
                        existing_accelerate_checkpoints = sorted(save_dir.glob('accelerate_pretrain_epoch*'))
                        if len(existing_accelerate_checkpoints) >= 2:
                            for old_ckp in existing_accelerate_checkpoints[:-1]:
                                try:
                                    # Accelerate检查点是目录，需要递归删除
                                    import shutil
                                    shutil.rmtree(old_ckp)
                                    logger.info(f"已删除旧Accelerate检查点: {old_ckp}")
                                except Exception as e:
                                    logger.warning(f"删除旧Accelerate检查点失败: {old_ckp}, 错误: {e}")
                else:
                    # 传统保存方式
                    model.eval()
                    ckp_path = save_dir / f'pretrain_epoch{epoch+1}_step{step}.pth'
                    
                    # 删除旧检查点
                    existing_checkpoints = sorted(save_dir.glob('pretrain_epoch*.pth'))
                    if len(existing_checkpoints) >= 2:
                        for old_ckp in existing_checkpoints[:-1]:
                            try:
                                old_ckp.unlink()
                                logger.info(f"已删除旧检查点: {old_ckp}")
                            except Exception as e:
                                logger.warning(f"删除旧检查点失败: {old_ckp}, 错误: {e}")
                    
                    state_dict = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'loss': avg_loss,
                        'config': lm_config
                    }
                    torch.save(state_dict, ckp_path)
                    logger.info(f"模型已保存到: {ckp_path}")
                    model.train()
        
        # Epoch结束时的统计
        epoch_time = time.time() - start_time
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        logger.info(
            f'Epoch {epoch+1}/{epochs} 完成 - '
            f'平均损失: {avg_epoch_loss:.4f}, '
            f'用时: {epoch_time/60:.1f}分钟'
        )
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            if accelerator:
                if accelerator.is_main_process:
                    best_model_path = save_dir / 'best_model'
                    accelerator.save_state(best_model_path)
                    logger.info(f"保存最佳模型: {best_model_path} (loss: {best_loss:.4f})")
            else:
                best_model_path = save_dir / 'best_model.pth'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                    'config': lm_config
                }, best_model_path)
                logger.info(f"保存最佳模型: {best_model_path} (loss: {best_loss:.4f})")
        
        # 设备特定的内存清理
        if device_type == 'mps':
            torch.mps.empty_cache()
        elif device_type == 'cuda':
            torch.cuda.empty_cache()
    
    # 训练完成后保存最终模型
    logger.info("训练完成！开始保存最终模型...")
    
    if accelerator:
        if accelerator.is_main_process:
            final_model_path = save_dir / f'accelerate_pretrain_model_{lm_config.dim}_{batch_size}'
            accelerator.save_state(final_model_path)
            logger.info(f"最终模型已保存到: {final_model_path}")
    else:
        model.eval()
        final_model_name = f'pretrain_model_{lm_config.dim}_{batch_size}.pth'
        final_ckp_path = save_dir / final_model_name
        
        final_state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epochs,
            'step': global_step,
            'loss': avg_epoch_loss,
            'config': lm_config,
            'training_completed': True
        }
        
        torch.save(final_state_dict, final_ckp_path)
        logger.info(f"最终模型已保存到: {final_ckp_path}")
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main()