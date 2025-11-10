import os
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from peft import get_peft_model, LoraConfig

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