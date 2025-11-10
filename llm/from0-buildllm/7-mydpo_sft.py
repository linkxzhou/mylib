from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
import shutil, os

def init_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    AutoConfig.register(MyPretrainConfig.model_type, MyPretrainConfig)
    AutoModelForCausalLM.register(MyPretrainConfig, Transformer)
    my_tokenizer = "./my_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(my_tokenizer, trust_remote_code=True, use_fast=False)
    ckp = f'./out/full_sft_{lm_config.dim}.pth.{batch_size}'

    print(f"lmconfigs: {lm_config.to_json_string()}")
    with open(ckp_path + "/config.json", 'w') as f:
        f.write(lm_config.to_json_string())

    # 拷贝文件到指定的目录
    for item in os.listdir(my_tokenizer):
        src_item = os.path.join(my_tokenizer, item)
        if os.path.isfile(src_item):
            dest_item = os.path.join(ckp_path, item)
            shutil.copy2(src_item, dest_item)
    shutil.copy2(ckp, ckp_path + "/pytorch_model.bin")

    model = AutoModelForCausalLM.from_pretrained(ckp_path, trust_remote_code=True).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    tokenizer.pad_token = tokenizer.eos_token
    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(device)
    return model, tokenizer

if __name__ == '__main__':
    lm_config = MyPretrainConfig()
    max_seq_len = lm_config.max_seq_len
    out_dir = 'out'
    epochs = 20             # 训练轮数
    batch_size = 8          # batch_size
    learning_rate = 1e-5    # 学习率
    device = 'cuda:0'       # or cpu
    dtype = 'bfloat16'

    ckp_path = f'./my_checkpoint'
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    model, tokenizer = init_model()
    training_config = DPOConfig(
        output_dir=ckp_path,
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to="none",
        save_steps=2000,
        learning_rate=learning_rate,
        save_safetensors=False,
    )

    # 下载训练图片：https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main/dpo
    dataset_path = f'{basepath}/dpo_train_data.json'
    train_dataset = load_dataset('json', data_files=dataset_path)
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        beta=0.1,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512
    )
    dpo_trainer.train()
    dpo_trainer.save_model(
        output_dir=f"./out/dpo_sft_{lm_config.dim}.pth.{batch_size}"
    )