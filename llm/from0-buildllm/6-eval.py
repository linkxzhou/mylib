import os
import time
import torch
from transformers import AutoTokenizer
from pre_transformer import (
    Transformer, 
    count_parameters,
)
from pre_pretrainconfig import (
    MyPretrainConfig,
    validate_config,
    validate_model_inputs
)

try:
    from pre_configurator import llmconfig, logger
    logger.info("已加载pre_configurator配置")
except ImportError as e:
    print("未找到pre_configurator.py，将使用默认配置")
    raise e

def load_model_config():
    """从配置文件加载模型配置"""
    try:
        # 从配置中获取模型参数
        model_config = llmconfig.get('model_config', {})
        
        config = MyPretrainConfig(
            dim=model_config.get('dim', 512),
            n_layers=model_config.get('n_layers', 8),
            n_heads=model_config.get('n_heads', 8),
            n_kv_heads=model_config.get('n_kv_heads', 8),
            vocab_size=model_config.get('vocab_size', 6400),
            hidden_dim=model_config.get('hidden_dim', None),
            multiple_of=model_config.get('multiple_of', 64),
            norm_eps=model_config.get('norm_eps', 1e-6),
            max_seq_len=model_config.get('max_seq_len', 256),
            dropout=model_config.get('dropout', 0.0),
            flash_attn=model_config.get('flash_attn', True)
        )
        
        # 验证配置
        validate_config(config)
        logger.info(f"模型配置加载成功: {config.get_config_info()}")
        return config
        
    except Exception as e:
        logger.error(f"加载模型配置失败: {e}")
        raise

def load_evaluation_config():
    """加载评估配置"""
    eval_config = {
        'batch_size': llmconfig.get('model_config.train.batch_size', 8),
        'max_seq_len': llmconfig.get('model_config.max_seq_len', 256),
        'temperature': llmconfig.get('evaluation_config.generation_config.temperature', 0.7),
        'top_k': llmconfig.get('evaluation_config.generation_config.top_k', 50),
        'max_new_tokens': llmconfig.get('evaluation_config.generation_config.max_new_tokens', 512),
    }
    
    logger.info(f"评估配置: {eval_config}")
    return eval_config

def generate_response(model, tokenizer, prompt, config, device, stream=True):
    """生成回答"""
    try:
        # 验证输入
        if not prompt.strip():
            raise ValueError("输入提示不能为空")
        
         # 使用简单的格式，您可以根据需要调整
        formatted_prompt = f"用户: {prompt}\n助手: "
        
        # 限制长度
        if len(formatted_prompt) > config['max_seq_len'] - 1:
            formatted_prompt = formatted_prompt[-(config['max_seq_len'] - 1):]
        
        # 编码输入
        input_ids = tokenizer(formatted_prompt).data['input_ids']
        x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
        
        # 验证模型输入
        validate_model_inputs(x, model.params)
        
        # 生成回答
        with torch.no_grad():
            res_y = model.generate(
                x, 
                tokenizer.eos_token_id,
                max_new_tokens=config['max_new_tokens'],
                temperature=config['temperature'],
                top_k=config['top_k'],
                stream=stream
            )
            
            return res_y, formatted_prompt
            
    except Exception as e:
        logger.error(f"生成回答失败: {e}")
        raise

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

def load_model_and_tokenizer(config, device_type):
    """初始化模型并加载预训练权重"""
    model = Transformer(config)
    
    tokenizer_path = llmconfig.get('paths.tokenizer_dir', 'my_tokenizer')
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 构建检查点路径
    checkpoint_dir = llmconfig.get('paths.checkpoint_dir', 'outputs')
    # 加载预训练模型
    model_name = f'pretrain_epoch1_step99000.pth'
    ckp_path = os.path.join(checkpoint_dir, model_name)
    
    if os.path.exists(ckp_path):
        try:
            # 实际加载检查点
            checkpoint = torch.load(ckp_path, map_location=device_type, weights_only=True)
            
            # 检查检查点格式并加载模型权重
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
            logger.info(f"成功加载预训练模型: {ckp_path}")
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            logger.warning("使用随机初始化的模型")
    else:
        logger.warning(f"预训练模型不存在: {ckp_path}，使用随机初始化")
        raise FileNotFoundError(f"预训练模型不存在: {ckp_path}")
    
    logger.info(f'模型加载完成，LLM总参数量：{count_parameters(model)["total_parameters"] / 1e6:.3f} 百万')
    model = model.to(device_type)
    return model, tokenizer

def main():
    try:
        # 设置随机种子
        torch.manual_seed(llmconfig.get('model_config.train.seed', 1337))

        # 加载评估配置
        eval_config = load_evaluation_config()
        device_type = llmconfig.get('model_config.train.device_type', 'cpu')
        
        # 初始化模型
        logger.info("开始初始化模型...")
        lmconfig = create_config_from_llmconfig()
        model, tokenizer = load_model_and_tokenizer(lmconfig, device_type)
        model = model.eval()
        
        # 测试问题
        prompt_datas = [
            '但是我喜欢守号买大乐透，中了头奖1500万（无追加），怎么办',
        ]
        
        # 评估模式配置
        contain_history_chat = False
        answer_way = 0  # 0: 自动问答, 1: 交互式
        stream = True
        
        messages_origin = []
        messages = messages_origin
        qa_index = 0
        
        logger.info("开始评估...")
        
        while True:
            start = time.time()
            
            if not contain_history_chat:
                messages = messages_origin.copy()
            
            # 获取问题
            if answer_way == 1:
                prompt = input('用户：')
                if prompt.lower() in ['quit', 'exit', '退出']:
                    break
            else:
                if qa_index >= len(prompt_datas):
                    break
                prompt = prompt_datas[qa_index]
                logger.info(f'问题 {qa_index + 1}: {prompt}')
                qa_index += 1
            
            try:
                # 生成回答
                res_y, formatted_prompt = generate_response(
                    model, tokenizer, prompt, eval_config, device_type, stream
                )
                
                logger.info('回答：')
                
                try:
                    y = next(res_y)
                except StopIteration:
                    logger.info("No answer")
                    continue
                
                history_idx = 0
                answer = ""
                
                while y is not None:
                    answer = tokenizer.decode(y[0].tolist())
                    
                    # 处理不完整的字符
                    if answer and answer[-1] == '�':
                        try:
                            y = next(res_y)
                        except StopIteration:
                            break
                        continue
                    
                    if not len(answer):
                        try:
                            y = next(res_y)
                        except StopIteration:
                            break
                        continue
                    
                    # 流式输出
                    print(answer[history_idx:], end='', flush=True)
                    
                    try:
                        y = next(res_y)
                    except StopIteration:
                        break
                    
                    history_idx = len(answer)
                    
                    if not stream:
                        break
                
                logger.info('\n')
                
                # 保存对话历史
                if contain_history_chat:
                    assistant_answer = answer.replace(formatted_prompt, "")
                    messages.append({"role": "user", "content": prompt})
                    messages.append({"role": "assistant", "content": assistant_answer})
                
            except Exception as e:
                logger.error(f"处理问题时出错: {e}")
                logger.info(f"处理问题时出错: {e}")
                continue
            
            end = time.time()
            logger.info(f"耗时: {end - start:.2f} 秒\n")
        
        logger.info("评估完成")
        
    except Exception as e:
        logger.error(f"评估过程中出现错误: {e}")
        logger.info(f"评估失败: {e}")
        raise

if __name__ == "__main__":
    main()