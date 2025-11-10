import gradio as gr
import torch
import os
import json
from pre_transformer import Transformer, MyPretrainConfig

try:
    from pre_configurator import llmconfig, logger
    logger.info("已加载pre_configurator配置")
except ImportError as e:
    print("未找到pre_configurator.py，将使用默认配置")
    raise e

def load_model(model_path, config_path=None):
    """加载模型和配置"""
    global current_model, current_tokenizer, model_config
    
    try:
        if not os.path.exists(model_path):
            return f"模型文件不存在: {model_path}"
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            model_config = MyPretrainConfig(**config_dict)
        else:
            # 使用默认配置
            model_config = MyPretrainConfig()
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        current_model = Transformer(model_config)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            current_model.load_state_dict(checkpoint['model'])
        else:
            current_model.load_state_dict(checkpoint)
        
        current_model.to(device)
        current_model.eval()
        
        return f"模型加载成功！\n设备: {device}\n参数量: {sum(p.numel() for p in current_model.parameters()):,}"
    
    except Exception as e:
        return f"模型加载失败: {str(e)}"

def generate_response(message, history, max_length=512, temperature=0.7, top_p=0.9):
    """生成对话回复"""
    global current_model, model_config
    
    if current_model is None:
        return "请先加载模型！"
    
    try:
        device = next(current_model.parameters()).device
        
        # 简单的文本编码（这里需要根据实际的tokenizer进行调整）
        # 假设使用字符级编码作为示例
        vocab = {chr(i): i for i in range(32, 127)}  # ASCII可打印字符
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        
        def encode_text(text):
            return [vocab.get(c, vocab['<unk>']) for c in text[:max_length]]
        
        # 编码输入
        input_ids = encode_text(message)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # 生成回复
        with torch.no_grad():
            output = current_model.generate(
                input_tensor, 
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        
        # 解码输出（简化版本）
        reverse_vocab = {v: k for k, v in vocab.items()}
        response = ''.join([reverse_vocab.get(token.item(), '<unk>') for token in output[0]])
        
        return response.strip()
    
    except Exception as e:
        return f"生成回复时出错: {str(e)}"

def clear_chat():
    """清空对话历史"""
    return [], ""

# 创建 Gradio 界面
with gr.Blocks(title="MyLLM 对话系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 MyLLM 对话系统")
    gr.Markdown("从零训练的大语言模型对话界面")
    
    with gr.Row():
        # 左侧模型配置区域
        with gr.Column(scale=1):
            gr.Markdown("### 📁 模型配置")
            
            model_path = gr.Textbox(
                label="模型路径",
                placeholder="输入模型文件路径 (.pth/.pt)...",
                value="outputs/model_final.pth"
            )
            
            config_path = gr.Textbox(
                label="配置文件路径（可选）",
                placeholder="输入配置文件路径 (.json)...",
                value=""
            )
            
            load_btn = gr.Button("🔄 加载模型", variant="primary")
            model_status = gr.Textbox(
                label="模型状态",
                interactive=False,
                value="未加载模型"
            )
            
            gr.Markdown("### ⚙️ 生成参数")
            
            max_length = gr.Slider(
                minimum=50,
                maximum=2048,
                value=512,
                step=50,
                label="最大生成长度"
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="温度（创造性）"
            )
            
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p（多样性）"
            )
        
        # 右侧对话区域
        with gr.Column(scale=2):
            gr.Markdown("### 💬 对话界面")
            
            chatbot = gr.Chatbot(
                height=500,
                label="对话历史",
                show_label=False,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入消息",
                    placeholder="在这里输入您的消息...",
                    scale=4,
                    show_label=False
                )
                
                send_btn = gr.Button("📤 发送", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                
            # 示例问题
            gr.Markdown("### 💡 示例问题")
            example_questions = [
                "你好，请介绍一下自己",
                "什么是人工智能？",
                "请写一首关于春天的诗",
                "解释一下机器学习的基本概念"
            ]
            
            for question in example_questions:
                gr.Button(question, size="sm").click(
                    lambda q=question: q, outputs=msg
                )
    
    # 事件绑定
    def respond(message, history, max_len, temp, top_p_val):
        if not message.strip():
            return history, ""
        
        # 添加用户消息到历史
        history.append([message, None])
        
        # 生成回复
        bot_response = generate_response(message, history, max_len, temp, top_p_val)
        
        # 添加机器人回复到历史
        history[-1][1] = bot_response
        
        return history, ""
    
    # 绑定事件
    load_btn.click(
        load_model,
        inputs=[model_path, config_path],
        outputs=model_status
    )
    
    send_btn.click(
        respond,
        inputs=[msg, chatbot, max_length, temperature, top_p],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, max_length, temperature, top_p],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )