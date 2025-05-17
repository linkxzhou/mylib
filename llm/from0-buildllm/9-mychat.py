import gradio as gr

# 定义一个处理函数
def chat_response(training_data, text):
    # 这里可以添加你的聊天逻辑
    response_text = f"训练数据: {training_data}\n你说: {text}"
    return response_text

# 创建 Gradio 接口
with gr.Blocks() as demo:
    gr.Markdown("## myllm（从0训练大模型）")
    
    with gr.Row():
        # 左侧输入区域
        with gr.Column():
            config_input = gr.Textbox(label="训练配置", placeholder="输入配置文件路径(JSON格式)...")
            pretain_input = gr.Textbox(label="预训练数据", placeholder="输入预训练文件路径...")
            fullsft_input = gr.Textbox(label="FULL SFT数据", placeholder="输入FULL SFT文件路径...")
            lorasft_input = gr.Textbox(label="Lora SFT数据", placeholder="输入Lora SFT文件路径...")
            dposft_input = gr.Textbox(label="DPO SFT数据", placeholder="输入DPO SFT文件路径...")
            train_submit_button = gr.Button("开始训练")
        
        # 右侧对话框区域
        with gr.Column():
            output_text = gr.Textbox(label="聊天回复", interactive=False)
            text_input = gr.Textbox(label="输入你的消息", placeholder="在这里输入文本...")
            image_input = gr.Image(type="pil", label="上传图片")
            chat_submit_button = gr.Button("提交")
    
    train_submit_button.click(chat_response, inputs=[config_input, pretain_input, fullsft_input, lorasft_input, dposft_input], outputs=None)
    chat_submit_button.click(chat_response, inputs=[text_input, image_input], outputs=output_text)

# 启动应用，设置端口为 7860
demo.launch(server_name="0.0.0.0", server_port=7860)