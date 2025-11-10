import gradio as gr
import sys
sys.path.append('../llmapi')
from translation import TranslationAgent
from llmapi.llm_factory import LLMFactory, LLMChatAdapter
from llmapi.util.mylog import logger

class AppUI:
    def __init__(self):
        self.translator = None
        self.llm_chat_adapter = None
        
    def initialize_model(self, model_type, model_name, temperature, top_p):
        """初始化模型"""
        try:
            llm = LLMFactory.create(model_type, model_name=model_name, temperature=temperature, top_p=top_p)
            self.translator = TranslationAgent(llm)
            self.llm_chat_adapter = LLMChatAdapter(llm)
            return f"✅ 模型 {model_type}/{model_name} 初始化成功"
        except Exception as e:
            return f"❌ 模型初始化失败: {str(e)}"
    
    def translate_text(self, source_lang, target_lang, source_text, country=""):
        """执行翻译"""
        if not self.translator:
            return "❌ 请先初始化模型"
        if not source_text.strip():
            return "❌ 请输入要翻译的文本"
        
        try:
            return self.translator.translate(source_lang, target_lang, source_text, country, self.llm_chat_adapter)
        except Exception as e:
            logger.error(f"翻译失败: {str(e)}")
            return f"❌ 翻译失败: {str(e)}"
    
    def chat_with_agent(self, message, history):
        """与智能体对话"""
        if not self.llm_chat_adapter:
            return history, ""
        
        try:
            # 构建对话上下文
            conversation = ""
            for msg in history:
                if msg["role"] == "user":
                    conversation += f"用户: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    conversation += f"助手: {msg['content']}\n"
            
            # 添加当前用户消息
            conversation += f"用户: {message}\n助手: "
            
            # 获取回复 - LLMChatAdapter.chat 返回 (bool, str) 元组
            success, response = self.llm_chat_adapter.chat(conversation)
            
            if not success:
                response = f"对话失败: {response}"
            
            # 更新历史记录
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return history, ""
        except Exception as e:
            error_msg = f"对话失败: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def create_interface(self):
        """创建 Gradio 界面"""
        with gr.Blocks(title="AI 翻译助手") as demo:
            gr.Markdown("# 🤖 AI 翻译助手")
            
            # 模型配置
            with gr.Row():
                model_type = gr.Dropdown(
                    choices=["qianfan", "openai", "qwen", "zhipu", "ollama", "siliconflow"],
                    value="qianfan", label="模型类型"
                )
                model_name = gr.Textbox(value="deepseek-v3", label="模型名称")
                temperature = gr.Slider(0.0, 2.0, 0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, 0.9, step=0.05, label="Top-p")
            
            init_btn = gr.Button("初始化模型", variant="primary")
            init_status = gr.Textbox(label="状态", interactive=False)
            
            init_btn.click(self.initialize_model, [model_type, model_name, temperature, top_p], init_status)
            
            with gr.Tab("对话"):
                chatbot = gr.Chatbot(height=400, type="messages")
                msg = gr.Textbox(label="消息", placeholder="请输入您的问题...")
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空")
                
                send_btn.click(self.chat_with_agent, [msg, chatbot], [chatbot, msg])
                msg.submit(self.chat_with_agent, [msg, chatbot], [chatbot, msg])
                clear_btn.click(
                    fn=lambda: ([], ""), 
                    inputs=[], 
                    outputs=[chatbot, msg]
                )
            
            with gr.Tab("翻译"):
                with gr.Row():
                    source_lang = gr.Dropdown(
                        choices=["English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish", "Russian"],
                        value="English", label="源语言"
                    )
                    target_lang = gr.Dropdown(
                        choices=["Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Russian"],
                        value="Chinese", label="目标语言"
                    )
                    country = gr.Textbox(label="地区 (可选)", placeholder="例如: Taiwan")
                
                with gr.Row():
                    source_text = gr.Textbox(label="待翻译文本", lines=6, placeholder="请输入要翻译的文本...")
                    translation_result = gr.Textbox(label="翻译结果", lines=6, interactive=False)
                
                translate_btn = gr.Button("开始翻译", variant="primary")
                translate_btn.click(self.translate_text, [source_lang, target_lang, source_text, country], translation_result)
        
        return demo

def main():
    try:
        ui = AppUI()
        demo = ui.create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        print(f"❌ 应用启动失败: {str(e)}")

if __name__ == "__main__":
    main()