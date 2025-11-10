#!/usr/bin/env python3
"""
AnythingConvert Gradio Web界面
提供图片、视频、音频、文档转换的Web界面
"""

import gradio as gr
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Generator

from converters.image import ImageConverter
from converters.video import VideoConverter
from converters.audio import AudioConverter
from converters.document import DocumentConverter

class AnythingConvertApp:
    """AnythingConvert Gradio应用主类"""
    
    def __init__(self):
        self.image_converter = ImageConverter()
        self.video_converter = VideoConverter()
        self.audio_converter = AudioConverter()
        self.document_converter = DocumentConverter()
        
        # 支持的格式
        self.image_formats = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg', 'ico']
        self.video_formats = ['mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm']
        self.audio_formats = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma']
        
        # 文档格式 - 按类别组织
        self.document_formats = {
            '常用格式': ['pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'],
            '标记语言': ['md', 'markdown', 'rst', 'asciidoc', 'org', 'textile'],
            'HTML/Web': ['html', 'htm', 'html5', 'xhtml'],
            '电子书': ['epub', 'epub3', 'fb2'],
            'TeX/LaTeX': ['latex', 'tex', 'context'],
            '幻灯片': ['pptx', 'beamer', 'revealjs', 'slidy', 'slideous', 's5', 'dzslides'],
            'Wiki格式': ['mediawiki', 'dokuwiki', 'jira', 'creole'],
            '数据格式': ['csv', 'tsv', 'json', 'xml'],
            '参考文献': ['bibtex', 'bib', 'biblatex', 'csljson', 'ris'],
            '其他格式': ['opml', 'ipynb', 'icml', 'typst', 'native', 'plain']
        }
        
        # 扁平化的格式列表（用于下拉菜单）
        self.all_document_formats = []
        for category, formats in self.document_formats.items():
            self.all_document_formats.extend(formats)
        
        # 获取支持的格式信息
        try:
            format_info = self.document_converter.get_supported_formats()
            self.input_formats = format_info.get('input', self.all_document_formats)
            self.output_formats = format_info.get('output', self.all_document_formats)
            self.bidirectional_formats = format_info.get('bidirectional', [])
        except:
            # 如果获取失败，使用默认格式
            self.input_formats = self.all_document_formats
            self.output_formats = self.all_document_formats
            self.bidirectional_formats = []
    
    def convert_image(self, 
                     input_file, 
                     output_format: str,
                     quality: int = 95,
                     width: Optional[int] = None,
                     height: Optional[int] = None,
                     keep_aspect: bool = True,
                     progress=gr.Progress()) -> Tuple[str, str]:
        """图片转换功能"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            progress(0.1, desc="🔍 正在准备图片转换...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.3, desc=f"🖼️ 正在转换为{output_format.upper()}格式...")
            
            # 执行转换
            result = self.image_converter.convert(
                input_path=Path(input_file.name),
                output_path=Path(output_path),
                quality=quality,
                width=width if width > 0 else None,
                height=height if height > 0 else None,
                keep_aspect_ratio=keep_aspect
            )
            
            progress(0.9, desc="🖼️ 正在完成图片转换...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ 图片转换完成")
                return output_path, f"✅ 图片转换成功！输出格式: {output_format.upper()}"
            else:
                progress(1.0, desc="❌ 图片转换失败")
                return None, "❌ 图片转换失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 转换过程中出现错误")
            return None, f"❌ 转换错误: {str(e)}"
    
    def compress_image(self, 
                      input_file,
                      quality: int = 85,
                      max_size_kb: Optional[int] = None,
                      progress=gr.Progress()) -> Tuple[str, str]:
        """图片压缩功能"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            progress(0.1, desc="🔍 正在准备图片压缩...")
            
            # 获取原文件扩展名
            input_path = Path(input_file.name)
            ext = input_path.suffix
            
            progress(0.2, desc="⚙️ 正在分析图片大小...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.4, desc="🗜️ 正在压缩图片...")
            
            # 执行压缩
            result = self.image_converter.compress(
                input_path=input_path,
                output_path=Path(output_path),
                quality=quality,
                max_size_kb=max_size_kb if max_size_kb > 0 else None
            )
            
            progress(0.8, desc="📊 正在计算压缩比...")
            
            if result:
                # 计算压缩比
                original_size = input_path.stat().st_size
                compressed_size = Path(output_path).stat().st_size
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                progress(1.0, desc="✅ 图片压缩完成")
                return output_path, f"✅ 图片压缩成功！压缩率: {compression_ratio:.1f}%"
            else:
                progress(1.0, desc="❌ 图片压缩失败")
                return None, "❌ 图片压缩失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 压缩过程中出现错误")
            return None, f"❌ 压缩错误: {str(e)}"
    
    def convert_video(self,
                     input_file,
                     output_format: str,
                     codec: Optional[str] = None,
                     bitrate: Optional[str] = None,
                     resolution: Optional[str] = None,
                     fps: Optional[int] = None,
                     progress=gr.Progress()) -> Tuple[str, str]:
        """视频转换功能"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            progress(0.1, desc="🔍 正在准备视频转换...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.2, desc="⚙️ 正在配置转换参数...")
            
            progress(0.3, desc=f"🎬 正在转换为{output_format.upper()}格式...")
            
            # 执行转换
            result = self.video_converter.convert(
                input_path=Path(input_file.name),
                output_path=Path(output_path),
                codec=codec if codec else None,
                bitrate=bitrate if bitrate else None,
                resolution=resolution if resolution else None,
                fps=fps if fps > 0 else None
            )
            
            progress(0.9, desc="🎬 正在完成视频转换...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ 视频转换完成")
                return output_path, f"✅ 视频转换成功！输出格式: {output_format.upper()}"
            else:
                progress(1.0, desc="❌ 视频转换失败")
                return None, "❌ 视频转换失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 转换过程中出现错误")
            return None, f"❌ 转换错误: {str(e)}"
    
    def get_video_info(self, input_file, progress=gr.Progress()) -> str:
        """获取视频信息"""
        try:
            if input_file is None:
                return "❌ 请选择视频文件"
            
            # 显示进度
            progress(0.1, desc="🔍 正在分析视频文件...")
            time.sleep(0.1)  # 短暂延迟以显示进度
            
            progress(0.3, desc="📊 正在读取视频元数据...")
            info = self.video_converter.get_video_info(Path(input_file.name))
            
            progress(0.7, desc="📝 正在格式化信息...")
            time.sleep(0.1)
            
            info_text = f"""
📹 **视频信息**
- **文件名**: {Path(input_file.name).name}
- **格式**: {info.get('format', 'N/A')}
- **时长**: {info.get('duration', 'N/A')} 秒
- **分辨率**: {info.get('width', 'N/A')}x{info.get('height', 'N/A')}
- **帧率**: {info.get('fps', 'N/A')} fps
- **比特率**: {info.get('bitrate', 'N/A')}
- **编码器**: {info.get('codec', 'N/A')}
            """
            
            progress(1.0, desc="✅ 视频信息获取完成")
            return info_text.strip()
            
        except Exception as e:
            progress(1.0, desc="❌ 获取视频信息失败")
            return f"❌ 获取视频信息失败: {str(e)}"
    
    def extract_audio_from_video(self, input_file, audio_format: str = 'mp3', progress=gr.Progress()) -> Tuple[str, str]:
        """从视频提取音频"""
        try:
            if input_file is None:
                return None, "❌ 请选择视频文件"
            
            progress(0.1, desc="🔍 正在准备音频提取...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.3, desc=f"🎵 正在从视频提取{audio_format.upper()}音频...")
            
            # 提取音频
            result = self.video_converter.extract_audio(
                input_path=Path(input_file.name),
                output_path=Path(output_path),
                audio_format=audio_format
            )
            
            progress(0.9, desc="🎵 正在完成音频提取...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ 音频提取完成")
                return output_path, f"✅ 音频提取成功！格式: {audio_format.upper()}"
            else:
                progress(1.0, desc="❌ 音频提取失败")
                return None, "❌ 音频提取失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 提取过程中出现错误")
            return None, f"❌ 提取错误: {str(e)}"
    
    def convert_audio(self,
                     input_file,
                     output_format: str,
                     bitrate: Optional[str] = None,
                     sample_rate: Optional[int] = None,
                     channels: Optional[int] = None,
                     progress=gr.Progress()) -> Tuple[str, str]:
        """音频转换功能"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            progress(0.1, desc="🔍 正在准备音频转换...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.2, desc="⚙️ 正在配置音频参数...")
            
            progress(0.3, desc=f"🎵 正在转换为{output_format.upper()}格式...")
            
            # 执行转换
            result = self.audio_converter.convert(
                input_path=input_file.name,
                output_path=output_path,
                bitrate=bitrate if bitrate else None,
                sample_rate=sample_rate if sample_rate > 0 else None,
                channels=channels if channels > 0 else None
            )
            
            progress(0.9, desc="🎵 正在完成音频转换...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ 音频转换完成")
                return output_path, f"✅ 音频转换成功！输出格式: {output_format.upper()}"
            else:
                progress(1.0, desc="❌ 音频转换失败")
                return None, "❌ 音频转换失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 转换过程中出现错误")
            return None, f"❌ 转换错误: {str(e)}"
    
    def get_audio_info(self, input_file, progress=gr.Progress()) -> str:
        """获取音频信息"""
        try:
            if input_file is None:
                return "❌ 请选择音频文件"
            
            # 显示进度
            progress(0.1, desc="🔍 正在分析音频文件...")
            time.sleep(0.1)
            
            progress(0.3, desc="🎵 正在读取音频元数据...")
            info = self.audio_converter.get_audio_info(input_file.name)
            
            progress(0.7, desc="📝 正在格式化信息...")
            time.sleep(0.1)
            
            info_text = f"""
🎵 **音频信息**
- **文件名**: {Path(input_file.name).name}
- **格式**: {info.get('format', 'N/A')}
- **时长**: {info.get('duration', 'N/A')} 秒
- **比特率**: {info.get('bitrate', 'N/A')} kbps
- **采样率**: {info.get('sample_rate', 'N/A')} Hz
- **声道数**: {info.get('channels', 'N/A')}
- **编码器**: {info.get('codec', 'N/A')}
            """
            
            progress(1.0, desc="✅ 音频信息获取完成")
            return info_text.strip()
            
        except Exception as e:
            progress(1.0, desc="❌ 获取音频信息失败")
            return f"❌ 获取音频信息失败: {str(e)}"
    
    def extract_audio_segment(self,
                            input_file,
                            start_time: float,
                            duration: float,
                            output_format: str = 'mp3',
                            progress=gr.Progress()) -> Tuple[str, str]:
        """提取音频片段"""
        try:
            if input_file is None:
                return None, "❌ 请选择音频文件"
            
            progress(0.1, desc="🔍 正在准备音频片段提取...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.3, desc=f"✂️ 正在提取音频片段 ({start_time}s - {start_time + duration}s)...")
            
            # 提取片段
            result = self.audio_converter.extract_segment(
                input_path=input_file.name,
                output_path=output_path,
                start_time=start_time,
                duration=duration
            )
            
            progress(0.9, desc="✂️ 正在完成音频片段提取...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ 音频片段提取完成")
                return output_path, f"✅ 音频片段提取成功！时长: {duration}秒"
            else:
                progress(1.0, desc="❌ 音频片段提取失败")
                return None, "❌ 音频片段提取失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 提取过程中出现错误")
            return None, f"❌ 提取错误: {str(e)}"
    
    def convert_document(self, 
                        input_file, 
                        output_format: str,
                        input_format: str = 'auto',
                        pdf_engine: str = 'xelatex',
                        extra_args: str = '',
                        progress=gr.Progress()) -> Tuple[str, str]:
        """文档转换功能"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            # 显示进度
            progress(0.1, desc="🔍 正在准备转换...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.2, desc="⚙️ 正在处理转换参数...")
            
            # 处理额外参数
            extra_args_list = []
            if extra_args.strip():
                extra_args_list = extra_args.strip().split()
            
            # 如果指定了PDF引擎且输出格式是PDF
            if output_format == 'pdf' and pdf_engine != 'xelatex':
                extra_args_list.extend([f'--pdf-engine={pdf_engine}'])
            
            progress(0.3, desc="🔄 正在执行文档转换...")
            
            # 执行转换
            if input_format == 'auto':
                # 自动检测输入格式
                progress(0.4, desc="🔍 正在自动检测输入格式...")
                result = self.document_converter.convert(
                    input_path=input_file.name,
                    output_path=output_path,
                    input_format=None,  # 自动检测
                    output_format=output_format,
                    extra_args=extra_args_list if extra_args_list else None,
                    pdf_engine=pdf_engine
                )
            else:
                # 手动指定输入格式
                progress(0.4, desc=f"🔄 正在从 {input_format.upper()} 转换为 {output_format.upper()}...")
                result = self.document_converter.convert(
                    input_path=input_file.name,
                    output_path=output_path,
                    input_format=input_format,
                    output_format=output_format,
                    extra_args=extra_args_list if extra_args_list else None,
                    pdf_engine=pdf_engine
                )
            
            progress(0.9, desc="📝 正在完成转换...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ 文档转换完成")
                return output_path, f"✅ 文档转换成功！输出格式: {output_format.upper()}"
            else:
                progress(1.0, desc="❌ 文档转换失败")
                return None, "❌ 文档转换失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 转换过程中出现错误")
            return None, f"❌ 转换错误: {str(e)}"
    
    def get_document_info(self, input_file, progress=gr.Progress()) -> str:
        """获取文档信息"""
        try:
            if input_file is None:
                return "❌ 请选择文档文件"
            
            # 显示进度
            progress(0.1, desc="🔍 正在分析文档文件...")
            time.sleep(0.1)
            
            progress(0.2, desc="📄 正在读取文档内容...")
            time.sleep(0.1)
            
            progress(0.5, desc="📊 正在分析文档结构...")
            info = self.document_converter.get_document_info(input_file.name)
            
            progress(0.8, desc="📝 正在格式化信息...")
            time.sleep(0.1)
            
            info_text = f"""
📄 **文档信息**
- **文件名**: {info.get('name', 'N/A')}
- **格式**: {info.get('format', 'N/A').upper()}
- **文件大小**: {info.get('size_mb', 'N/A'):.2f} MB
- **文本长度**: {info.get('text_length', 'N/A'):,} 字符
- **单词数**: {info.get('word_count', 'N/A'):,}
- **行数**: {info.get('line_count', 'N/A'):,}
- **段落数**: {info.get('paragraph_count', 'N/A'):,}
            """
            
            progress(1.0, desc="✅ 文档信息获取完成")
            return info_text.strip()
            
        except Exception as e:
            progress(1.0, desc="❌ 获取文档信息失败")
            return f"❌ 获取文档信息失败: {str(e)}"
    
    def convert_to_markdown(self, input_file, progress=gr.Progress()) -> Tuple[str, str]:
        """快速转换为Markdown"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            progress(0.1, desc="🔍 正在准备转换为Markdown...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.3, desc="📝 正在转换为Markdown格式...")
            
            # 使用便捷方法转换
            result = self.document_converter.convert_to_markdown(
                input_path=input_file.name,
                output_path=output_path
            )
            
            progress(0.9, desc="📝 正在完成Markdown转换...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ Markdown转换完成")
                return output_path, "✅ 转换为Markdown成功！"
            else:
                progress(1.0, desc="❌ Markdown转换失败")
                return None, "❌ 转换为Markdown失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 转换过程中出现错误")
            return None, f"❌ 转换错误: {str(e)}"
    
    def convert_to_html(self, input_file, standalone: bool = True, progress=gr.Progress()) -> Tuple[str, str]:
        """快速转换为HTML"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            progress(0.1, desc="🔍 正在准备转换为HTML...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.3, desc="🌐 正在转换为HTML格式...")
            
            # 使用便捷方法转换
            result = self.document_converter.convert_to_html(
                input_path=input_file.name,
                output_path=output_path,
                standalone=standalone
            )
            
            progress(0.9, desc="🌐 正在完成HTML转换...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ HTML转换完成")
                return output_path, "✅ 转换为HTML成功！"
            else:
                progress(1.0, desc="❌ HTML转换失败")
                return None, "❌ 转换为HTML失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 转换过程中出现错误")
            return None, f"❌ 转换错误: {str(e)}"
    
    def convert_to_slides(self, input_file, slide_format: str = 'revealjs', progress=gr.Progress()) -> Tuple[str, str]:
        """快速转换为幻灯片"""
        try:
            if input_file is None:
                return None, "❌ 请选择输入文件"
            
            progress(0.1, desc="🔍 正在准备转换为幻灯片...")
            
            # 确定文件扩展名
            ext_map = {
                'revealjs': 'html',
                'slidy': 'html',
                'slideous': 'html',
                's5': 'html',
                'dzslides': 'html',
                'beamer': 'pdf',
                'pptx': 'pptx'
            }
            ext = ext_map.get(slide_format, 'html')
            
            progress(0.2, desc=f"⚙️ 正在配置{slide_format.upper()}格式...")
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            progress(0.4, desc=f"🎞️ 正在转换为{slide_format.upper()}幻灯片...")
            
            # 使用便捷方法转换
            result = self.document_converter.convert_to_slides(
                input_path=input_file.name,
                output_path=output_path,
                slide_format=slide_format
            )
            
            progress(0.9, desc="🎞️ 正在完成幻灯片转换...")
            time.sleep(0.1)
            
            if result:
                progress(1.0, desc="✅ 幻灯片转换完成")
                return output_path, f"✅ 转换为{slide_format.upper()}幻灯片成功！"
            else:
                progress(1.0, desc="❌ 幻灯片转换失败")
                return None, f"❌ 转换为{slide_format.upper()}幻灯片失败"
                
        except Exception as e:
            progress(1.0, desc="❌ 转换过程中出现错误")
            return None, f"❌ 转换错误: {str(e)}"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 图片转换界面
        with gr.Blocks(title="AnythingConvert - 图片转换") as image_interface:
            gr.Markdown("# 🖼️ 图片转换")
            
            with gr.Tab("格式转换"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.File(label="选择图片文件", file_types=["image"])
                        image_format = gr.Dropdown(
                            choices=self.image_formats,
                            value="png",
                            label="输出格式"
                        )
                        image_quality = gr.Slider(1, 100, 95, label="质量")
                        
                        with gr.Row():
                            image_width = gr.Number(label="宽度 (像素)", value=0, precision=0)
                            image_height = gr.Number(label="高度 (像素)", value=0, precision=0)
                        
                        image_keep_aspect = gr.Checkbox(label="保持宽高比", value=True)
                        image_convert_btn = gr.Button("转换图片", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.File(label="转换结果")
                        image_status = gr.Textbox(label="状态", interactive=False)
                
                image_convert_btn.click(
                    self.convert_image,
                    inputs=[image_input, image_format, image_quality, image_width, image_height, image_keep_aspect],
                    outputs=[image_output, image_status]
                )
            
            with gr.Tab("图片压缩"):
                with gr.Row():
                    with gr.Column():
                        compress_input = gr.File(label="选择图片文件", file_types=["image"])
                        compress_quality = gr.Slider(1, 100, 85, label="压缩质量")
                        compress_max_size = gr.Number(label="最大文件大小 (KB)", value=0, precision=0)
                        compress_btn = gr.Button("压缩图片", variant="primary")
                    
                    with gr.Column():
                        compress_output = gr.File(label="压缩结果")
                        compress_status = gr.Textbox(label="状态", interactive=False)
                
                compress_btn.click(
                    self.compress_image,
                    inputs=[compress_input, compress_quality, compress_max_size],
                    outputs=[compress_output, compress_status]
                )
        
        # 视频转换界面
        with gr.Blocks(title="AnythingConvert - 视频转换") as video_interface:
            gr.Markdown("# 🎬 视频转换")
            
            with gr.Tab("格式转换"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(label="选择视频文件", file_types=["video"])
                        video_format = gr.Dropdown(
                            choices=self.video_formats,
                            value="mp4",
                            label="输出格式"
                        )
                        video_codec = gr.Textbox(label="视频编码器 (可选)", placeholder="如: libx264, libx265")
                        video_bitrate = gr.Textbox(label="视频比特率 (可选)", placeholder="如: 1M, 2000k")
                        video_resolution = gr.Textbox(label="分辨率 (可选)", placeholder="如: 1920x1080")
                        video_fps = gr.Number(label="帧率 (可选)", value=0, precision=0)
                        video_convert_btn = gr.Button("转换视频", variant="primary")
                    
                    with gr.Column():
                        video_output = gr.File(label="转换结果")
                        video_status = gr.Textbox(label="状态", interactive=False)
                
                video_convert_btn.click(
                    self.convert_video,
                    inputs=[video_input, video_format, video_codec, video_bitrate, video_resolution, video_fps],
                    outputs=[video_output, video_status]
                )
            
            with gr.Tab("视频信息"):
                with gr.Row():
                    with gr.Column():
                        video_info_input = gr.File(label="选择视频文件", file_types=["video"])
                        video_info_btn = gr.Button("获取信息", variant="secondary")
                    
                    with gr.Column():
                        video_info_output = gr.Markdown(label="视频信息")
                
                video_info_btn.click(
                    self.get_video_info,
                    inputs=[video_info_input],
                    outputs=[video_info_output]
                )
            
            with gr.Tab("提取音频"):
                with gr.Row():
                    with gr.Column():
                        extract_video_input = gr.File(label="选择视频文件", file_types=["video"])
                        extract_audio_format = gr.Dropdown(
                            choices=self.audio_formats,
                            value="mp3",
                            label="音频格式"
                        )
                        extract_audio_btn = gr.Button("提取音频", variant="primary")
                    
                    with gr.Column():
                        extract_audio_output = gr.File(label="提取结果")
                        extract_audio_status = gr.Textbox(label="状态", interactive=False)
                
                extract_audio_btn.click(
                    self.extract_audio_from_video,
                    inputs=[extract_video_input, extract_audio_format],
                    outputs=[extract_audio_output, extract_audio_status]
                )
        
        # 音频转换界面
        with gr.Blocks(title="AnythingConvert - 音频转换") as audio_interface:
            gr.Markdown("# 🎵 音频转换")
            
            with gr.Tab("格式转换"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.File(label="选择音频文件", file_types=["audio"])
                        audio_format = gr.Dropdown(
                            choices=self.audio_formats,
                            value="mp3",
                            label="输出格式"
                        )
                        audio_bitrate = gr.Textbox(label="比特率 (可选)", placeholder="如: 128k, 320k")
                        audio_sample_rate = gr.Number(label="采样率 (可选)", value=0, precision=0)
                        audio_channels = gr.Number(label="声道数 (可选)", value=0, precision=0)
                        audio_convert_btn = gr.Button("转换音频", variant="primary")
                    
                    with gr.Column():
                        audio_output = gr.File(label="转换结果")
                        audio_status = gr.Textbox(label="状态", interactive=False)
                
                audio_convert_btn.click(
                    self.convert_audio,
                    inputs=[audio_input, audio_format, audio_bitrate, audio_sample_rate, audio_channels],
                    outputs=[audio_output, audio_status]
                )
            
            with gr.Tab("音频信息"):
                with gr.Row():
                    with gr.Column():
                        audio_info_input = gr.File(label="选择音频文件", file_types=["audio"])
                        audio_info_btn = gr.Button("获取信息", variant="secondary")
                    
                    with gr.Column():
                        audio_info_output = gr.Markdown(label="音频信息")
                
                audio_info_btn.click(
                    self.get_audio_info,
                    inputs=[audio_info_input],
                    outputs=[audio_info_output]
                )
            
            with gr.Tab("片段提取"):
                with gr.Row():
                    with gr.Column():
                        segment_input = gr.File(label="选择音频文件", file_types=["audio"])
                        segment_start = gr.Number(label="开始时间 (秒)", value=0)
                        segment_duration = gr.Number(label="持续时间 (秒)", value=30)
                        segment_format = gr.Dropdown(
                            choices=self.audio_formats,
                            value="mp3",
                            label="输出格式"
                        )
                        segment_btn = gr.Button("提取片段", variant="primary")
                    
                    with gr.Column():
                        segment_output = gr.File(label="提取结果")
                        segment_status = gr.Textbox(label="状态", interactive=False)
                
                segment_btn.click(
                    self.extract_audio_segment,
                    inputs=[segment_input, segment_start, segment_duration, segment_format],
                    outputs=[segment_output, segment_status]
                )
        
        # 文档转换界面
        with gr.Blocks(title="AnythingConvert - 文档转换") as document_interface:
            gr.Markdown("# 📄 文档转换")
            
            with gr.Tab("格式转换"):
                with gr.Row():
                    with gr.Column():
                        doc_input = gr.File(label="选择文档文件")
                        
                        with gr.Row():
                            doc_input_format = gr.Dropdown(
                                choices=['auto'] + self.input_formats,
                                value="auto",
                                label="输入格式 (auto=自动检测)"
                            )
                            doc_output_format = gr.Dropdown(
                                choices=self.output_formats,
                                value="pdf",
                                label="输出格式"
                            )
                        
                        with gr.Accordion("高级选项", open=False):
                            doc_pdf_engine = gr.Dropdown(
                                choices=['xelatex', 'pdflatex', 'lualatex', 'context', 'wkhtmltopdf'],
                                value="xelatex",
                                label="PDF引擎 (仅PDF输出)"
                            )
                            doc_extra_args = gr.Textbox(
                                label="额外参数",
                                placeholder="如: --toc --number-sections",
                                info="Pandoc额外参数，用空格分隔"
                            )
                        
                        doc_convert_btn = gr.Button("转换文档", variant="primary")
                    
                    with gr.Column():
                        doc_output = gr.File(label="转换结果")
                        doc_status = gr.Textbox(label="状态", interactive=False)
                        
                        # 格式说明
                        gr.Markdown("""
                        ### 📋 支持的格式类别
                        - **常用格式**: PDF, DOCX, DOC, TXT, RTF, ODT
                        - **标记语言**: Markdown, reStructuredText, AsciiDoc, Org-mode
                        - **HTML/Web**: HTML, HTML5, XHTML
                        - **电子书**: EPUB, EPUB3, FB2
                        - **TeX/LaTeX**: LaTeX, TeX, ConTeXt
                        - **幻灯片**: PowerPoint, Beamer, reveal.js, Slidy
                        - **Wiki格式**: MediaWiki, DokuWiki, Jira, Creole
                        - **数据格式**: CSV, TSV, JSON, XML
                        - **参考文献**: BibTeX, BibLaTeX, CSL JSON, RIS
                        """)
                
                doc_convert_btn.click(
                    self.convert_document,
                    inputs=[doc_input, doc_output_format, doc_input_format, doc_pdf_engine, doc_extra_args],
                    outputs=[doc_output, doc_status]
                )
            
            with gr.Tab("快速转换"):
                gr.Markdown("### 🚀 一键转换为常用格式")
                
                with gr.Row():
                    with gr.Column():
                        quick_input = gr.File(label="选择文档文件")
                        
                        with gr.Row():
                            markdown_btn = gr.Button("转为Markdown", variant="secondary")
                            html_btn = gr.Button("转为HTML", variant="secondary")
                        
                        with gr.Row():
                            slides_format = gr.Dropdown(
                                choices=['revealjs', 'slidy', 'beamer', 'pptx'],
                                value="revealjs",
                                label="幻灯片格式"
                            )
                            slides_btn = gr.Button("转为幻灯片", variant="secondary")
                        
                        html_standalone = gr.Checkbox(
                            label="HTML自包含 (包含CSS/JS)",
                            value=True
                        )
                    
                    with gr.Column():
                        quick_output = gr.File(label="转换结果")
                        quick_status = gr.Textbox(label="状态", interactive=False)
                
                # 绑定快速转换按钮
                markdown_btn.click(
                    self.convert_to_markdown,
                    inputs=[quick_input],
                    outputs=[quick_output, quick_status]
                )
                
                html_btn.click(
                    self.convert_to_html,
                    inputs=[quick_input, html_standalone],
                    outputs=[quick_output, quick_status]
                )
                
                slides_btn.click(
                    self.convert_to_slides,
                    inputs=[quick_input, slides_format],
                    outputs=[quick_output, quick_status]
                )
            
            with gr.Tab("文档信息"):
                with gr.Row():
                    with gr.Column():
                        doc_info_input = gr.File(label="选择文档文件")
                        doc_info_btn = gr.Button("获取信息", variant="secondary")
                    
                    with gr.Column():
                        doc_info_output = gr.Markdown(label="文档信息")
                
                doc_info_btn.click(
                    self.get_document_info,
                    inputs=[doc_info_input],
                    outputs=[doc_info_output]
                )
            
            with gr.Tab("格式支持"):
                gr.Markdown("### 📚 支持的文档格式详情")
                
                # 创建格式支持表格
                format_info_md = "| 类别 | 支持的格式 |\n|------|------------|\n"
                for category, formats in self.document_formats.items():
                    format_list = ", ".join([f"`{fmt}`" for fmt in formats])
                    format_info_md += f"| **{category}** | {format_list} |\n"
                
                format_info_md += f"""

### 🔄 格式转换说明
- **输入格式**: {len(self.input_formats)} 种
- **输出格式**: {len(self.output_formats)} 种  
- **双向转换**: {len(self.bidirectional_formats)} 种

### ⚙️ 高级功能
- **自动格式检测**: 根据文件扩展名自动识别输入格式
- **PDF引擎选择**: 支持多种PDF生成引擎 (XeLaTeX, PDFLaTeX, LuaLaTeX等)
- **自定义参数**: 支持传递Pandoc的所有命令行参数
- **批量转换**: 支持批量处理多个文件
- **编码转换**: 支持文本文件编码转换

### 📖 使用提示
1. **PDF输出**: 推荐使用XeLaTeX引擎，支持中文字体
2. **幻灯片**: reveal.js适合网页展示，Beamer适合学术演示
3. **电子书**: EPUB格式兼容性最好
4. **Wiki格式**: 可在不同Wiki系统间转换
5. **参考文献**: 支持多种引用格式转换
                """
                
                gr.Markdown(format_info_md)
        
        # 主界面 - 使用TabbedInterface组合所有功能
        main_interface = gr.TabbedInterface(
            [image_interface, video_interface, audio_interface, document_interface],
            ["🖼️ 图片转换", "🎬 视频转换", "🎵 音频转换", "📄 文档转换"],
            title="AnythingConvert - 万能文件转换工具"
        )
        
        return main_interface

def main():
    """启动Gradio应用"""
    app = AnythingConvertApp()
    interface = app.create_interface()
    
    # 启动界面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()