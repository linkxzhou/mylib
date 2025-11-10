#!/usr/bin/env python3
"""
文档转换模块 - 基于Pandoc
支持多种文档格式之间的转换
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
from utils.mylog import logger

# 导入Pandoc库
try:
    import pypandoc
    PANDOC_SUPPORT = True
except ImportError:
    PANDOC_SUPPORT = False

# 导入其他文档处理库作为备用
try:
    from docx import Document
    from docx2txt import process as docx2txt_process
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import PyPDF2
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

class DocumentConverter:
    """基于Pandoc的文档转换器类"""
    
    # Pandoc支持的格式映射 (基于官方文档)
    PANDOC_FORMATS = {
        # 轻量级标记格式
        'md': 'markdown',
        'markdown': 'markdown',
        'rst': 'rst',
        'asciidoc': 'asciidoc',
        'org': 'org',
        'muse': 'muse',
        'textile': 'textile',
        'markua': 'markua',
        't2t': 'txt2tags',
        'djot': 'djot',
        
        # HTML格式
        'html': 'html',
        'htm': 'html',
        'xhtml': 'html',
        'html5': 'html5',
        
        # 电子书格式
        'epub': 'epub',
        'epub3': 'epub3',
        'fb2': 'fb2',
        
        # 文档格式
        'texinfo': 'texinfo',
        'pod': 'pod',
        'haddock': 'haddock',
        'vimdoc': 'vimdoc',
        
        # Roff格式
        'man': 'man',
        'ms': 'ms',
        'mdoc': 'mdoc',
        
        # TeX格式
        'latex': 'latex',
        'tex': 'latex',
        'context': 'context',
        
        # XML格式
        'docbook': 'docbook',
        'docbook4': 'docbook4',
        'docbook5': 'docbook5',
        'jats': 'jats',
        'bits': 'bits',
        'tei': 'tei',
        'opendocument': 'opendocument',
        
        # 大纲格式
        'opml': 'opml',
        
        # 参考文献格式
        'bibtex': 'bibtex',
        'bib': 'bibtex',
        'biblatex': 'biblatex',
        'csljson': 'csljson',
        'cslyaml': 'cslyaml',
        'ris': 'ris',
        'endnotexml': 'endnotexml',
        
        # 文字处理器格式
        'docx': 'docx',
        'doc': 'doc',
        'rtf': 'rtf',
        'odt': 'odt',
        
        # 交互式笔记本格式
        'ipynb': 'ipynb',
        'jupyter': 'ipynb',
        
        # 页面布局格式
        'icml': 'icml',
        'typst': 'typst',
        
        # Wiki标记格式
        'mediawiki': 'mediawiki',
        'dokuwiki': 'dokuwiki',
        'tikiwiki': 'tikiwiki',
        'twiki': 'twiki',
        'vimwiki': 'vimwiki',
        'xwiki': 'xwiki',
        'zimwiki': 'zimwiki',
        'jira': 'jira',
        'creole': 'creole',
        
        # 幻灯片格式
        'beamer': 'beamer',
        'pptx': 'pptx',
        'slidy': 'slidy',
        'revealjs': 'revealjs',
        'slideous': 'slideous',
        's5': 's5',
        'dzslides': 'dzslides',
        
        # 数据格式
        'csv': 'csv',
        'tsv': 'tsv',
        
        # 终端输出
        'ansi': 'ansi',
        
        # 序列化格式
        'native': 'native',
        'json': 'json',
        'xml': 'xml',
        
        # 纯文本
        'txt': 'plain',
        'plain': 'plain',
        
        # PDF (需要额外引擎)
        'pdf': 'pdf'
    }
    
    # 支持的格式集合
    SUPPORTED_FORMATS = set(PANDOC_FORMATS.keys())
    
    # 输入输出格式分类
    INPUT_ONLY_FORMATS = {
        't2t', 'pod', 'bits', 'ris', 'endnotexml', 'tikiwiki', 'twiki', 
        'vimwiki', 'creole', 'csv', 'tsv', 'mdoc'
    }
    
    OUTPUT_ONLY_FORMATS = {
        'asciidoc', 'markua', 'texinfo', 'vimdoc', 'ms', 'context', 'tei',
        'opendocument', 'icml', 'xwiki', 'zimwiki', 'beamer', 'pptx', 'slidy',
        'revealjs', 'slideous', 's5', 'dzslides', 'ansi', 'pdf'
    }
    
    # 双向转换格式
    BIDIRECTIONAL_FORMATS = SUPPORTED_FORMATS - INPUT_ONLY_FORMATS - OUTPUT_ONLY_FORMATS
    
    def __init__(self):
        """初始化文档转换器"""
        self._check_dependencies()
        self._check_pandoc_installation()
    
    def _check_dependencies(self):
        """检查依赖库是否可用"""
        if not PANDOC_SUPPORT:
            logger.warning("Pandoc Python库不可用，请安装: pip install pypandoc")
        
        if not DOCX_SUPPORT:
            logger.warning("DOCX支持不可用，请安装: pip install python-docx docx2txt")
        
        if not PDF_SUPPORT:
            logger.warning("PDF支持不可用，请安装: pip install PyPDF2 reportlab")
    
    def _check_pandoc_installation(self):
        """检查Pandoc是否已安装"""
        try:
            if PANDOC_SUPPORT:
                # 检查pypandoc是否能找到pandoc
                pypandoc.get_pandoc_version()
                logger.info(f"Pandoc版本: {pypandoc.get_pandoc_version()}")
            else:
                # 尝试直接调用pandoc命令
                result = subprocess.run(['pandoc', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Pandoc命令行工具可用")
                else:
                    logger.warning("Pandoc未安装或不在PATH中")
        except Exception as e:
            logger.warning(f"Pandoc检查失败: {e}")
            logger.info("请安装Pandoc: https://pandoc.org/installing.html")
    
    def convert(self, input_path: str, output_path: str, 
                input_format: Optional[str] = None, 
                output_format: Optional[str] = None,
                extra_args: Optional[List[str]] = None,
                pdf_engine: str = 'xelatex') -> bool:
        """
        转换文档格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            input_format: 输入格式，如果为None则自动检测
            output_format: 输出格式，如果为None则从输出文件扩展名推断
            extra_args: 额外的Pandoc参数
            pdf_engine: PDF引擎（当输出为PDF时使用）
            
        Returns:
            bool: 转换是否成功
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # 验证输入文件
            if not input_path.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 获取文件格式
            if input_format is None:
                input_format = input_path.suffix.lower().lstrip('.')
            if output_format is None:
                output_format = output_path.suffix.lower().lstrip('.')
            
            # 验证格式支持
            if input_format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的输入格式: {input_format}")
            
            if output_format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的输出格式: {output_format}")
            
            # 验证格式兼容性
            # PDF 是特殊情况：Pandoc 不支持从 PDF 读取，但我们有备用方法
            if input_format in self.OUTPUT_ONLY_FORMATS and input_format != 'pdf':
                raise ValueError(f"格式 '{input_format}' 只能作为输出格式")
            
            if output_format in self.INPUT_ONLY_FORMATS:
                raise ValueError(f"格式 '{output_format}' 只能作为输入格式")
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 执行转换
            # 如果输入格式是PDF，使用备用转换方法
            if input_format == 'pdf':
                success = self._convert_fallback(
                    input_path, output_path, input_format, output_format
                )
            else:
                success = self._convert_with_pandoc(
                    input_path, output_path, input_format, output_format, extra_args, pdf_engine
                )
            
            if success:
                logger.info(f"文档转换成功: {input_path} -> {output_path}")
                return True
            else:
                raise RuntimeError("文档转换失败")
                
        except Exception as e:
            logger.error(f"文档转换失败: {e}")
            raise
    
    def _convert_with_pandoc(self, input_path: Path, output_path: Path,
                           input_format: str, output_format: str,
                           extra_args: Optional[List[str]] = None,
                           pdf_engine: str = 'xelatex') -> bool:
        """使用Pandoc执行文档转换"""
        try:
            # 获取Pandoc格式名称
            pandoc_input_format = self.PANDOC_FORMATS.get(input_format, input_format)
            pandoc_output_format = self.PANDOC_FORMATS.get(output_format, output_format)
            
            if PANDOC_SUPPORT:
                # 使用pypandoc库
                return self._convert_with_pypandoc(
                    input_path, output_path, 
                    pandoc_input_format, pandoc_output_format, 
                    extra_args, pdf_engine
                )
            else:
                # 使用pandoc命令行
                return self._convert_with_pandoc_cli(
                    input_path, output_path,
                    pandoc_input_format, pandoc_output_format,
                    extra_args, pdf_engine
                )
                
        except Exception as e:
            logger.error(f"Pandoc转换失败: {e}")
            # 如果Pandoc转换失败，尝试使用备用方法
            return self._convert_fallback(input_path, output_path, input_format, output_format)
    
    def _convert_with_pypandoc(self, input_path: Path, output_path: Path,
                             input_format: str, output_format: str,
                             extra_args: Optional[List[str]] = None,
                             pdf_engine: str = 'xelatex') -> bool:
        """使用pypandoc库进行转换"""
        try:
            # 准备额外参数
            pandoc_args = extra_args or []
            
            # 特殊处理某些格式
            if output_format == 'pdf':
                # PDF转换需要LaTeX引擎
                pandoc_args.extend([f'--pdf-engine={pdf_engine}'])
            
            elif output_format in ['beamer', 'slidy', 'revealjs', 'slideous', 's5', 'dzslides']:
                # 幻灯片格式的特殊处理
                if output_format == 'revealjs':
                    pandoc_args.extend(['--self-contained'])
                elif output_format == 'beamer':
                    pandoc_args.extend(['--slide-level=2'])
            
            elif output_format == 'epub':
                # EPUB格式优化
                pandoc_args.extend(['--epub-cover-image=cover.jpg']) if Path('cover.jpg').exists() else None
            
            elif output_format in ['html', 'html5']:
                # HTML格式优化
                pandoc_args.extend(['--self-contained', '--mathjax'])
            
            # 执行转换
            pypandoc.convert_file(
                str(input_path),
                output_format,
                outputfile=str(output_path),
                format=input_format,
                extra_args=pandoc_args
            )
            
            return output_path.exists()
            
        except Exception as e:
            logger.error(f"pypandoc转换失败: {e}")
            raise
    
    def _convert_with_pandoc_cli(self, input_path: Path, output_path: Path,
                               input_format: str, output_format: str,
                               extra_args: Optional[List[str]] = None,
                               pdf_engine: str = 'xelatex') -> bool:
        """使用pandoc命令行进行转换"""
        try:
            # 构建pandoc命令
            cmd = [
                'pandoc',
                str(input_path),
                '-f', input_format,
                '-t', output_format,
                '-o', str(output_path)
            ]
            
            # 添加额外参数
            if extra_args:
                cmd.extend(extra_args)
            
            # 特殊处理不同格式
            if output_format == 'pdf':
                cmd.extend([f'--pdf-engine={pdf_engine}'])
            
            elif output_format in ['beamer', 'slidy', 'revealjs', 'slideous', 's5', 'dzslides']:
                if output_format == 'revealjs':
                    cmd.extend(['--self-contained'])
                elif output_format == 'beamer':
                    cmd.extend(['--slide-level=2'])
            
            elif output_format == 'epub':
                if Path('cover.jpg').exists():
                    cmd.extend(['--epub-cover-image=cover.jpg'])
            
            elif output_format in ['html', 'html5']:
                cmd.extend(['--self-contained', '--mathjax'])
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return output_path.exists()
            else:
                logger.error(f"Pandoc命令执行失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Pandoc CLI转换失败: {e}")
            raise
    
    def _convert_fallback(self, input_path: Path, output_path: Path,
                         input_format: str, output_format: str) -> bool:
        """备用转换方法（使用原有的库）"""
        try:
            logger.info("使用备用转换方法")
            
            # 提取文本内容
            text_content = self._extract_text_fallback(input_path, input_format)
            
            if not text_content:
                raise ValueError("无法从输入文件中提取文本内容")
            
            # 生成输出文档
            return self._generate_document_fallback(text_content, output_path, output_format)
            
        except Exception as e:
            logger.error(f"备用转换方法失败: {e}")
            return False
    
    def _extract_text_fallback(self, file_path: Path, file_format: str) -> str:
        """备用文本提取方法"""
        try:
            if file_format == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_format == 'docx' and DOCX_SUPPORT:
                return docx2txt_process(str(file_path))
            
            elif file_format == 'pdf' and PDF_SUPPORT:
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            else:
                raise ValueError(f"备用方法不支持的格式: {file_format}")
                
        except Exception as e:
            logger.error(f"备用文本提取失败: {e}")
            return ""
    
    def _generate_document_fallback(self, text_content: str, output_path: Path, 
                                  output_format: str) -> bool:
        """备用文档生成方法"""
        try:
            if output_format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                return True
            
            elif output_format == 'docx' and DOCX_SUPPORT:
                doc = Document()
                paragraphs = text_content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        doc.add_paragraph(para.strip())
                doc.save(str(output_path))
                return True
            
            else:
                raise ValueError(f"备用方法不支持的输出格式: {output_format}")
                
        except Exception as e:
            logger.error(f"备用文档生成失败: {e}")
            return False
    
    def convert_text_encoding(self, input_path: str, output_path: str,
                            from_encoding: str = 'auto', 
                            to_encoding: str = 'utf-8') -> bool:
        """
        转换文本文件编码
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            from_encoding: 源编码 ('auto' 为自动检测)
            to_encoding: 目标编码
            
        Returns:
            bool: 转换是否成功
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 自动检测编码
            if from_encoding == 'auto':
                import chardet
                with open(input_path, 'rb') as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    from_encoding = result['encoding']
                    logger.info(f"检测到编码: {from_encoding} (置信度: {result['confidence']:.2f})")
            
            # 读取并转换编码
            with open(input_path, 'r', encoding=from_encoding) as f:
                content = f.read()
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入新编码
            with open(output_path, 'w', encoding=to_encoding) as f:
                f.write(content)
            
            logger.info(f"编码转换成功: {from_encoding} -> {to_encoding}")
            return True
            
        except Exception as e:
            logger.error(f"编码转换失败: {e}")
            return False
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文档信息
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            Dict: 文档信息
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            file_format = file_path.suffix.lower().lstrip('.')
            file_size = file_path.stat().st_size
            
            # 基本信息
            info = {
                'name': file_path.name,
                'format': file_format,
                'size': file_size,
                'size_mb': file_size / (1024 * 1024),
                'path': str(file_path)
            }
            
            # 尝试提取文本内容以获取更多信息
            try:
                if PANDOC_SUPPORT and file_format in self.PANDOC_FORMATS:
                    # 使用Pandoc转换为纯文本获取内容
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    try:
                        pypandoc.convert_file(
                            str(file_path),
                            'plain',
                            outputfile=tmp_path,
                            format=self.PANDOC_FORMATS[file_format]
                        )
                        
                        with open(tmp_path, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        
                        # 添加文本统计信息
                        info.update({
                            'text_length': len(text_content),
                            'word_count': len(text_content.split()),
                            'line_count': len(text_content.split('\n')),
                            'paragraph_count': len([p for p in text_content.split('\n\n') if p.strip()])
                        })
                        
                    finally:
                        # 清理临时文件
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                else:
                    # 使用备用方法
                    text_content = self._extract_text_fallback(file_path, file_format)
                    if text_content:
                        info.update({
                            'text_length': len(text_content),
                            'word_count': len(text_content.split()),
                            'line_count': len(text_content.split('\n')),
                            'paragraph_count': len([p for p in text_content.split('\n\n') if p.strip()])
                        })
                    
            except Exception as e:
                logger.warning(f"无法提取文本内容: {e}")
                info.update({
                    'text_length': 0,
                    'word_count': 0,
                    'line_count': 0,
                    'paragraph_count': 0
                })
            
            return info
            
        except Exception as e:
            logger.error(f"获取文档信息失败: {e}")
            raise
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        获取支持的格式列表
        
        Returns:
            Dict: 支持的输入和输出格式
        """
        # PDF 特殊处理：虽然在 OUTPUT_ONLY_FORMATS 中，但通过备用方法支持输入
        input_formats = self.SUPPORTED_FORMATS - self.OUTPUT_ONLY_FORMATS
        if PDF_SUPPORT:
            input_formats = input_formats | {'pdf'}  # 添加 PDF 作为输入格式
        
        return {
            'all_formats': list(self.SUPPORTED_FORMATS),
            'input_formats': list(input_formats),
            'output_formats': list(self.SUPPORTED_FORMATS - self.INPUT_ONLY_FORMATS),
            'bidirectional_formats': list(self.BIDIRECTIONAL_FORMATS),
            'input_only_formats': list(self.INPUT_ONLY_FORMATS),
            'output_only_formats': list(self.OUTPUT_ONLY_FORMATS),
            'pandoc_available': PANDOC_SUPPORT,
            'pandoc_formats': self.PANDOC_FORMATS,
            'pdf_input_support': PDF_SUPPORT  # 标识 PDF 输入支持状态
        }
    
    def convert_to_markdown(self, input_path: str, output_path: str = None,
                          input_format: str = None) -> str:
        """
        转换任意格式到Markdown
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径 (可选)
            input_format: 输入格式 (可选，自动检测)
            
        Returns:
            str: Markdown内容
        """
        if not output_path:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}.md"
        
        self.convert(input_path, output_path, input_format, 'md')
        
        with open(output_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def convert_to_html(self, input_path: str, output_path: str = None,
                       input_format: str = None, standalone: bool = True) -> str:
        """
        转换任意格式到HTML
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径 (可选)
            input_format: 输入格式 (可选，自动检测)
            standalone: 是否生成独立的HTML文件
            
        Returns:
            str: HTML内容
        """
        if not output_path:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}.html"
        
        extra_args = ['--standalone'] if standalone else []
        self.convert(input_path, output_path, input_format, 'html', extra_args)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def convert_to_pdf(self, input_path: str, output_path: str = None,
                      input_format: str = None, pdf_engine: str = 'xelatex') -> bool:
        """
        转换任意格式到PDF
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径 (可选)
            input_format: 输入格式 (可选，自动检测)
            pdf_engine: PDF引擎
            
        Returns:
            bool: 转换是否成功
        """
        if not output_path:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}.pdf"
        
        return self.convert(input_path, output_path, input_format, 'pdf', 
                          pdf_engine=pdf_engine)
    
    def convert_to_slides(self, input_path: str, output_path: str = None,
                         slide_format: str = 'revealjs', input_format: str = None) -> bool:
        """
        转换任意格式到幻灯片
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径 (可选)
            slide_format: 幻灯片格式 (revealjs, beamer, slidy等)
            input_format: 输入格式 (可选，自动检测)
            
        Returns:
            bool: 转换是否成功
        """
        if slide_format not in ['revealjs', 'beamer', 'slidy', 'slideous', 's5', 'dzslides', 'pptx']:
            raise ValueError(f"不支持的幻灯片格式: {slide_format}")
        
        if not output_path:
            input_file = Path(input_path)
            ext = 'html' if slide_format != 'pptx' and slide_format != 'beamer' else slide_format
            if slide_format == 'beamer':
                ext = 'pdf'
            output_path = input_file.parent / f"{input_file.stem}_slides.{ext}"
        
        return self.convert(input_path, output_path, input_format, slide_format)
    
    def convert_bibliography(self, input_path: str, output_path: str = None,
                           input_format: str = None, output_format: str = 'bibtex') -> bool:
        """
        转换参考文献格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径 (可选)
            input_format: 输入格式 (可选，自动检测)
            output_format: 输出格式 (bibtex, biblatex, csljson等)
            
        Returns:
            bool: 转换是否成功
        """
        bib_formats = ['bibtex', 'biblatex', 'csljson', 'cslyaml', 'ris', 'endnotexml']
        if output_format not in bib_formats:
            raise ValueError(f"不支持的参考文献格式: {output_format}")
        
        if not output_path:
            input_file = Path(input_path)
            ext = 'bib' if output_format in ['bibtex', 'biblatex'] else output_format
            output_path = input_file.parent / f"{input_file.stem}.{ext}"
        
        return self.convert(input_path, output_path, input_format, output_format)