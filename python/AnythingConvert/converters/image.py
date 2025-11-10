"""
图片转换器
支持格式：JPG, PNG, GIF, BMP, TIFF, WEBP, SVG, ICO
"""

from PIL import Image
from pathlib import Path
from typing import Optional
from utils.mylog import logger

# SVG处理需要额外的库
try:
    from cairosvg import svg2png
    SVG_SUPPORT = True
except ImportError:
    SVG_SUPPORT = False

class ImageConverter:
    """图片转换器类"""
    
    SUPPORTED_FORMATS = {
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg', 'ico'
    }
    
    def convert(
        self,
        input_path: Path,
        output_path: Path,
        quality: int = 95,
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_aspect_ratio: bool = True
    ) -> bool:
        """
        转换图片格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            quality: 输出质量 (1-100)
            width: 目标宽度
            height: 目标高度
            keep_aspect_ratio: 是否保持宽高比
            
        Returns:
            bool: 转换是否成功
        """
        try:
            # 验证输入文件
            if not input_path.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 验证格式支持
            input_ext = input_path.suffix.lower().lstrip('.')
            output_ext = output_path.suffix.lower().lstrip('.')
            
            if input_ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的输入格式: {input_ext}")
            
            if output_ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的输出格式: {output_ext}")
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 特殊处理SVG格式
            if input_ext == 'svg':
                if not SVG_SUPPORT:
                    raise ImportError("SVG转换需要安装cairosvg库: pip install cairosvg")
                return self._convert_svg(input_path, output_path, width, height, quality)
            
            # 打开图片
            with Image.open(input_path) as img:
                # 处理透明度
                if output_ext in ['jpg', 'jpeg'] and img.mode in ['RGBA', 'LA']:
                    # JPEG不支持透明度，转换为RGB
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                
                # 调整尺寸
                if width or height:
                    img = self._resize_image(img, width, height, keep_aspect_ratio)
                
                # 保存图片
                save_kwargs = {}
                if output_ext in ['jpg', 'jpeg']:
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif output_ext == 'png':
                    save_kwargs['optimize'] = True
                elif output_ext == 'webp':
                    save_kwargs['quality'] = quality
                    save_kwargs['method'] = 6
                
                img.save(output_path, **save_kwargs)
                
            logger.info(f"成功转换: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"转换失败: {input_path} -> {output_path}, 错误: {str(e)}")
            return False
    
    def _resize_image(
        self, 
        img: Image.Image, 
        width: Optional[int], 
        height: Optional[int], 
        keep_aspect_ratio: bool
    ) -> Image.Image:
        """调整图片尺寸"""
        original_width, original_height = img.size
        
        if keep_aspect_ratio:
            if width and height:
                # 计算缩放比例，保持宽高比
                ratio = min(width / original_width, height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
            elif width:
                ratio = width / original_width
                new_width = width
                new_height = int(original_height * ratio)
            elif height:
                ratio = height / original_height
                new_width = int(original_width * ratio)
                new_height = height
            else:
                return img
        else:
            new_width = width or original_width
            new_height = height or original_height
        
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def compress(
        self,
        input_path: Path,
        output_path: Path,
        quality: int = 85,
        max_size_kb: Optional[int] = None
    ) -> bool:
        """
        压缩图片
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            quality: 压缩质量
            max_size_kb: 最大文件大小(KB)
            
        Returns:
            bool: 压缩是否成功
        """
        try:
            with Image.open(input_path) as img:
                # 如果是RGBA模式且输出为JPEG，转换为RGB
                if img.mode == 'RGBA' and output_path.suffix.lower() in ['.jpg', '.jpeg']:
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                
                # 如果指定了最大文件大小，进行迭代压缩
                if max_size_kb:
                    current_quality = quality
                    while current_quality > 10:
                        img.save(output_path, quality=current_quality, optimize=True)
                        
                        # 检查文件大小
                        if output_path.stat().st_size <= max_size_kb * 1024:
                            break
                            
                        current_quality -= 5
                else:
                    img.save(output_path, quality=quality, optimize=True)
                
            return True
            
        except Exception as e:
            logger.error(f"压缩失败: {input_path} -> {output_path}, 错误: {str(e)}")
            return False
    
    def _convert_svg(
        self,
        input_path: Path,
        output_path: Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
        quality: int = 95
    ) -> bool:
        """
        转换SVG格式图片
        
        Args:
            input_path: SVG输入文件路径
            output_path: 输出文件路径
            width: 目标宽度
            height: 目标高度
            quality: 输出质量
            
        Returns:
            bool: 转换是否成功
        """
        try:
            output_ext = output_path.suffix.lower().lstrip('.')
            
            # 读取SVG内容
            with open(input_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # 转换SVG到PNG
            png_data = svg2png(
                bytestring=svg_content.encode('utf-8'),
                output_width=width,
                output_height=height
            )
            
            # 如果输出格式是PNG，直接保存
            if output_ext == 'png':
                with open(output_path, 'wb') as f:
                    f.write(png_data)
            else:
                # 否则通过PIL转换到其他格式
                from io import BytesIO
                png_image = Image.open(BytesIO(png_data))
                
                # 处理透明度
                if output_ext in ['jpg', 'jpeg'] and png_image.mode == 'RGBA':
                    background = Image.new('RGB', png_image.size, (255, 255, 255))
                    background.paste(png_image, mask=png_image.split()[-1])
                    png_image = background
                
                # 保存为目标格式
                save_kwargs = {}
                if output_ext in ['jpg', 'jpeg']:
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif output_ext == 'webp':
                    save_kwargs['quality'] = quality
                    save_kwargs['method'] = 6
                
                png_image.save(output_path, **save_kwargs)
            
            logger.info(f"成功转换SVG: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"SVG转换失败: {input_path} -> {output_path}, 错误: {str(e)}")
            return False