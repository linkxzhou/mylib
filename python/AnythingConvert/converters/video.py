"""
视频转换器
支持格式：MP4, AVI, MOV, WMV, FLV, MKV, WEBM
"""

import ffmpeg
from pathlib import Path
from typing import Optional, Dict, Any
from utils.mylog import logger

class VideoConverter:
    """视频转换器类"""
    
    SUPPORTED_FORMATS = {
        'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm'
    }
    
    def convert(
        self,
        input_path: Path,
        output_path: Path,
        codec: Optional[str] = None,
        bitrate: Optional[str] = None,
        resolution: Optional[str] = None,
        fps: Optional[int] = None,
        audio_codec: Optional[str] = None,
        audio_bitrate: Optional[str] = None
    ) -> bool:
        """
        转换视频格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            codec: 视频编码器 (如 'libx264', 'libx265')
            bitrate: 视频比特率 (如 '1M', '2000k')
            resolution: 分辨率 (如 '1920x1080', '1280x720')
            fps: 帧率
            audio_codec: 音频编码器 (如 'aac', 'mp3')
            audio_bitrate: 音频比特率 (如 '128k', '192k')
            
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
            
            # 构建ffmpeg输入流
            input_stream = ffmpeg.input(str(input_path))
            
            # 构建输出参数
            output_kwargs = {}
            
            # 视频参数
            if codec:
                output_kwargs['vcodec'] = codec
            elif output_ext == 'mp4':
                output_kwargs['vcodec'] = 'libx264'
            elif output_ext == 'webm':
                output_kwargs['vcodec'] = 'libvpx-vp9'
            elif output_ext == 'mkv':
                output_kwargs['vcodec'] = 'libx264'
            
            if bitrate:
                output_kwargs['video_bitrate'] = bitrate
            
            if resolution:
                if 'x' in resolution:
                    width, height = resolution.split('x')
                    output_kwargs['s'] = f"{width}x{height}"
            
            if fps:
                output_kwargs['r'] = fps
            
            # 音频参数
            if audio_codec:
                output_kwargs['acodec'] = audio_codec
            elif output_ext in ['mp4', 'mkv']:
                output_kwargs['acodec'] = 'aac'
            elif output_ext == 'webm':
                output_kwargs['acodec'] = 'libvorbis'
            elif output_ext == 'avi':
                output_kwargs['acodec'] = 'mp3'
            
            if audio_bitrate:
                output_kwargs['audio_bitrate'] = audio_bitrate
            
            # 执行转换
            output_stream = ffmpeg.output(input_stream, str(output_path), **output_kwargs)
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            logger.info(f"成功转换: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"转换失败: {input_path} -> {output_path}, 错误: {str(e)}")
            return False
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """获取视频信息"""
        try:
            probe = ffmpeg.probe(str(video_path))
            video_info = {}
            
            # 获取视频流信息
            video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
            if video_streams:
                video_stream = video_streams[0]
                video_info.update({
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream.get('r_frame_rate') else 0,
                    'duration': float(video_stream.get('duration', 0)),
                    'codec': video_stream.get('codec_name'),
                    'bitrate': video_stream.get('bit_rate')
                })
            
            # 获取音频流信息
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            if audio_streams:
                audio_stream = audio_streams[0]
                video_info.update({
                    'audio_codec': audio_stream.get('codec_name'),
                    'audio_bitrate': audio_stream.get('bit_rate'),
                    'sample_rate': audio_stream.get('sample_rate'),
                    'channels': audio_stream.get('channels')
                })
            
            return video_info
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {video_path}, 错误: {str(e)}")
            return {}
    
    def extract_audio(self, input_path: Path, output_path: Path, audio_format: str = 'mp3') -> bool:
        """从视频中提取音频"""
        try:
            input_stream = ffmpeg.input(str(input_path))
            audio_codec = 'mp3' if audio_format == 'mp3' else 'copy'
            output_stream = ffmpeg.output(input_stream, str(output_path), acodec=audio_codec, vn=None)
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            logger.info(f"成功提取音频: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"提取音频失败: {input_path} -> {output_path}, 错误: {str(e)}")
            return False