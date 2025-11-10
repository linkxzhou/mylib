#!/usr/bin/env python3
"""
音频转换模块
支持多种音频格式之间的转换
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import ffmpeg
from utils.mylog import logger


class AudioConverter:
    """音频转换器类"""
    
    SUPPORTED_FORMATS = {
        'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma'
    }
    
    def convert(self, input_path: str, output_path: str, 
                bitrate: Optional[str] = None,
                sample_rate: Optional[int] = None,
                channels: Optional[int] = None) -> bool:
        """
        转换音频格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            bitrate: 比特率 (如: '128k', '320k')
            sample_rate: 采样率 (如: 44100, 48000)
            channels: 声道数 (1=单声道, 2=立体声)
            
        Returns:
            bool: 转换是否成功
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # 验证输入文件
            if not input_path.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 验证格式支持
            input_format = input_path.suffix.lower().lstrip('.')
            output_format = output_path.suffix.lower().lstrip('.')
            
            if input_format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的输入格式: {input_format}")
            
            if output_format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的输出格式: {output_format}")
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建ffmpeg命令
            stream = ffmpeg.input(str(input_path))
            
            # 设置输出参数
            output_kwargs = {}
            
            if bitrate:
                output_kwargs['audio_bitrate'] = bitrate
            
            if sample_rate:
                output_kwargs['ar'] = sample_rate
                
            if channels:
                output_kwargs['ac'] = channels
            
            # 根据输出格式设置特定参数
            if output_format == 'mp3':
                output_kwargs['acodec'] = 'libmp3lame'
            elif output_format == 'aac':
                output_kwargs['acodec'] = 'aac'
            elif output_format == 'flac':
                output_kwargs['acodec'] = 'flac'
            elif output_format == 'ogg':
                output_kwargs['acodec'] = 'libvorbis'
            elif output_format == 'wav':
                output_kwargs['acodec'] = 'pcm_s16le'
            
            # 执行转换
            stream = ffmpeg.output(stream, str(output_path), **output_kwargs)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"音频转换成功: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"音频转换失败: {e}")
            raise
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            Dict: 音频信息
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            probe = ffmpeg.probe(str(file_path))
            audio_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'audio'), None)
            
            if not audio_stream:
                raise ValueError("文件中没有找到音频流")
            
            info = {
                'duration': float(probe['format'].get('duration', 0)),
                'bitrate': int(probe['format'].get('bit_rate', 0)),
                'format': probe['format']['format_name'],
                'codec': audio_stream.get('codec_name', 'unknown'),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'channel_layout': audio_stream.get('channel_layout', 'unknown'),
                'size': file_path.stat().st_size
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取音频信息失败: {e}")
            raise
    
    def extract_segment(self, input_path: str, output_path: str,
                       start_time: float, duration: float) -> bool:
        """
        提取音频片段
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            start_time: 开始时间(秒)
            duration: 持续时间(秒)
            
        Returns:
            bool: 提取是否成功
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            if not input_path.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 提取片段
            stream = ffmpeg.input(str(input_path), ss=start_time, t=duration)
            stream = ffmpeg.output(stream, str(output_path))
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"音频片段提取成功: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"音频片段提取失败: {e}")
            raise