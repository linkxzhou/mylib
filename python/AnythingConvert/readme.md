# AnythingConvert

一个功能强大的多媒体文件转换工具，支持图片、视频、音频和文档格式之间的转换。

## 📋 项目概述

### 🎯 项目目标
构建一个统一的文件转换平台，支持多种媒体格式的高质量转换，提供命令行和编程接口。

### 🏗️ 架构设计 (SDD)

#### 核心模块架构
```
AnythingConvert/
├── converters/          # 转换器核心模块
│   ├── image.py        # 图片转换器
│   ├── video.py        # 视频转换器  
│   ├── audio.py        # 音频转换器
│   └── document.py     # 文档转换器
├── cli/                # 命令行接口
│   ├── image_cli.py    # 图片CLI
│   ├── video_cli.py    # 视频CLI
│   ├── audio_cli.py    # 音频CLI
│   └── document_cli.py # 文档CLI
├── utils/              # 工具模块
│   └── file_utils.py   # 文件处理工具
└── main.py            # 主入口
```

## 🚀 功能特性

### 🖼️ 图片转换
- **支持格式**: JPG, PNG, GIF, BMP, TIFF, WEBP, SVG, ICO
- **核心功能**: 
  - 格式转换
  - 尺寸调整 
  - 质量压缩
  - 批量处理
- **特殊支持**: SVG矢量图转换
- **技术栈**: Pillow, OpenCV, CairoSVG

### 🎬 视频转换  
- **支持格式**: MP4, AVI, MOV, WMV, FLV, MKV, WEBM
- **核心功能**: 
  - 格式转换
  - 视频信息查看
  - 音频提取
  - 分辨率调整
  - 编码器选择
- **技术栈**: FFmpeg-python

### 🎵 音频转换
- **支持格式**: MP3, WAV, FLAC, AAC, OGG, M4A, WMA
- **核心功能**: 
  - 音频格式转换
  - 音频信息获取
  - 音频片段提取
  - 比特率调整
  - 采样率转换
- **技术栈**: FFmpeg-python

### 📄 文档转换
- **支持格式**: PDF, DOC, DOCX, TXT, RTF, ODT
- **核心功能**: 
  - 文档格式转换
  - 文本提取
  - 文档信息获取
  - 批量处理
- **技术栈**: PyPDF2, python-docx, reportlab, odfpy

### 🗜️ 图片压缩
- **支持格式**: JPG, PNG, GIF, BMP, TIFF, WEBP
- **核心功能**:
  - 质量压缩
  - 尺寸压缩
  - 批量压缩
  - 智能压缩算法

### 📝 文本转换
- **支持格式**: TXT, CSV, JSON
- **核心功能**:
  - 编码转换
  - 格式转换
  - 数据结构转换

## 📋 开发任务分解 (SDD)

### 🎯 Phase 1: 核心转换功能完善
#### Task 1.1: 图片转换增强 ✅
- [x] 基础格式转换 (JPG,PNG,GIF,BMP,TIFF,WEBP,SVG,ICO)
- [x] 图片压缩功能
- [x] 尺寸调整
- [x] 批量处理
- [ ] 高级压缩算法优化
- [ ] 水印添加功能

#### Task 1.2: 视频转换完善 ✅  
- [x] 基础格式转换 (MP4,AVI,MOV,WMV,FLV,MKV,WEBM)
- [x] 音频提取
- [x] 视频信息获取
- [ ] 视频剪辑功能
- [ ] 字幕处理
- [ ] 视频合并

#### Task 1.3: 音频转换完善 ✅
- [x] 基础格式转换 (MP3,WAV,FLAC,AAC,OGG,M4A,WMA)
- [x] 音频片段提取
- [x] 音频信息获取
- [ ] 音频混合
- [ ] 音效处理
- [ ] 音频标准化

#### Task 1.4: 文档转换完善 ✅
- [x] 基础格式转换 (PDF,DOC,DOCX,TXT,RTF,ODT)
- [x] 文本提取
- [ ] 文档合并
- [ ] 文档分割
- [ ] 格式保持优化

### 🎯 Phase 2: 新功能开发
#### Task 2.1: 文本转换模块 🆕
- [ ] TXT编码转换 (UTF-8, GBK, ASCII)
- [ ] CSV格式处理 (分隔符转换, 编码转换)
- [ ] JSON格式转换 (CSV↔JSON, TXT↔JSON)
- [ ] 批量文本处理

#### Task 2.2: 高级图片压缩 🆕
- [ ] 智能压缩算法
- [ ] 无损压缩选项
- [ ] 压缩率预览
- [ ] 批量压缩优化

### 🎯 Phase 3: 用户体验优化
#### Task 3.1: CLI界面增强
- [ ] 进度条显示
- [ ] 错误处理优化
- [ ] 配置文件支持
- [ ] 预设模板

#### Task 3.2: 性能优化
- [ ] 多线程处理
- [ ] 内存优化
- [ ] 缓存机制
- [ ] 大文件处理

#### Task 3.3: 测试覆盖
- [ ] 单元测试完善
- [ ] 集成测试
- [ ] 性能测试
- [ ] 兼容性测试

## 📦 安装

### 系统要求
- Python 3.8+
- FFmpeg (视频/音频转换)
- 足够的磁盘空间用于临时文件

### 安装步骤

1. 克隆项目:
```bash
git clone <repository-url>
cd AnythingConvert
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 安装FFmpeg (视频转换必需):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# 下载FFmpeg并添加到PATH
```

## 🎯 使用方法

### 基本命令

```bash
# 查看帮助
python -m AnythingConvert --help

# 查看支持的格式
python -m AnythingConvert info

# 查看版本
python -m AnythingConvert version
```

### 图片转换

```bash
# 基本转换
python -m AnythingConvert image convert input.jpg output.png

# 调整尺寸
python -m AnythingConvert image convert input.jpg output.png --width 800 --height 600

# 设置质量 (JPEG/WebP)
python -m AnythingConvert image convert input.jpg output.jpg --quality 85

# 保持宽高比
python -m AnythingConvert image convert input.jpg output.png --width 800 --keep-aspect

# SVG转换
python -m AnythingConvert image convert input.svg output.png --width 512 --height 512
```

### 视频转换

```bash
# 基本转换
python -m AnythingConvert video convert input.mp4 output.avi

# 查看视频信息
python -m AnythingConvert video info input.mp4

# 提取音频
python -m AnythingConvert video extract-audio input.mp4 output.mp3
```

### 音频转换

```bash
# 基本转换
python -m AnythingConvert audio convert input.mp3 output.wav

# 查看音频信息
python -m AnythingConvert audio info input.flac

# 提取音频片段 (从30秒到90秒)
python -m AnythingConvert audio extract input.mp3 output.mp3 --start 30 --end 90

# 设置音频质量
python -m AnythingConvert audio convert input.wav output.mp3 --bitrate 320
```

### 文档转换

```bash
# 基本转换
python -m AnythingConvert doc convert input.pdf output.txt
python -m AnythingConvert doc convert input.docx output.pdf

# 查看文档信息
python -m AnythingConvert doc info document.pdf

# 提取文本
python -m AnythingConvert doc extract-text document.pdf output.txt
```

## 🛠️ 开发

### 项目结构
```
AnythingConvert/
├── src/
│   ├── __init__.py
│   ├── main.py              # 主入口
│   ├── cli/                 # 命令行接口
│   │   ├── image_cli.py     # 图片CLI
│   │   ├── video_cli.py     # 视频CLI
│   │   ├── audio_cli.py     # 音频CLI
│   │   └── document_cli.py  # 文档CLI
│   ├── converters/          # 转换器模块
│   │   ├── image.py         # 图片转换
│   │   ├── video.py         # 视频转换
│   │   ├── audio.py         # 音频转换
│   │   ├── document.py      # 文档转换
│   │   └── text.py          # 文本转换
│   ├── compressors/         # 压缩模块
│   ├── utils/               # 工具函数
## 🚀 快速开始

### 命令行使用

#### 图片转换
```bash
# 单个文件转换
python -m anythingconvert image convert input.jpg output.png --quality 90

# 批量转换
python -m anythingconvert image batch ./images ./converted --format webp

# 图片压缩
python -m anythingconvert image compress input.jpg output.jpg --quality 70
```

#### 视频转换
```bash
# 视频格式转换
python -m anythingconvert video convert input.mp4 output.avi

# 提取音频
python -m anythingconvert video extract-audio input.mp4 output.mp3

# 获取视频信息
python -m anythingconvert video info input.mp4
```

#### 音频转换
```bash
# 音频格式转换
python -m anythingconvert audio convert input.mp3 output.wav --bitrate 320k

# 提取音频片段
python -m anythingconvert audio extract input.mp3 output.mp3 --start 30 --duration 60
```

#### 文档转换
```bash
# 文档转换
python -m anythingconvert doc convert input.pdf output.txt
python -m anythingconvert doc convert input.docx output.pdf
```

#### 文本转换 (新功能)
```bash
# 编码转换
python -m anythingconvert text convert input.txt output.txt --from-encoding gbk --to-encoding utf-8

# CSV转JSON
python -m anythingconvert text convert data.csv data.json

# JSON转CSV
python -m anythingconvert text convert data.json data.csv
```

## 💻 Python API 使用

### 图片转换 API
```python
from anythingconvert.converters.image import ImageConverter

converter = ImageConverter()

# 基本转换
converter.convert('photo.jpg', 'photo.png')

# 带参数转换
converter.convert(
    'image.png', 
    'image.webp', 
    quality=85,
    width=800,
    height=600
)

# 图片压缩
converter.compress('large.jpg', 'compressed.jpg', quality=70, max_size_kb=500)
```

### 视频转换 API
```python
from anythingconvert.converters.video import VideoConverter

converter = VideoConverter()

# 视频转换
converter.convert(
    'movie.mp4', 
    'movie.avi',
    codec='libx264',
    bitrate='2M',
    resolution='1280x720'
)

# 获取视频信息
info = converter.get_video_info('movie.mp4')
print(f"时长: {info['duration']}秒")
print(f"分辨率: {info['width']}x{info['height']}")

# 提取音频
converter.extract_audio('video.mp4', 'audio.mp3', audio_format='mp3')
```

### 音频转换 API
```python
from anythingconvert.converters.audio import AudioConverter

converter = AudioConverter()

# 音频转换
converter.convert(
    'song.mp3', 
    'song.wav',
    bitrate='320k',
    sample_rate=44100,
    channels=2
)

# 获取音频信息
info = converter.get_audio_info('song.mp3')
print(f"时长: {info['duration']}秒")
print(f"比特率: {info['bitrate']}kbps")

# 提取音频片段 (30秒开始，持续60秒)
converter.extract_segment('song.mp3', 'clip.mp3', 30.0, 60.0)
```

### 文档转换 API
```python
from anythingconvert.converters.document import DocumentConverter

converter = DocumentConverter()

# 文档转换
converter.convert('document.pdf', 'document.txt')
converter.convert('report.docx', 'report.pdf')

# 获取文档信息
info = converter.get_document_info('document.pdf')
print(f"页数: {info.get('pages', 'N/A')}页")
print(f"文件大小: {info.get('size', 'N/A')}字节")
```

### 文本转换 API (新功能)
```python
from anythingconvert.converters.text import TextConverter

converter = TextConverter()

# 编码转换
converter.convert_encoding('input.txt', 'output.txt', 'gbk', 'utf-8')

# CSV转JSON
converter.csv_to_json('data.csv', 'data.json')

# JSON转CSV
converter.json_to_csv('data.json', 'data.csv')
```

## 🔧 技术规范

### 代码规范
- **编码标准**: PEP 8
- **文档字符串**: Google Style
- **类型提示**: 使用 typing 模块
- **错误处理**: 统一异常处理机制

### 测试规范
- **测试框架**: pytest
- **覆盖率要求**: ≥80%
- **测试类型**: 单元测试、集成测试、性能测试

### 性能要求
- **内存使用**: 单个转换任务 ≤ 512MB
- **处理速度**: 图片转换 ≤ 2秒/张，视频转换实时处理
- **并发支持**: 支持多线程批量处理

### 兼容性
- **Python版本**: 3.8+
- **操作系统**: Windows, macOS, Linux
- **依赖管理**: requirements.txt + setup.py

## 📁 项目结构
```
AnythingConvert/
├── __init__.py              # 包初始化
├── main.py                  # 主入口文件
├── converters/              # 转换器模块
│   ├── __init__.py
│   ├── image.py            # 图片转换器
│   ├── video.py            # 视频转换器
│   ├── audio.py            # 音频转换器
│   ├── document.py         # 文档转换器
│   └── text.py             # 文本转换器 (新增)
├── cli/                    # 命令行接口
│   ├── __init__.py
│   ├── image_cli.py        # 图片CLI
│   ├── video_cli.py        # 视频CLI
│   ├── audio_cli.py        # 音频CLI
│   ├── document_cli.py     # 文档CLI
│   └── text_cli.py         # 文本CLI (新增)
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── file_utils.py       # 文件处理工具
│   ├── validation.py       # 验证工具
│   └── config.py           # 配置管理
├── tests/                  # 测试文件
│   ├── __init__.py
│   ├── test_image.py
│   ├── test_video.py
│   ├── test_audio.py
│   ├── test_document.py
│   └── test_text.py        # 文本转换测试 (新增)
├── requirements.txt        # 依赖列表
├── setup.py               # 安装脚本
└── README.md              # 项目文档
```

## 📦 依赖管理

### 核心依赖
```txt
# 图片处理
Pillow>=10.0.0
opencv-python>=4.8.0
cairosvg>=2.7.0

# 视频音频处理  
ffmpeg-python>=0.2.0

# 文档处理
python-docx>=0.8.11
PyPDF2>=3.0.1
reportlab>=4.0.0
docx2txt>=0.8
odfpy>=1.4.0

# 文本处理 (新增)
chardet>=5.0.0
pandas>=2.0.0

# 命令行界面
typer>=0.9.0
rich>=13.0.0
click>=8.1.0

# 工具库
pathlib2>=2.3.0
```

### 可选依赖
```txt
# 性能优化
numpy>=1.24.0
numba>=0.57.0

# 测试工具
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
```

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_image.py

# 生成覆盖率报告
pytest --cov=anythingconvert --cov-report=html
```

### 测试覆盖
- 图片转换: 单元测试 + 集成测试
- 视频转换: 功能测试 + 性能测试  
- 音频转换: 格式兼容性测试
- 文档转换: 多格式转换测试
- 文本转换: 编码兼容性测试

### 内存优化
- 大文件分块处理
- 临时文件自动清理
- 内存使用监控

## 🛠️ 开发指南

### 环境设置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装预提交钩子
pre-commit install
```

### 代码贡献
1. Fork 项目
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🔗 相关资源

### 官方文档
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Pillow Documentation](https://pillow.readthedocs.io/)
- [Typer Documentation](https://typer.tiangolo.com/)

### 参考项目
- [ConvertX](https://github.com/C4illin/ConvertX) - 多媒体转换工具
- [FFmpeg-Python](https://github.com/kkroening/ffmpeg-python) - Python FFmpeg 绑定
- [ImageIO](https://github.com/imageio/imageio) - 图像I/O库
- https://www.freeconvert.com/ai-converter