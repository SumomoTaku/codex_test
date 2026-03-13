# FLUX ImageNet Generative Framework

一个基于 Hugging Face `diffusers` 的 FLUX 生成式框架，支持：

- 通过 ImageNet class id（0-999）生成图像
- 通过 class label 文本生成图像
- 通过 style prompt 对画面风格进行控制

## 从 0 开始（推荐）

```bash
bash scripts/bootstrap_env.sh
```

该脚本会自动完成：

1. 创建 `.venv`
2. 升级 `pip/setuptools/wheel`
3. 安装 CPU 版 PyTorch（默认）
4. 安装本项目与依赖
5. 运行 `flux-imagenet --help` 做可执行性检查

> 默认是 CPU 环境（最通用）。如需自定义，可通过环境变量：
>
> - `PYTHON_BIN`（默认 `python3`）
> - `VENV_DIR`（默认 `.venv`）
> - `INSTALL_MODE`（默认 `cpu`）

例如：

```bash
PYTHON_BIN=python3.11 VENV_DIR=.venv INSTALL_MODE=cpu bash scripts/bootstrap_env.sh
```

## 手动安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## 快速开始

### 1) 使用 ImageNet class id 生成

```bash
source .venv/bin/activate
flux-imagenet --class-id 207 --style "studio lighting, ultra realistic" --seed 42 --output outputs/golden_retriever.png
```

### 2) 使用 class label 生成

```bash
source .venv/bin/activate
flux-imagenet --class-label "tabby cat" --style "cinematic, 35mm film" --seed 123 --output outputs/tabby.png
```

## Python API

```python
from flux_imagenet import FluxImageNetGenerator

g = FluxImageNetGenerator()
g.generate_to_file(
    output_path="outputs/lynx.png",
    class_id=287,
    style_prompt="wildlife photography, bokeh",
    seed=7,
)
```

## 说明

- 默认模型为 `black-forest-labs/FLUX.1-schnell`。
- 首次运行会下载模型权重（请确保网络可访问 Hugging Face）。
- 某些模型可能需要先登录 Hugging Face：`huggingface-cli login`。
- 如果你有 GPU，推理速度会明显更快。
