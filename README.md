# FLUX ImageNet Generative Framework

一个基于 Hugging Face `diffusers` 的 FLUX 生成式框架，支持：

- 通过 ImageNet class id（0-999）生成图像
- 通过 class label 文本生成图像
- 通过 style prompt 对画面风格进行控制

## 安装

```bash
pip install -e .
```

## 快速开始

### 1) 使用 ImageNet class id 生成

```bash
flux-imagenet --class-id 207 --style "studio lighting, ultra realistic" --seed 42 --output outputs/golden_retriever.png
```

### 2) 使用 class label 生成

```bash
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
- 首次运行会下载模型权重。
- 如果你有 GPU，推理速度会明显更快。
