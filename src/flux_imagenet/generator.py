from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from diffusers import FluxPipeline
from PIL import Image
from imagenet_simple_labels import IMAGENET_SIMPLE_LABELS


@dataclass
class GenerationConfig:
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    guidance_scale: float = 3.5
    num_inference_steps: int = 4
    height: int = 1024
    width: int = 1024
    max_sequence_length: int = 256


class FluxImageNetGenerator:
    """A small framework to generate images from ImageNet class labels using FLUX."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config or GenerationConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch_dtype if self.device == "cuda" else torch.float32

        self.pipeline = FluxPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        )
        self.pipeline.enable_model_cpu_offload()

    @staticmethod
    def imagenet_label(index: int) -> str:
        if not 0 <= index < len(IMAGENET_SIMPLE_LABELS):
            raise ValueError(f"ImageNet class id must be in [0, 999], got {index}.")
        return IMAGENET_SIMPLE_LABELS[index]

    @staticmethod
    def resolve_class_label(class_id: Optional[int], class_label: Optional[str]) -> str:
        if class_id is not None:
            return FluxImageNetGenerator.imagenet_label(class_id)
        if class_label:
            return class_label.strip()
        raise ValueError("Either class_id or class_label must be provided.")

    def build_prompt(self, class_name: str, style_prompt: Optional[str] = None) -> str:
        base = f"A high-quality, detailed photograph of a {class_name}, centered composition"
        return f"{base}, {style_prompt}" if style_prompt else base

    def generate(
        self,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        style_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        resolved_class = self.resolve_class_label(class_id, class_label)
        prompt = self.build_prompt(resolved_class, style_prompt)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            height=self.config.height,
            width=self.config.width,
            max_sequence_length=self.config.max_sequence_length,
            generator=generator,
        )

        return result.images[0]

    def generate_to_file(
        self,
        output_path: str | Path,
        class_id: Optional[int] = None,
        class_label: Optional[str] = None,
        style_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Path:
        image = self.generate(
            class_id=class_id,
            class_label=class_label,
            style_prompt=style_prompt,
            seed=seed,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return output
