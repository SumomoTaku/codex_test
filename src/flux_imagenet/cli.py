from __future__ import annotations

import argparse
from pathlib import Path

from .generator import FluxImageNetGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate images with FLUX from ImageNet labels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--class-id", type=int, help="ImageNet class id in [0, 999]")
    group.add_argument("--class-label", type=str, help="Free-text class label (e.g. 'golden retriever')")
    parser.add_argument("--style", type=str, default=None, help="Optional style prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("outputs/generated.png"), help="Output image path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    generator = FluxImageNetGenerator()
    out = generator.generate_to_file(
        output_path=args.output,
        class_id=args.class_id,
        class_label=args.class_label,
        style_prompt=args.style,
        seed=args.seed,
    )
    print(f"Saved generated image to: {out}")


if __name__ == "__main__":
    main()
