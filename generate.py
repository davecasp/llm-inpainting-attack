import argparse
import random

import numpy as np
import pandas as pd
import torch

from jaxtyping import install_import_hook

with install_import_hook("src.inpainting_wrapper", "beartype.beartype"):
    from src.inpainting_wrapper import LLaDaWrapper
from src.utils import load_data


def generate_llada(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_data(args.dataset)
    inpainter = LLaDaWrapper(
        model_path=args.model_path,
        device=device,
        batch_size=args.batch_size,
        attack_input_size=args.attack_input_size,
        num_diffusion_steps=args.num_diffusion_steps,
        temperature=args.temperature,
        mask_padding=args.mask_padding,
    )

    pairs = [
        (item["goal"], item["target"])
        for item in dataset
        if "goal" in item and "target" in item
        for _ in range(args.number_samples)
    ]
    goals, targets = zip(*pairs) if len(pairs) > 0 else ([], [])
    
    output = inpainter.batched_inpainting(list(targets))

    df = pd.DataFrame({"prompt": output, "goal": goals, "target": targets})

    df.to_csv(args.output_file, index=False)
    print(f"Generated prompts saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate.py")
    parser.add_argument(
        "--dataset",
        help="Dataset to use for generating adversarial prompts. You can also provide a CSV file with goals and targets.",
        type=str,
        required=True,
    )
    parser.add_argument("--model_path", help="Path to the model", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--number_samples", help="Number of samples to generate for each Goal", type=int, default=2)
    parser.add_argument("--attack_input_size", help="Number of attack tokens", type=int, default=64)
    parser.add_argument("--batch_size", help="Batch size to use for generation", type=int, default=1)
    parser.add_argument(
        "--output_file", help="File path to save generated samples", type=str, default="./output/generated_prompts.csv"
    )
    parser.add_argument(
        "--num_diffusion_steps", help="Number of diffusion steps to use for generation", type=int, default=64
    )
    parser.add_argument("--temperature", help="Temperature to use for generation", type=float, default=0.0)
    parser.add_argument("--seed", help="Random seed to use for generation", type=int, default=42)
    parser.add_argument(
        "--mask_padding", help="Boolean to mask padding", action=argparse.BooleanOptionalAction, default=True
    )

    args = parser.parse_args()

    print("Using seed:", args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    generate_llada(args)
