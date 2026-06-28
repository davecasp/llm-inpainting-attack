#!/bin/bash
#SBATCH --job-name=inpainting           # Job name
#SBATCH --output=logs/%x_%j.out         # Output log (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err          # Error log
#SBATCH --time=20:00:00                 # Max runtime (hh:mm:ss)
#SBATCH --partition=gpu_h200            # Partition/queue name
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --cpus-per-task=4               # Number of CPU cores
#SBATCH --mem=60G                       # Memory per node
#SBATCH --ntasks=1                      # One task per node

export HUGGINGFACE_HUB_CACHE=path/to/cache

# Run your Python script
python generate.py --dataset JailbreakBench --batch_size 100 --number_samples 2000 --output_file ./output/moe_generated_prompts.csv --model_path inclusionAI/LLaDA-MoE-7B-A1B-Base
