#!/bin/bash

#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH -o /home/mila/b/bertranh/dev/llama/slurm-%j.out

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate llama

TARGET_FOLDER=/home/mila/p/poradaia/scratch/llama_weights

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
torchrun --nproc_per_node 2 example.py --ckpt_dir $TARGET_FOLDER/13B/ --tokenizer_path $TARGET_FOLDER/tokenizer.model
