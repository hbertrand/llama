#!/bin/bash

#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -o /home/mila/b/bertranh/dev/llama/slurm-7b-%j.out

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate llama

TARGET_FOLDER=/home/mila/p/poradaia/scratch/llama_weights

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
torchrun --nproc_per_node 1 example.py --ckpt_dir $TARGET_FOLDER/7B/ --tokenizer_path $TARGET_FOLDER/tokenizer.model --max_batch_size 128
