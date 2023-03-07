#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:v100:1                                     # Ask for 1 GPU
#SBATCH --mem=30G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours
#SBATCH -o /home/mila/b/bertranh/dev/llama/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate llama

TARGET_FOLDER=/home/mila/p/poradaia/scratch/llama_weights

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
torchrun --nproc_per_node 1 example.py --ckpt_dir $TARGET_FOLDER/7B/ --tokenizer_path $TARGET_FOLDER/tokenizer.model
