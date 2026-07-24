#!/bin/bash

#SBATCH --job-name=classifier

# Run the job 10 times
#SBATCH --array=1-10

#SBATCH --account=aicore
#SBATCH --qos=aicore

#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=128G

#SBATCH --nodelist=chimera12
#SBATCH --partition=AICORE_A100

#SBATCH --time=3-00:00:00

#SBATCH --output=classifier_%A_%a.out
#SBATCH --error=classifier_%A_%a.err

#Optional
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=anya.tongprasith001@umb.edu

source ~/.bashrc
conda activate tfgpu

cd /home/anya.tongprasith001/U54REC/image_classification || { echo "cd failed"; exit 1; }

python -u resnet_img.py