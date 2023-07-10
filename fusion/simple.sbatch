#!/bin/bash
# parameters for slurm
#SBATCH -c 16                          # number of cores, 1
#SBATCH --gres=gpu:1                  # number of gpus 1, remove if you don't use gpu's
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --mail-type=END,FAIL          # email status changes (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=24:00:00                # time limit 1h

# show actual node in output file, usefull for diagnostics
hostname

# load all required software modules
module load nvidia/cuda-10.1 

# It's nice to have some information logged for debugging
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

# Run the job -- make sure that it terminates itself before time is up
#220704
./gpu_burn 60  # if your cluster has GPU
python train.py