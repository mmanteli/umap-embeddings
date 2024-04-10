#!/bin/bash
#SBATCH --job-name=embeds_xlmr
#SBATCH --account=project_2002026
#SBATCH --time=00:15:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:a100:1,nvme:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

lang=$1
fold=$2
echo $lang , $fold, xlm-r-bce
module load pytorch
srun python3 embeds.py $lang $fold

seff $SLURM_JOBID

mkdir -p logs/${SLURM_JOB_NAME}_fold_${fold}/
mv logs/${SLURM_JOBID}.* logs/${SLURM_JOB_NAME}_fold_${fold}/
