#!/bin/bash
#SBATCH --job-name=embeds_e5
#SBATCH --account=project_2002026
#SBATCH --time=00:20:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:a100:1,nvme:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

lang=$1
echo $lang , embeds_e5
module load pytorch
srun python3 embeds-e5.py $lang

seff $SLURM_JOBID

mkdir -p logs/${SLURM_JOB_NAME}_${lang}/
mv logs/${SLURM_JOBID}.* logs/${SLURM_JOB_NAME}_${lang}/