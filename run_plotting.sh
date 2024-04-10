#!/bin/bash
#SBATCH --job-name=plotting
#SBATCH --account=project_2009199
#SBATCH --time=00:20:00
#SBATCH --partition=medium
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

langs=$1
fold=$2
embeds=/scratch/project_2009199/embeddings-and-umap/model_embeds/xlmr-fold-${fold}/
langs_combined=$(echo $langs | tr ',' '-')   # for saving the log files

echo $langs, $fold, plotting
module load pytorch
srun python3 plot_embeddings.py $embeds $langs

seff $SLURM_JOBID

mkdir -p logs/${SLURM_JOB_NAME}_fold_${fold}/
mv logs/${SLURM_JOBID}.* logs/${SLURM_JOB_NAME}_${langs_combined}_fold_${fold}/
