#!/bin/bash
#SBATCH --job-name=plotting
#SBATCH --account=project_2009199
#SBATCH --time=00:20:00
#SBATCH --partition=medium
##SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


dir=$1
langs=$2

embeds=/scratch/project_2009199/umap-embeddings/model_embeds/${dir}/
langs_combined=$(echo $langs | tr ',' '-')   # for saving the log files

echo $langs, $dir, plotting
module load pytorch
srun python3 plot_embeddings.py $embeds $langs

seff $SLURM_JOBID

mkdir -p logs/plotting_${langs_combined}_${dir}/
mv logs/${SLURM_JOBID}.* logs/plotting_${langs_combined}_${dir}/
