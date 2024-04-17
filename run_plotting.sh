#!/bin/bash
#SBATCH --job-name=plotting
#SBATCH --account=project_2002026
#SBATCH --time=00:40:00
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


langs=$1
dir=$2


embeds=/scratch/project_2009199/umap-embeddings/model_embeds/${dir}/
langs_combined=$(echo $langs | tr ',' '-')   # for saving the log files

echo $langs, $dir, plotting
module load pytorch
srun python3 plot_embeddings.py $embeds $langs

seff $SLURM_JOBID

mkdir -p logs/plotting_${dir}_${langs_combined}/
mv logs/${SLURM_JOBID}.* logs/plotting_${dir}_${langs_combined}/
