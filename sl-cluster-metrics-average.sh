#!/bin/bash
#SBATCH --job-name=l-clustering
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=18:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
source .venv/bin/activate

model="bge-m3-fold*"   # this is used to find the data
cmethod="kmeans"
sample_per_lang=12000
model_name="bge-m3-${sample_per_lang}-per-lang-${cmethod}"    # this for naming
data="hplt"


srun python clusters_mean.py --data="/scratch/project_462000353/amanda/register-clustering/data/model_embeds/${data}/${model}/" \
                           --langs=["en","fr","ur","zh"] \
                           --data_name=$data \
                           --model_name=$model_name \
                           --labels="without_MT" \
                           --cmethod=$cmethod \
                           --sample=$sample_per_lang \
                           --rmethod="umap" \
                           --n_umap="[2,8,2]" \
                           --save_dir="/scratch/project_462000353/amanda/register-clustering/data/cluster_plots/averaged/${data}/${model_name}/" \
                           --save_prefix="averaged"

sacct -j $SLURM_JOBID
