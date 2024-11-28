#!/bin/bash
#SBATCH --job-name=average-clustering
#SBATCH --account=project_462000353
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --mem=36G
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
model_name="bge-m3"    # this for naming
data="CORE"


srun python clusters_mean.py --data="/scratch/project_462000353/amanda/register-clustering/data/model_embeds/${data}/${model}/" \
                           --langs=["en","fi","fr"] \
                           --data_name=$data \
                           --model_name=$model_name \
                           --labels="without_MT" \
                           --cmethod="all" \
                           --rmethod="umap" \
                           --n_umap="[2,10,2]" \
                           --save_dir="/scratch/project_462000353/amanda/register-clustering/data/cluster_plots/averaged/${data}/${model_name}/" \
                           --save_prefix="averaged"

sacct -j $SLURM_JOBID
