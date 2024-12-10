#!/bin/bash
#SBATCH --job-name=7-clustering
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
source .venv/bin/activate

model="bge-m3-fold-7"
data="hplt"

echo $model $data "cluster metrics"
srun python clusters.py --data="/scratch/project_462000353/amanda/register-clustering/data/model_embeds/${data}/${model}/" \
                           --langs=["en","fr","ur","zh"] \
                           --data_name=$data \
                           --model_name=$model \
                           --labels="without_MT" \
                           --hover_text="text" \
                           --sample=2400 \
                           --cmethod="all" \
                           --rmethod="umap" \
                           --n_umap="[2,4,1]" \
                           --save_dir="/scratch/project_462000353/amanda/register-clustering/data/cluster_plots/${data}/${model}/with_hover/" \
                           --save_prefix="core_test"
sacct -j $SLURM_JOBID
exit 0
