#!/bin/bash
#SBATCH --job-name=xlmr-long-cluster-mean
#SBATCH --account=project_2009199
#SBATCH --partition=medium
#SBATCH --time=09:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load pytorch
model="xlmr-long-fold-*"
model_name="xlmr-long-runs"
data="balanced_register_oscar"

srun python clusters_mean.py --data="model_embeds/${data}/${model}/" \
                           --langs=["en","fr","zh","ur"] \
                           --data_name=$data \
                           --model_name=$model_name \
                           --labels="without_MT" \
                           --sample=5000 \
                           --cmethod="all" \
                           --rmethod="umap" \
                           --n_umap="[2,10,2]" \
                           --save_dir="metric_plots/average/${data}/${model_name}/" \
                           --save_prefix="averaged"

seff $SLURM_JOBID
exit 0
srun python clusters.py --data="model_embeds/${data}/${model}/" \
                           --langs=["en","fr","zh","ur"] \
                           --data_name=$data \
                           --model_name=$model \
                           --labels="without_MT" \
                           --sample=5000 \
                           --cmethod="all" \
                           --rmethod="umap" \
                           --n_umap="[2,10,2]" \
                           --save_dir="metric_plots/${data}/${model}/" \
                           --save_prefix="averaged"
seff $SLURM_JOBID
exit 0





python clusters.py --data="model_embeds/${data}/${model}/" \
                   --langs=["en","fr","zh"] \
                   --data_name=$data \
                   --model_name=$model \
                   --labels="without_MT" \
                   --sample=5000 \
                   --cmethod="spherical-kmeans" \
                   --rmethod="umap" \
                   --n_umap="[2,8,2]" \
                   --save_dir="metric_plots/${data}/${model}/" \
                   --save_prefix="new_fig"
                   
                   #--pca_before_umap=5 \


exit 0
python clusters.py --data="model_embeds/veronikas/" \
                   --langs=["fr","en","fi"] \
                   --data_name="Veronika's" \
                   --labels="without_MT" \
                   --header=["lang","label","embedding"] \
                   --use_column_embeddings=["embedding"] \
                   --use_column_labels="label" \
                   --cmethod="all" \
                   --rmethod="umap" \
                   --n_umap="[2,5,1]" \
                   --save_dir="metric_plots/veronikas/" \
                   --save_prefix="fig"

