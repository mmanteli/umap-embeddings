#!/bin/bash
#SBATCH --job-name=cluster
#SBATCH --account=project_2009199
#SBATCH --partition=medium
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load pytorch
model="bge-m3-long-fold-1"
data="balanced_register_oscar"


srun python clusters.py --data="model_embeds/${data}/${model}/" \
                           --langs=["en","fr","zh","ur"] \
                           --data_name=$data \
                           --model_name=$model \
                           --labels="without_MT" \
                           --sample=5000 \
                           --hover_text="text" \
                           --cmethod="kmeans" \
                           --rmethod="umap" \
                           --n_umap="[2,4,1]" \
                           --save_dir="metric_plots/${data}/${model}/with_hover/" \
                           --save_prefix="fig"
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

