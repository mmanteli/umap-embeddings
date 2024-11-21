#!/bin/bash
#SBATCH --job-name=cluster-e5
#SBATCH --account=project_2009199
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x.out
#SBATCH -e logs/%x.err

module load pytorch
model="e5"
data="balanced_register_oscar"

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

