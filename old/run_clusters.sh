#!/bin/bash

#model="xlmr-long-fold-1"
data="balanced_register_oscar"

python clusters_mean.py --data="model_embeds/${data}/xlmr-long-fold-*/" \
                       --langs=["en","fr","zh","ur"] \
                       --data_name=$data \
                       --model_name="xlmr-long" \
                       --labels="without_MT" \
                       --sample=2000 \
                       --cmethod="all" \
                       --rmethod="umap" \
                       --n_umap="[2,6,2]" \
                       --save_dir="metric_plots/average/${data}/xlmr-long/" \
                       --save_prefix="mean"
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
                   