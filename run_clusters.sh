#!/bin/bash

model="e5"
data="balanced_register_oscar"

python clusters.py --data="model_embeds/${data}/${model}/" \
                   --langs=["fr"] \
                   --data_name=$data \
                   --model_name=$model \
                   --labels="without_MT" \
                   --sample=6000 \
                   --cmethod="all" \
                   --rmethod="umap" \
                   --n_umap="[2,5,1]" \
                   --save_dir="metric_plots/${data}/${model}/" \
                   --save_prefix="fig"
                   

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
                   