#!/bin/bash

# Define arrays for model names and language combinations
#model_names=("bge-m3" "e5" "xlmr-fold-1" "xlmr-fold-5" "xlmr-fold-7" "xlmr-fold-8")
model_names=("xlmr-long-fold-1" "xlmr-fold-1" "xlmr-long-fold-7" "xlmr-fold-7" "bge-m3" "e5")
data_names=("balanced_register_oscar")
langs_combinations=("en,fr,zh,ur") # "en,fr,fi" "en,fi,sv" "en,fi,fr,sv,tr")
#model_names=("bge-m3")
wrt_column="preds_best"

# Loop through each model name
for model_name in "${model_names[@]}"; do
    # Loop through each data name
    for data_name in "${data_names[@]}"; do
        # Loop through each language combination
        for langs in "${langs_combinations[@]}"; do
            # Replace commas with hyphens for langs_hyphen
            langs_hyphen=$(echo "$langs" | tr ',' '-')
            # this is for saving
            python plot_embed.py \
                    --embeddings="/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}/" \
                    --languages=$langs \
                    --model_name=$model_name \
                    --data_name=$data_name \
                    --use_column=$wrt_column \
                    --sample=3000 \
                    --save_path="/scratch/project_2009199/umap-embeddings/umap-figures/${data_name}/${model_name}/${langs_hyphen}/${wrt_column}/"
        done
    done
done

