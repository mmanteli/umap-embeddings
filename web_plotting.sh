#!/bin/bash

# Define arrays for model names and language combinations
#model_names=("bge-m3" "e5" "xlmr-fold-1" "xlmr-fold-5" "xlmr-fold-7" "xlmr-fold-8")
model_names=("bge-m3")
data_name="CORE"
langs_combinations=("en,fr,tr") # "en,fr,fi" "en,fi,sv" "en,fi,fr,sv,tr")
#model_names=("bge-m3")


# Loop through each model name
for model_name in "${model_names[@]}"; do
    # Loop through each language combination
    for langs in "${langs_combinations[@]}"; do
        # Replace commas with hyphens for langs_hyphen
        langs_hyphen=$(echo "$langs" | tr ',' '-')
        python plot_embed.py \
                --embeddings="/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}/" \
                --languages=$langs \
                --model_name=$model_name \
                --data_name=$data_name \
                --use_column="preds_best" \
                --save_path="/scratch/project_2009199/umap-embeddings/umap-figures/${data_name}/${model_name}/"
    done
done

