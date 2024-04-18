#!/bin/bash

# Define arrays for model names and language combinations
#model_names=("bge-m3" "e5" "xlmr-fold-1" "xlmr-fold-5" "xlmr-fold-7" "xlmr-fold-8")
model_names=("xlmr-long-fold-7")
data_name="final_cleaned"
langs_combinations=("en,fr,tr" "en,fr,fi" "en,fi,sv" "en,fi,fr,sv,tr")
#model_names=("bge-m3")


# Loop through each model name
for model_name in "${model_names[@]}"; do
    # Loop through each language combination
    for langs in "${langs_combinations[@]}"; do
        # Replace commas with hyphens for langs_hyphen
        langs_hyphen=$(echo "$langs" | tr ',' '-')
        python plot_umap_embeddings.py "model_embeds/${data_name}/${model_name}/" "$langs" "umap-figures/${data_name}/${model_name}/${langs_hyphen}/" 2000
    done
done

