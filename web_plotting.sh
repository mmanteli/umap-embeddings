#!/bin/bash

# Define arrays for model names and language combinations
model_names=("bge-m3" "e5" "xlmr-fold-1" "xlmr-fold-5" "xlmr-fold-7" "xlmr-fold-8")
langs_combinations=("en,fr,tr" "en,fr,fi" "en,fi,sv" "en,fi,fr,sv,tr")
#model_names=("bge-m3")


# Loop through each model name
for model_name in "${model_names[@]}"; do
    # Loop through each language combination
    for langs in "${langs_combinations[@]}"; do
        # Replace commas with hyphens for langs_hyphen
        langs_hyphen=$(echo "$langs" | tr ',' '-')
        
        # Run the command
        #python plot_umap_embeddings.py TEST_model_embeds/final_core/bge-m3/ en,fr,tr CORE_umap_figures/all_labels/bge-m3/en-fr-tr/
        mkdir -p "CORE_umap_figures/big_labels/${model_name}/${langs_hyphen}/"
        python plot_umap_embeddings.py "TEST_model_embeds/final_core/${model_name}/" "$langs" "CORE_umap_figures/big_labels/${model_name}/${langs_hyphen}/"
    done
done

