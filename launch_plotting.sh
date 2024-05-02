#!/bin/bash

# Define arrays for model names and language combinations
#model_names=("bge-m3" "e5" "xlmr-fold-1" "xlmr-fold-5" "xlmr-fold-7" "xlmr-fold-8")
model_names=("bge-m3" "e5" "xlmr-fold-1" "xlmr-fold-7" "xlmr-long-fold-1" "xlmr-long-fold-7")
data_names=("cleaned")
langs_combinations=("en,fi,fr,sv,tr")
wrt_column="preds"

# Loop through each model name
for model_name in "${model_names[@]}"; do
    # Loop through each data name
    for data_name in "${data_names[@]}"; do
        # Loop through each language combination
        for langs in "${langs_combinations[@]}"; do
            sbatch run_plotting.sh $model_name $data_name $langs $wrt_column
        done
    done
done
