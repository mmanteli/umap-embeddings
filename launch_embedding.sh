#!/bin/bash

# Note: not all langs in all data, this is to show all possibilities
models=("xlmr" "xlmr-long")
folds=(6 7 8 9 10)
#data=("CORE" "register_oscar" "dirty" "cleaned")
data=("balanced_register_oscar")




for model in "${models[@]}"; do
    if [ "$model" == "bge-m3" ] || [ "$model" == "e5" ]; then  
        for d in "${data[@]}"; do
            sbatch run_embeds.sh $d $model 0
        done    
    else
        for fold in "${folds[@]}"; do
            for d in "${data[@]}"; do
                sbatch run_embeds.sh $lang $d $model $fold 
            done
        done
    fi
done
