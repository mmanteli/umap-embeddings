#!/bin/bash

# Note: not all langs in all data, this is to show all possibilities
models=("bge-m3" "e5" "xlmr" "xlmr-long")
folds=(1 2 3 4 5 6 7 8 9 10)
data=("CORE" "register_oscar" "dirty" "cleaned")




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
