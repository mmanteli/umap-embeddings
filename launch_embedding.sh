#!/bin/bash

# Note: not all langs in all data, this is to show all possibilities
models=("bge-m3" "e5" "xlmr" "xlmr-long")
data=("CORE" "register_oscar" "cleaned")
folds=(1 2 3 4 5 6 7 8 9 10)
langs=("en" "fa" "fi" "fr" "sv" "tr" "ur" "zh")

for model in "${models[@]}"; do
    if [ "$model" == "bge-m3" ] || [ "$model" == "e5" ]; then
        for lang in "${langs[@]}"; do
            for d in "${data[@]}"; do
                sbatch run_embeds.sh $lang $d $model 0
                sleep 1
            done
        done
    else
        for lang in "${langs[@]}"; do
            for fold in "${folds[@]}"; do
                for d in "${data[@]}"; do
                    sbatch run_embeds.sh $lang $d $model $fold 
                    sleep 1
                done
            done
            #sleep 15
        done
    fi
done