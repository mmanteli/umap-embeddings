#!/bin/bash

models=("bge-m3" "e5" "xlmr")
folds=(1 2 3 4 5 6 7 8 9 10)
langs=("en,fr,zh" "en,fi" "en,fi,tr" "fa,ur,sv", "en,fi,fr,sv,tr")

for model in "${models[@]}"; do
    if [ "$model" == "bge-m3" ] || [ "$model" == "e5" ]; then
        for lang in "${langs[@]}"; do
            sbatch run_plotting.sh $lang $model
            sleep 1
        done
    else
        for lang in "${langs[@]}"; do
            for fold in "${folds[@]}"; do
                sbatch run_plotting.sh $lang "${model}-fold-${fold}"
                sleep 1
            done
            #sleep 15
        done
    fi
done
