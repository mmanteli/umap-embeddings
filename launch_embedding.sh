#!/bin/bash


models=("bge" "e5" "xlmr")
#models=("e5")
#data=("/scratch/project_2009199/sampling_oscar/final_cleaned/" "/scratch/project_2009199/sampling_oscar/final_uncleaned/" "/scratch/project_2009199/sampling_oscar/final_reg_oscar/")
#data=("/scratch/project_2009199/sampling_oscar/final_core/")
data=("/scratch/project_2009199/sampling_oscar/final_dirty/")
folds=(1 5 7 8)
langs=("en" "fa" "fi" "fr" "sv" "tr" "ur" "zh")
#langs=("en" "fr" "tr" "fi" "sv")

for model in "${models[@]}"; do
    if [ "$model" == "bge" ] || [ "$model" == "e5" ]; then
        for lang in "${langs[@]}"; do
            for d in "${data[@]}"; do
                sbatch run_embeds.sh $lang $d $model 
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