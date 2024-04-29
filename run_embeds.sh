#!/bin/bash
#SBATCH --job-name=embeds
#SBATCH --account=project_2002026
#SBATCH --time=00:20:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:a100:1,nvme:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


data_name=$1
model_name=$2
fold=$3

case $data_name in
    CORE)
        langs=("en" "fi" "fr" "sv" "tr")
        ;;
    register_oscar)
        langs=("en" "fr" "ur" "zh")
        ;;
    cleaned)
        langs=("en" "fa" "fi" "fr" "sv" "ur" "tr" "zh")
        ;;
    dirty)
        langs=("en" "fa" "fi" "fr" "sv" "ur" "tr" "zh")
        ;;
    *)
        echo "Invalid data_name. Please specify CORE or REG."
        exit 1
        ;;
esac


echo $langs, $data_name, $model_name, $fold
module load pytorch

for lang in "${langs[@]}"; do
    srun python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name --fold=$fold
    #echo python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name --fold=$fold
done
exit 0
seff $SLURM_JOBID
mkdir -p logs/embeds_${model_name}_${fold_str}_${data_name}/${lang}/
mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${fold_str}_${data_name}/${lang}/
exit 0
