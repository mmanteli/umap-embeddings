#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --account=project_462000353
#SBATCH --time=02:45:00
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus-per-node=1
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
    hplt)
        langs=("en" "fr" "ur" "zh")
        ;;
    balanced_register_oscar)
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
module purge
#module load LUMI
#module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315
module use /appl/local/csc/modulefiles
module load pytorch/2.4

for lang in "${langs[@]}"; do
    srun python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name --fold=$fold
    #echo python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name --fold=$fold
done
sacct --format="jobid,Elapsed" -j $SLURM_JOBID 
mkdir -p logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
exit 0
