#!/bin/bash
#SBATCH --job-name=embeds
#SBATCH --account=project_2005092
#SBATCH --time=00:10:00
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1,nvme:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

lang=$1
data_path=$2
model_name=$3
fold=$4
echo $lang, $data_path, $model_name, $fold
module load pytorch

data_name=$(basename "$2")
#srun python3 embeds.py $lang $fold

case "$model_name" in
    xlmr)
        srun python3 embeds-xlmr.py $lang $fold $data_path "/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}-fold-${fold}/"
        seff $SLURM_JOBID
        mkdir -p logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
        mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
        exit 0
        ;;
    xlmr-long)
        srun python3 embeds-xlmr-long-run.py $lang $fold $data_path "/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}-fold-${fold}/"
        seff $SLURM_JOBID
        mkdir -p logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
        mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
        exit 0
        ;;
    bge-m3)
        srun python3 embeds-bge.py $lang $data_path "/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}/"
        seff $SLURM_JOBID
        mkdir -p logs/embeds_${model_name}_${data_name}/${lang}/
        mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${data_name}/${lang}/
        exit 0
        ;;
    e5)
        srun python3 embeds-e5.py $lang $data_path "/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}/"
        seff $SLURM_JOBID
        mkdir -p logs/embeds_${model_name}_${data_name}/${lang}/
        mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${data_name}/${lang}/
        exit 0
        ;;
    *)
        echo "Unsupported model name: $model_name"
        exit 1
        ;;
esac

#seff $SLURM_JOBID

#mkdir -p logs/${SLURM_JOB_NAME}_fold_${fold}/
#mv logs/${SLURM_JOBID}.* logs/${SLURM_JOB_NAME}_fold_${fold}/
