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
data_name=$2
model_name=$3
fold=$4

echo $lang, $data_name, $model_name, $fold
module load pytorch

srun python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name --fold=$fold
seff $SLURM_JOBID
mkdir -p logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
exit 0
