#!/bin/bash
#SBATCH --job-name=plotting
#SBATCH --account=project_2002026
#SBATCH --time=00:15:00
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# call with this
# run_plotting.sh $model_name $data_name $langs $wrt_column

model_name=$1
data_name=$2
langs=$3
wrt_column=$4

# Replace commas with hyphens for langs_hyphen
langs_hyphen=$(echo "$langs" | tr ',' '-')
# this is for saving

echo "plotting ${model_name} ${dataname} with ${langs_hyphen} wrt ${wrt_column}"

module load pytorch
srun python3 plot_embed.py \
        --embeddings="/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}/" \
        --languages=$langs \
        --model_name=$model_name \
        --data_name=$data_name \
        --use_column=$wrt_column \
        --save_path="/scratch/project_2009199/umap-embeddings/umap-figures/${data_name}/${model_name}/${langs_hyphen}/${wrt_column}/"

seff $SLURM_JOBID

mkdir -p logs/plotting_${model_name}_${dataname}/${langs_hyphen}/${wrt_column}/
mv logs/${SLURM_JOBID}.* logs/plotting_${model_name}_${dataname}/${langs_hyphen}/${wrt_column}/
