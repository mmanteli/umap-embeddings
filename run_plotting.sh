#!/bin/bash
#SBATCH --job-name=plotting
#SBATCH --account=project_2002026
#SBATCH --time=00:30:00
#SBATCH --partition=test
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
wrt_column="preds_best"

# Replace commas with hyphens for langs_hyphen and make a list for params
langs_hyphen=$(echo "$langs" | tr ',' '-')
langs_list=$(echo "$langs" | sed 's/,/","/g' | awk '{print "[\""$0"\"]"}')

echo "plotting ${model_name} ${dataname} with ${langs_hyphen} wrt ${wrt_column}"

module load pytorch
srun python3 plot_embeddings.py \
        --embeddings="/scratch/project_2009199/umap-embeddings/model_embeds/${data_name}/${model_name}/" \
        --languages=$langs_list \
        --model_name=$model_name \
        --data_name=$data_name \
        --use_column_labels=$wrt_column \
        --extension="html" \
        --hover_text="text" \
        --save_dir="/scratch/project_2009199/umap-embeddings/umap-figures/${data_name}/${model_name}/${langs_hyphen}/${wrt_column}/"

seff $SLURM_JOBID

mkdir -p logs/plotting_${model_name}_${data_name}/${langs_hyphen}/${wrt_column}/
mv logs/${SLURM_JOBID}.* logs/plotting_${model_name}_${data_name}/${langs_hyphen}/${wrt_column}/
