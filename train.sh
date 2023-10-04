#!/bin/bash
#SBATCH --job-name=MUSIC_AVQA_one_model
#SBATCH --account=project_462000189
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=48:00:00

#SBATCH --output=outputs/output_%A_%a.txt
#SBATCH --error=errors/errors_%A_%a.txt

module use /appl/local/csc/modulefiles/
module load pytorch
export PYTHONUSERBASE=/scratch/project_462000189/ines/python_base
encoder_type=ViT
encoder=vit_base_patch16_224_in21k

output_dir=checkpoints/lavish16/$encoder_type/$encoder
output_best_dir=checkpoints/lavish16/$encoder_type/$encoder/best

mkdir -p $output_dir
mkdir -p $output_best_dir

srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 net_grd_avst/main_avst.py
