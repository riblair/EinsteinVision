#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=32g
#SBATCH -J "Einstein Vision Generation"
#SBATCH -A rbe549
#SBATCH --array=0-3
#SBATCH -p academic
#SBATCH -t 23:59:59
#SBATCH --gres=gpu:1
#SBATCH --error=slurm_out/slurm_einstein_%A_%a.err
#SBATCH --output=slurm_out/slurm_einstein_%A_%a.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

module load py-pip/24.0 ffmpeg

source ./einsteinvenv/bin/activate

pip install -r requirements.txt
# so far 10 and 7 are running fully
scene_list=(6 5 4 8)
start_list=(850 600 254 60)
Json_Name_list=("6" "5" "4" "8")
Outputs_list=("Outputs6/" "Outputs5/" "Outputs4/" "Outputs8/")
Video_Name_list=("out6" "out5" "out4" "out8")
export EASYOCR_CACHE_DIR=$HOME/.easyocr_models
python -u Wrapper.py --Scene "Videos/scene${scene_list[$SLURM_ARRAY_TASK_ID]}_front.mp4" --Start ${start_list[$SLURM_ARRAY_TASK_ID]} --Json_Name "Scenes/scene${Json_Name_list[$SLURM_ARRAY_TASK_ID]}.json" --Outputs ${Outputs_list[$SLURM_ARRAY_TASK_ID]} --Video_Name "${Video_Name_list[$SLURM_ARRAY_TASK_ID]}.mp4"
