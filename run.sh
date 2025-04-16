#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=32g
#SBATCH -J "Einstein Vision Generation"
#SBATCH -A rbe549
#SBATCH --array=0-7
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

# Edit values so that they are videos 7 through 12
scene_list=(1 2 3 4 5 6 7 8 9 10 11 12 13)  # Do all 14
Json_Name_list=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13")
Outputs_list=("Outputs1/" "Outputs2/" "Outputs3/" "Outputs4/" "Outputs5/" "Outputs6/" "Outputs7/" "Outputs8/" "Outputs9/" "Outputs10/" "Outputs11/" "Outputs12/" "Outputs13/")
Video_Name_list=("out1" "out2" "out3" "out4" "out5" "out6" "out7" "out8" "out9" "out10" "out11" "out12" "out13")
export EASYOCR_CACHE_DIR=$HOME/.easyocr_models
python -u Wrapper.py --Scene "Videos/scene${scene_list[$SLURM_ARRAY_TASK_ID]}_front.mp4" --Json_Name "Scenes/scene${Json_Name_list[$SLURM_ARRAY_TASK_ID]}.json" --Outputs ${Outputs_list[$SLURM_ARRAY_TASK_ID]} --Video_Name "${Video_Name_list[$SLURM_ARRAY_TASK_ID]}.mp4"
