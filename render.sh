#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Einstein Vision Rendering"
#SBATCH -A rbe549
#SBATCH -p academic
#SBATCH -t 23:59:59
#SBATCH --gres=gpu:1
#SBATCH --error=slurm_out/slurm_einstein_render_%A.err
#SBATCH --output=slurm_out/slurm_einstein_render_%A.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

module load py-pip/24.0 ffmpeg

source ./einsteinvenv/bin/activate

pip install -r requirements.txt

python -u BlenderStuff.py
