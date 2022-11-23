#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512gb
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --mail-user=sshrestha8@student.gsu.edu
#SBATCH --account=csc344r73
#SBATCH --partition=qGPU48
#SBATCH --gres=gpu:V100:4
#SBATCH --output=outputs/output_%j
#SBATCH --error=errors/error_%j

cd /scratch
mkdir $SLURM_JOB_ID
cd $SLURM_JOB_ID

iget -r /arctic/work/sshrestha8/masterproject/super-resolution2

source /userapp/virtualenv/ENV_SR/venv/bin/activate

python super-resolution2/fakr_train_samples.py

cd /scratch
icd /arctic/projects/csc344s73
iput -rf $SLURM_JOB_ID

