#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH -n 4

cd /home/zihanwu7/big_matrix_cocluster
conda activate cocluster
python notebook.py
