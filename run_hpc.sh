#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --time=24:00:00

cd /home/zihanwu7/big_matrix_cocluster
/home/zihanwu7/miniconda3/envs/cocluster/bin/python notebook.py
