#!/bin/bash
#SBATCH --account=def-masd
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=12:00:00
#SBATCH --job-name=LM
module spider r
module load gcc/9.3.0 r/4.0.2

lscpu

Rscript  /home/kyang/SCAD_MCP/LM/ncvreg_LM_sim.R
