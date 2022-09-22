# nonconvexAG

This repository contains simulation study codes and results for [my paper](https://arxiv.org/abs/2009.10629). The paper-related material is under [this directory](/paper). All the codes and outputs (both intermediate outputs and final outputs) can be found [here](/paper/simulation_study). All the results were run on Compute Canada. All studies using GPUs were run on Compute Canada Nvidia A100 GPU(s), which the slurm outputs ([slurm_file_1](/paper/simulation_study/tasks/task1speed/(task1speed_SCAD)_slurm-9615091.out), [slurm_file_2](/paper/simulation_study/tasks/task1speed/(task1speed_MCP)_slurm-9615089.out), [slurm_file_3](/paper/simulation_study/tasks/task2speed/(task2speed_SCAD)_slurm-9615092.out), [slurm_file_4](/paper/simulation_study/tasks/task2speed/(task2speed_MCP)_slurm-9615093.out)) show to have a CUDA compute capability of 8.0 ([`cupy.cuda.Device.compute_capability`](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html) returns '80' stands for compute capability of 8.0).


The summary of the simulation results is [here](/paper/simulation_study/summary.ipynb), where most of the simulation study results in the paper are from. The two original Jupyter notebooks are [the simulation study for SCAD/MCP-penalized linear models](/paper/simulation_study/LM_SCAD_MCP_cp%20(cupy).ipynb) and [the simulation study for SCAD/MCP-penalized logistic models](/paper/simulation_study/logistic_SCAD_MCP_cp%20(cupy).ipynb). To run on the server, I divided the codes into several chunks; they were in [this folder](/paper/simulation_study/tasks) *for python simulations* -- [task1](/paper/simulation_study/tasks/task1) and [task2](/paper/simulation_study/tasks/task2) contain files to test signal recovery performance for SCAD/MCP-penalized linear models and logistic models using AG; [task1speed](/paper/simulation_study/tasks/task1speed) and [task2speed](/paper/simulation_study/tasks/task2speed) contain files to test convergence speed and computing times for SCAD/MCP-penalized linear models and logistic models using AG v.s. ISTA v.s. coordinate descent. The R codes and results for `ncvreg` simulations are contained in [this directory](/paper/simulation_study/SCAD_MCP) -- click [here for penalized linear models](/paper/simulation_study/SCAD_MCP/LM) or [here for penalized logistic models](/paper/simulation_study/SCAD_MCP/logistic).


The manual for the PyPI package [`nonconvexAG`](https://pypi.org/project/nonconvexAG/) can be found [here](/nonconvexAG/README.md).
