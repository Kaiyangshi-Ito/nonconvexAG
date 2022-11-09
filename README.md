# nonconvexAG

This repository contains simulation study codes and results for [my paper](https://arxiv.org/abs/2009.10629). The paper-related material is under [this directory](/paper). All the codes and outputs (both intermediate outputs and final outputs) can be found [here](/paper/simulation_study). All the results were run on Compute Canada. The job submission bash scripts contain commands showing the computing resource name and information in the slurm outputs; the seff outputs show the computing time. All studies using GPUs were run on Compute Canada Nvidia A100 GPU(s), which the Compute Canada slurm outputs confirm and show to have a CUDA compute capability of 8.0 ([`cupy.cuda.Device.compute_capability`](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html) returns '80' stands for compute capability of 8.0).

All the `Python` scripts referred in the `bash` job submission scripts were generated from the Jupyter (iPython) notebooks (i.e., `jupyter nbconvert *.ipynb --to python` and move the python script to a separate sub-directory called `dist`) -- **with some parts of codes commented out for some generated `Python` scripts in order to submit different jobs for Compute Canada to run in parallel.** Compute Canada has `slurm-[jobID].out` files consisting of outputs from running the scripts; I also created `seff-[jobID].out` files consisting of the output of `seff [jobID] >> seff-[jobID].output` command to record and report the wall-clock times to finish the computing-time comparison jobs.

With identical simulation setups, under the same $(\epsilon-)$ convergence criteria, seff files show that the computing time simulations for the AG method finished within one day for SCAD or MCP-penalized logistic models; however, the computing time simulations could not finish within the [7-day time limit imposed by Compute Canada Narval cluster](https://docs.alliancecan.ca/wiki/Job_scheduling_policies#Time_limits) for the coordinate descent method on SCAD or MCP-penalized logistic models. Again, all the above simulations were run on [identical GPUs](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en#Available_hardware) -- Nvidia A100 with CUDA compute capability of 8.0. To ensure the fairness of comparison, we coded coordinate descent in `Python`/`CuPy` and compared the computing time with AG -- this was coded based on the state-of-the-art pseudo-code for the coordinate descent method (Breheny & Huang, 2011).


| Model                        	| Penalty 	| Comparison                     	| Optimization Method                             	| Output Data                                                                                                                               	| Jupyter Notebook/R code                                                                                                                              	| Bash Script                                                                                                                                    	| slurm file                                                                                                               	| seff output                                                                                                            	|
|------------------------------	|---------	|--------------------------------	|-------------------------------------------------	|-------------------------------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------	|--------------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------	|
| Penalized Linear Models (LM) 	| SCAD    	| Signal Recovery Performance    	| AG, with strong rule                            	| [`results_SCAD_signal_recovery.npy`](/paper/simulation_study/tasks/task1/results_SCAD_signal_recovery.npy)                                	| [`task1.ipynb`](/paper/simulation_study/tasks/task1/task1.ipynb)                                                                                     	| [`task1.sh`](/paper/simulation_study/tasks/task1/task1.sh)                                                                                     	| [`slurm-10933901.out`](/paper/simulation_study/tasks/task1/slurm-10933901.out)                                           	|                                                                                                                        	|
| Penalized Linear Models (LM) 	| SCAD    	| Signal Recovery Performance    	| coordinate descent (`ncvreg`), with strong rule 	| [`R_results_SCAD_signal_recovery.npy`](/paper/simulation_study/SCAD_MCP/LM/R_results_SCAD_signal_recovery.npy)                            	| [`ncvreg_LM_sim.R`](/paper/simulation_study/SCAD_MCP/LM/ncvreg_LM_sim.R)                                                                             	| [`LM.sh`](/paper/simulation_study/SCAD_MCP/LM/LM.sh)                                                                                           	| [`slurm-10933899.out`](/paper/simulation_study/SCAD_MCP/LM/slurm-10933899.out)                                           	|                                                                                                                        	|
| Penalized Linear Models (LM) 	| SCAD    	| Number of Gradient Evaluations 	| AG, proximal gradient descent                   	| [`SCAD_sim_results.npy`](/paper/simulation_study/tasks/task1speed/SCAD_sim_results.npy)                                                   	| [`task1speed.ipynb`](/paper/simulation_study/tasks/task1speed/task1speed.ipynb)                                                                      	| [`task1speed.sh`](/paper/simulation_study/tasks/task1speed/task1speed.sh)                                                                      	| [`slurm-10933903.out`](/paper/simulation_study/tasks/task1speed/slurm-10933903.out)                                      	| [`seff-10933903.out`](/paper/simulation_study/tasks/task1speed/seff-10933903.out)                                      	|
| Penalized Linear Models (LM) 	| SCAD    	| GPU Computing Time             	| AG, coordinate descent (coded in `Python`)      	| [`SCAD_sim_results.npy`](/paper/simulation_study/tasks/task1speed/SCAD_sim_results.npy)                                                   	| [`task1speed.ipynb`](/paper/simulation_study/tasks/task1speed/task1speed.ipynb)                                                                      	| [`task1speed.sh`](/paper/simulation_study/tasks/task1speed/task1speed.sh)                                                                      	| [`slurm-10933903.out`](/paper/simulation_study/tasks/task1speed/slurm-10933903.out)                                      	| [`seff-10933903.out`](/paper/simulation_study/tasks/task1speed/seff-10933903.out)                                      	|
| Penalized Linear Models (LM) 	| MCP     	| Signal Recovery Performance    	| AG, with strong rule                            	| [`results_MCP_signal_recovery.npy`](/paper/simulation_study/tasks/task1/results_MCP_signal_recovery.npy)                                  	| [`task1.ipynb`](/paper/simulation_study/tasks/task1/task1.ipynb)                                                                                     	| [`task1.sh`](/paper/simulation_study/tasks/task1/task1.sh)                                                                                     	| [`slurm-10933901.out`](/paper/simulation_study/tasks/task1/slurm-10933901.out)                                           	|                                                                                                                        	|
| Penalized Linear Models (LM) 	| MCP     	| Signal Recovery Performance    	| coordinate descent (`ncvreg`), with strong rule 	| [`R_results_MCP_signal_recovery.npy`](/paper/simulation_study/SCAD_MCP/LM/R_results_MCP_signal_recovery.npy)                              	| [`ncvreg_LM_sim.R`](/paper/simulation_study/SCAD_MCP/LM/ncvreg_LM_sim.R)                                                                             	| [`LM.sh`](/paper/simulation_study/SCAD_MCP/LM/LM.sh)                                                                                           	| [`slurm-10933899.out`](/paper/simulation_study/SCAD_MCP/LM/slurm-10933899.out)                                           	|                                                                                                                        	|
| Penalized Linear Models (LM) 	| MCP     	| Number of Gradient Evaluations 	| AG, proximal gradient descent                   	| [`MCP_sim_results.npy`](/paper/simulation_study/tasks/task1speed/MCP_sim_results.npy)                                                     	| [`task1speed.ipynb`](/paper/simulation_study/tasks/task1speed/task1speed.ipynb)                                                                      	| [`task1speed.sh`](/paper/simulation_study/tasks/task1speed/task1speed.sh)                                                                      	| [`slurm-10933903.out`](/paper/simulation_study/tasks/task1speed/slurm-10933903.out)                                      	| [`seff-10933903.out`](/paper/simulation_study/tasks/task1speed/seff-10933903.out)                                      	|
| Penalized Linear Models (LM) 	| MCP     	| GPU Computing Time             	| AG, coordinate descent (coded in `Python`)      	| [`MCP_sim_results.npy`](/paper/simulation_study/tasks/task1speed/MCP_sim_results.npy)                                                     	| [`task1speed.ipynb`](/paper/simulation_study/tasks/task1speed/task1speed.ipynb)                                                                      	| [`task1speed.sh`](/paper/simulation_study/tasks/task1speed/task1speed.sh)                                                                      	| [`slurm-10933903.out`](/paper/simulation_study/tasks/task1speed/slurm-10933903.out)                                      	| [`seff-10933903.out`](/paper/simulation_study/tasks/task1speed/seff-10933903.out)                                      	|
| Penalized Logistic Models    	| SCAD    	| Signal Recovery Performance    	| AG, with strong rule                            	| [`results_SCAD_signal_recovery.npy`](/paper/simulation_study/tasks/task2/results_SCAD_signal_recovery.npy)                                	| [`task2.ipynb`](/paper/simulation_study/tasks/task2/task2.ipynb)                                                                                     	| [`task2.sh`](/paper/simulation_study/tasks/task2/task2.sh)                                                                                     	| [`slurm-10933902.out`](/paper/simulation_study/tasks/task2/slurm-10933902.out)                                           	|                                                                                                                        	|
| Penalized Logistic Models    	| SCAD    	| Signal Recovery Performance    	| coordinate descent (`ncvreg`), with strong rule 	| [`R_results_SCAD_signal_recovery.npy`](/paper/simulation_study/SCAD_MCP/logistic/R_results_SCAD_signal_recovery.npy)                      	| [`ncvreg_logistic_sim.R`](/paper/simulation_study/SCAD_MCP/logistic/ncvreg_logistic_sim.R)                                                           	| [`logistic.sh`](/paper/simulation_study/SCAD_MCP/logistic/logistic.sh)                                                                         	| [`slurm-10933900.out`](/paper/simulation_study/SCAD_MCP/logistic/slurm-10933900.out)                                     	|                                                                                                                        	|
| Penalized Logistic Models    	| SCAD    	| Number of Gradient Evaluations 	| AG, proximal gradient descent                   	| [`SCAD_sim_results.npy`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD/SCAD_sim_results.npy)                         	| [`task2speed_SCAD.ipynb`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD/task2speed_SCAD.ipynb)                                  	| [`task2speed_SCAD.sh`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD/task2speed_SCAD.sh)                                  	| [`slurm-10933908.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD/slurm-10933908.out)            	| [`seff-10933908.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD/seff-10933908.out)            	|
| Penalized Logistic Models    	| SCAD    	| GPU Computing Time             	| AG                                              	| [`SCAD_sim_results_AG_time.npy`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_AG_time/SCAD_sim_results_AG_time.npy) 	| [`task2speed_SCAD_AG_time.ipynb`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_AG_time/task2speed_SCAD_AG_time.ipynb)          	| [`task2speed_SCAD_AG_time.sh`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_AG_time/task2speed_SCAD_AG_time.sh)          	| [`slurm-10933906.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_AG_time/slurm-10933906.out)    	| [`seff-10933906.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_AG_time/seff-10933906.out)    	|
| Penalized Logistic Models    	| SCAD    	| GPU Computing Time             	| coordinate descent (coded in `Python`)          	|                                                                                                                                           	| [`task2speed_SCAD_coord_time.ipynb`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_coord_time/task2speed_SCAD_coord_time.ipynb) 	| [`task2speed_SCAD_coord_time.sh`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_coord_time/task2speed_SCAD_coord_time.sh) 	| [`slurm-10933904.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_coord_time/slurm-10933904.out) 	| [`seff-10933904.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_SCAD_coord_time/seff-10933904.out) 	|
| Penalized Logistic Models    	| MCP     	| Signal Recovery Performance    	| AG, with strong rule                            	| [`results_MCP_signal_recovery.npy`](/paper/simulation_study/tasks/task2/results_MCP_signal_recovery.npy)                                  	| [`task2.ipynb`](/paper/simulation_study/tasks/task2/task2.ipynb)                                                                                     	| [`task2.sh`](/paper/simulation_study/tasks/task2/task2.sh)                                                                                     	| [`slurm-10933902.out`](/paper/simulation_study/tasks/task2/slurm-10933902.out)                                           	|                                                                                                                        	|
| Penalized Logistic Models    	| MCP     	| Signal Recovery Performance    	| coordinate descent (`ncvreg`), with strong rule 	| [`R_results_MCP_signal_recovery.npy`](/paper/simulation_study/SCAD_MCP/logistic/R_results_MCP_signal_recovery.npy)                        	| [`ncvreg_logistic_sim.R`](/paper/simulation_study/SCAD_MCP/logistic/ncvreg_logistic_sim.R)                                                           	| [`logistic.sh`](/paper/simulation_study/SCAD_MCP/logistic/logistic.sh)                                                                         	| [`slurm-10933900.out`](/paper/simulation_study/SCAD_MCP/logistic/slurm-10933900.out)                                     	|                                                                                                                        	|
| Penalized Logistic Models    	| MCP     	| Number of Gradient Evaluations 	| AG, proximal gradient descent                   	| [`MCP_sim_results.npy`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP/MCP_sim_results.npy)                            	| [`task2speed_MCP.ipynb`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP/task2speed_MCP.ipynb)                                     	| [`task2speed_MCP.sh`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP/task2speed_MCP.sh)                                     	| [`slurm-10933909.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP/slurm-10933909.out)             	| [`seff-10933909.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP/seff-10933909.out)             	|
| Penalized Logistic Models    	| MCP     	| GPU Computing Time             	| AG                                              	| [`MCP_sim_results_AG_time.npy`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_AG_time/MCP_sim_results_AG_time.npy)    	| [`task2speed_MCP_AG_time.ipynb`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_AG_time/task2speed_MCP_AG_time.ipynb)             	| [`task2speed_MCP_AG_time.sh`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_AG_time/task2speed_MCP_AG_time.sh)             	| [`slurm-10933907.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_AG_time/slurm-10933907.out)     	| [`seff-10933907.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_AG_time/seff-10933907.out)     	|
| Penalized Logistic Models    	| MCP     	| GPU Computing Time             	| coordinate descent (coded in `Python`)          	|                                                                                                                                           	| [`task2speed_MCP_coord_time.ipynb`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_coord_time/task2speed_MCP_coord_time.ipynb)    	| [`task2speed_MCP_coord_time.sh`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_coord_time/task2speed_MCP_coord_time.sh)    	| [`slurm-10933905.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_coord_time/slurm-10933905.out)  	| [`seff-10933905.out`](/paper/simulation_study/tasks/task2speed/sub_tasks/task2speed_MCP_coord_time/seff-10933905.out)  	|


- **The summary of the simulation results is [in this Jupyter notebook](/paper/simulation_study/summary.ipynb),** where most of the simulation study results in the paper are from. 
- The two very original Jupyter notebooks are [the simulation study for SCAD/MCP-penalized linear models](/paper/simulation_study/LM_SCAD_MCP_cp_(cupy).ipynb) and [the simulation study for SCAD/MCP-penalized logistic models](/paper/simulation_study/logistic_SCAD_MCP_cp_(cupy).ipynb) -- all other notebooks and `Python` codes are generated and modified based on them. 
- To run on the server, I divided the codes into several chunks; they were in [this folder](/paper/simulation_study/tasks) *for python simulations*
  *  [`task1`](/paper/simulation_study/tasks/task1) and [`task2`](/paper/simulation_study/tasks/task2) contain files to test signal recovery performance for SCAD/MCP-penalized linear models and logistic models using AG; 
  *  [`task1speed`](/paper/simulation_study/tasks/task1speed) and [`task2speed`](/paper/simulation_study/tasks/task2speed) contain files to test $(\epsilon-)$ convergence speed and computing times for SCAD/MCP-penalized linear models and logistic models using AG v.s. proximal gradient v.s. coordinate descent. 
  *  The R codes and results for `ncvreg` simulations are contained in [this directory](/paper/simulation_study/SCAD_MCP) -- click [here for penalized linear models](/paper/simulation_study/SCAD_MCP/LM) or [here for penalized logistic models](/paper/simulation_study/SCAD_MCP/logistic).

Some algebra calculations from the paper can be found at [this SageMath notebook](/paper/SageMath_algebra.ipynb); the MATLAB codes to generate plots are [here fore Figure 1](/paper/optimize_b_k.m).


<!-- The manual for the PyPI package [`nonconvexAG`](https://pypi.org/project/nonconvexAG/) can be found [here](/nonconvexAG/README.md). -->

# Bibliography

- Breheny, P., & Huang, J. (2011). Coordinate descent algorithms for nonconvex penalized regression, with applications to biological feature selection. Annals of Applied Statistics 2011, Vol. 5, No. 1, 232-253. [https://doi.org/10.1214/10-AOAS388](https://doi.org/10.1214/10-AOAS388)
