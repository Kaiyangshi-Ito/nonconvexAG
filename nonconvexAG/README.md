# nonconvexAG

This is an implementation of restarting accelerated gradient algorithm with strong rules for (high-dimensional) nonconvex sparse learning problems. The corresponding paper can be found at [arXiv](https://arxiv.org/abs/2009.10629).

The available functions are:
- `UAG_LM_SCAD_MCP`, `UAG_logistic_SCAD_MCP`: these functions find a local minizer for the SCAD/MCP penalized linear models/logistic models. The arguments are:
        * `design_matrix`: the design matrix input, should be a two-dimensional numpy array;
        * `outcome`: the outcome, should be one dimensional numpy array, continuous for linear model, binary for logistic model;
        * `beta_0`: starting value; optional, if not declared, it will be calculated based on the Gauss-Markov theory estimators of $\beta$;
        * `tol`: tolerance parameter; the tolerance parameter is set to be the uniform norm of two iterations;
        * `maxit`: maximum number of iteratios allowed;
        * `_lambda`: _lambda value;
        * `penalty`: could be `"SCAD"` or `"MCP"`;
        * `a=3.7`, `gamma=2`: `a` for SCAD and `gamma` for MCP; it is recommended for `a` to be set as $3.7$;
        * `L_convex`: the L-smoothness constant for the convex component, if not declared, it will be calculated by itself
        * `add_intercept_column`: boolean, should the fucntion add an intercept column?

- `solution_path_LM`, `solution_path_logistic`: calculate the solution path for linear/logistic models; the only difference from above is that `lambda_` is now a one-dimensional numpy array for the values of $\lambda$ to be used.

- `UAG_LM_SCAD_MCP_strongrule`, `UAG_logistic_SCAD_MCP_strongrule` work just like `UAG_LM_SCAD_MCP`, `UAG_logistic_SCAD_MCP` -- except they use strong rule to filter out many covariates before carrying out the optimization step. Same for `solution_path_LM_strongrule` and `solution_path_logistic_strongrule`. Strong rule increases the computational speed dramatically.

- The package also offers implementation of certain features using memory mapping. `memmap_lambda_max_LM` and `memmap_lambda_max_logistic` calculate the least values for *\lambda* to vanish all penalized coefficients. `memmap_UAG_LM_SCAD_MCP`, `memmap_UAG_logisitc_SCAD_MCP`, and `memmap_solution_path_LM`, `memmap_solution_path_LM` work similar to the non-memorymapping version of the function. **For memory mapping versions of the functions, the `design_matrix` or `X` parameter should declare the path of the memorymapped file, and `_dtype='float32'` declares the data type, `_order` declares the order of the memorymapped files ("F" for Fortran or "C" for C++).** Multiprocess with memory mapping is also available as `memmap_lambda_max_LM_parallel`, `memmap_lambda_max_logistic_parallel`, `memmap_UAG_LM_SCAD_MCP_parallel`, `memmap_UAG_logistic_SCAD_MCP_parallel`, `memmap_solution_path_LM_parallel`, `memmap_solution_path_logistic_parallel` -- for these functions, an extra argument `core_num` can be used to declare the cores to be used -- if not decalred, it will use all the cores.
