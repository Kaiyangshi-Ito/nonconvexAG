#!/usr/bin/env python
# coding: utf-8

# # Set up the class fundementals 

# In[1]:


import os, sys
import collections
import numpy as np
import cupy as cp
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import timeit
from scipy.stats import logistic
from cupyx.scipy.linalg import toeplitz, block_diag
from scipy.stats import median_abs_deviation as mad
import multiprocessing
import cProfile
import itertools
import warnings

warnings.filterwarnings('ignore')  # this is just to hide all the warnings
import rpy2.robjects as robjects

import matplotlib.pyplot as plt  # change font globally to Times

plt.style.use('ggplot')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": ["Times New Roman"],
    "font.size": 12
})

os.chdir(sys.path[0])  # ensure working direcotry is set same as the file


# In[2]:


print("This script gets the running time for accelerated gradient running on MCP-penalized logistic models.")
print("The installed CuPy version is:", cp.__version__)
print("The CUDA device has compute capability of:",
      cp.cuda.device.get_compute_capability())


# In[ ]:


class tensor_computation:
    '''
    This is just a fundemental class used for tensor computation;
    its main purpose here is to serve as a class which LMM_SCAD_MCP can inherent from later. 
    '''
    def __str__(self):
        '''
        Basic declaration for class. 
        '''
        return "This is just a fundemental linear algebra computation setup for class creation in the future steps"

    ######################################  some SCAD and MCP things  #######################################
    def soft_thresholding(self, x, lambda_):
        '''
        To calculate soft-thresholding mapping of a given ONE-DIMENSIONAL tensor, BESIDES THE FIRST TERM (so beta_0 will not be penalized). 
        This function is to be used for calculation involving L1 penalty term later. 
        '''
        return cp.concatenate((cp.array([x[0]]),
                               cp.where(
                                   cp.abs(x[1:]) > lambda_,
                                   x[1:] - cp.sign(x[1:]) * lambda_, 0)))

    def soft_thresholding_scalar(self, x, lambda_):
        '''
        To calculate soft-thresholding mapping of a given ONE-DIMENSIONAL tensor, BESIDES THE FIRST TERM (so beta_0 will not be penalized). 
        This function is to be used for calculation involving L1 penalty term later. 
        '''
        return cp.where(cp.abs(x) > lambda_, x - cp.sign(x) * lambda_, 0)

    def SCAD(self, x, lambda_, a=3.7):
        '''
        To calculate SCAD penalty value;
        #x can be a multi-dimensional tensor;
        lambda_, a are scalars;
        Fan and Li suggests to take a as 3.7 
        '''
        # here I notice the function is de facto a function of absolute value of x, therefore take absolute value first to simplify calculation
        x = cp.abs(x)
        temp = cp.where(
            x <= lambda_, lambda_ * x,
            cp.where(x < a * lambda_,
                     (2 * a * lambda_ * x - x**2 - lambda_**2) / (2 * (a - 1)),
                     lambda_**2 * (a + 1) / 2))
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp

    def SCAD_grad(self, x, lambda_, a=3.7):
        '''
        To calculate the gradient of SCAD wrt. input x; 
        #x can be a multi-dimensional tensor. 
        '''
        # here decompose x to sign and its absolute value for easier calculation
        sgn = cp.sign(x)
        x = cp.abs(x)
        temp = cp.where(
            x <= lambda_, lambda_ * sgn,
            cp.where(x < a * lambda_, (a * lambda_ * sgn - sgn * x) / (a - 1),
                     0))
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp

    def MCP(self, x, lambda_, gamma):
        '''
        To calculate MCP penalty value; 
        #x can be a multi-dimensional tensor. 
        '''
        # the function is a function of absolute value of x
        x = cp.abs(x)
        temp = cp.where(x <= gamma * lambda_, lambda_ * x - x**2 / (2 * gamma),
                        .5 * gamma * lambda_**2)
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp

    def MCP_grad(self, x, lambda_, gamma):
        '''
        To calculate MCP gradient wrt. input x; 
        #x can be a multi-dimensional tensor. 
        '''
        temp = cp.where(
            cp.abs(x) < gamma * lambda_,
            lambda_ * cp.sign(x) - x / gamma, cp.zeros_like(x))
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp

    def SCAD_concave(self, x, lambda_, a=3.7):
        '''
        The value of concave part of SCAD penalty; 
        #x can be a multi-dimensional tensor. 
        '''
        x = cp.abs(x)
        temp = cp.where(
            x <= lambda_, 0.,
            cp.where(x < a * lambda_,
                     (lambda_ * x - (x**2 + lambda_**2) / 2) / (a - 1),
                     (a + 1) / 2 * lambda_**2 - lambda_ * x))
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp

    def SCAD_concave_grad(self, x, lambda_, a=3.7):
        '''
        The gradient of concave part of SCAD penalty wrt. input x; 
        #x can be a multi-dimensional tensor. 
        '''
        sgn = cp.sign(x)
        x = cp.abs(x)
        temp = cp.where(
            x <= lambda_, 0.,
            cp.where(x < a * lambda_, (lambda_ * sgn - sgn * x) / (a - 1),
                     -lambda_ * sgn))
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp

    def MCP_concave(self, x, lambda_, gamma):
        '''
        The value of concave part of MCP penalty; 
        #x can be a multi-dimensional tensor. 
        '''
        # similiar as in MCP
        x = cp.abs(x)
        temp = cp.where(x <= gamma * lambda_, -(x**2) / (2 * gamma),
                        (gamma * lambda_**2) / 2 - lambda_ * x)
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp

    def MCP_concave_grad(self, x, lambda_, gamma):
        '''
        The gradient of concave part of MCP penalty wrt. input x; 
        #x can be a multi-dimensional tensor. 
        '''
        temp = cp.where(
            cp.abs(x) < gamma * lambda_, -x / gamma, -lambda_ * cp.sign(x))
        temp[0] = 0.  # this is to NOT penalize intercept beta later
        return temp


# # Graphical Illustrations for SCAD and MCP 

# In[ ]:


x = cp.arange(-15.1, 15.2, .1)
class_temp = tensor_computation()

plt.plot(x[1:-1].get(), class_temp.soft_thresholding(x, lambda_=2)[1:-1].get())
plt.xlabel(r'$x$')
plt.ylabel(r'soft-thresholding when $\lambda$=2')

plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x[1:-1].get(),
               class_temp.SCAD(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 0].plot(x[1:-1].get(),
               class_temp.MCP(x, lambda_=2, gamma=1)[1:-1].get(),
               label="MCP")
axs[0, 0].plot(x[1:-1].get(), np.abs(2 * x)[1:-1].get(), label="LASSO")
axs[0, 0].set_title(r'$\lambda=2,a=3.7,\gamma=1$')
axs[0, 1].plot(x[1:-1].get(),
               class_temp.SCAD(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 1].plot(x[1:-1].get(),
               class_temp.MCP(x, lambda_=2, gamma=3.7)[1:-1].get(),
               label="MCP")
axs[0, 1].plot(x[1:-1].get(), np.abs(2 * x)[1:-1].get(), label="LASSO")
axs[0, 1].set_title(r'$\lambda=2,a=3.7,\gamma=3.7$')
axs[1, 0].plot(x[1:-1].get(),
               class_temp.SCAD(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 0].plot(x[1:-1].get(),
               class_temp.MCP(x, lambda_=2, gamma=4.7)[1:-1].get(),
               label="MCP")
axs[1, 0].plot(x[1:-1].get(), np.abs(2 * x)[1:-1].get(), label="LASSO")
axs[1, 0].set_title(r'$\lambda=2,a=3.7,\gamma=4.7$')
axs[1, 1].plot(x[1:-1].get(),
               class_temp.SCAD(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 1].plot(x[1:-1].get(),
               class_temp.MCP(x, lambda_=2, gamma=2.7)[1:-1].get(),
               label="MCP")
axs[1, 1].plot(x[1:-1].get(), np.abs(2 * x)[1:-1].get(), label="LASSO")
axs[1, 1].set_title(r'$\lambda=2,a=3.7,\gamma=2.7$')

for ax in axs.flat:
    ax.set(xlabel=r'$\theta$', ylabel='penalty')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.legend(loc='center left',
           bbox_to_anchor=(1, 1.05),
           ncol=1,
           fancybox=True,
           shadow=True)
# plt.savefig('SCAD_MCP.eps', format='eps', bbox_inches='tight')
plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x[1:-1].get(),
               class_temp.SCAD_concave(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 0].plot(x[1:-1].get(),
               class_temp.MCP_concave(x, lambda_=2, gamma=1)[1:-1].get(),
               label="MCP")
axs[0, 0].set_title(r'$\lambda=2,a=3.7,\gamma=1$')
axs[0, 1].plot(x[1:-1].get(),
               class_temp.SCAD_concave(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 1].plot(x[1:-1].get(),
               class_temp.MCP_concave(x, lambda_=2, gamma=3.7)[1:-1].get(),
               label="MCP")
axs[0, 1].set_title(r'$\lambda=2,a=3.7,\gamma=3.7$')
axs[1, 0].plot(x[1:-1].get(),
               class_temp.SCAD_concave(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 0].plot(x[1:-1].get(),
               class_temp.MCP_concave(x, lambda_=2, gamma=4.7)[1:-1].get(),
               label="MCP")
axs[1, 0].set_title(r'$\lambda=2,a=3.7,\gamma=4.7$')
axs[1, 1].plot(x[1:-1].get(),
               class_temp.SCAD_concave(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 1].plot(x[1:-1].get(),
               class_temp.MCP_concave(x, lambda_=2, gamma=2.7)[1:-1].get(),
               label="MCP")
axs[1, 1].set_title(r'$\lambda=2,a=3.7,\gamma=2.7$')

for ax in axs.flat:
    ax.set(xlabel=r'$\theta$', ylabel='concave part')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.legend(loc='center left',
           bbox_to_anchor=(1, 1.05),
           ncol=1,
           fancybox=True,
           shadow=True)
# plt.savefig('SCAD_MCP_concave.eps', format='eps', bbox_inches='tight')
plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x[1:-1].get(),
               class_temp.SCAD_concave_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 0].plot(x[1:-1].get(),
               class_temp.MCP_concave_grad(x, lambda_=2, gamma=1)[1:-1].get(),
               label="MCP")
axs[0, 0].set_title(r'$\lambda=2,a=3.7,\gamma=1$')
axs[0, 1].plot(x[1:-1].get(),
               class_temp.SCAD_concave_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 1].plot(x[1:-1].get(),
               class_temp.MCP_concave_grad(x, lambda_=2,
                                           gamma=3.7)[1:-1].get(),
               label="MCP")
axs[0, 1].set_title(r'$\lambda=2,a=3.7,\gamma=3.7$')
axs[1, 0].plot(x[1:-1].get(),
               class_temp.SCAD_concave_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 0].plot(x[1:-1].get(),
               class_temp.MCP_concave_grad(x, lambda_=2,
                                           gamma=4.7)[1:-1].get(),
               label="MCP")
axs[1, 0].set_title(r'$\lambda=2,a=3.7,\gamma=4.7$')
axs[1, 1].plot(x[1:-1].get(),
               class_temp.SCAD_concave_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 1].plot(x[1:-1].get(),
               class_temp.MCP_concave_grad(x, lambda_=2,
                                           gamma=2.7)[1:-1].get(),
               label="MCP")
axs[1, 1].set_title(r'$\lambda=2,a=3.7,\gamma=2.7$')

for ax in axs.flat:
    ax.set(xlabel=r'$\theta$', ylabel='concavity gradient')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.legend(loc='center left',
           bbox_to_anchor=(1, 1.05),
           ncol=1,
           fancybox=True,
           shadow=True)
plt.show()


# In[ ]:


# To plot the derivatives of SCAD and MCP
x[np.abs(x) < 1e-10] = float("nan")
markerstyle = {
    "markersize": 8,
    "markeredgecolor": "black",
    "markerfacecolor": "w",
    "linestyle": "none"
}
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x[1:-1].get(),
               class_temp.SCAD_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 0].plot(x[1:-1].get(),
               class_temp.MCP_grad(x, lambda_=2, gamma=1)[1:-1].get(),
               label="MCP")
axs[0, 0].plot([0, 0], [-2, 2], marker=".", **markerstyle)
axs[0, 0].set_title(r'$\lambda=2,a=3.7,\gamma=1$')
axs[0, 1].plot(x[1:-1].get(),
               class_temp.SCAD_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[0, 1].plot(x[1:-1].get(),
               class_temp.MCP_grad(x, lambda_=2, gamma=3.7)[1:-1].get(),
               label="MCP")
axs[0, 1].plot([0, 0], [-2, 2], marker=".", **markerstyle)
axs[0, 1].set_title(r'$\lambda=2,a=3.7,\gamma=3.7$')
axs[1, 0].plot(x[1:-1].get(),
               class_temp.SCAD_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 0].plot(x[1:-1].get(),
               class_temp.MCP_grad(x, lambda_=2, gamma=4.7)[1:-1].get(),
               label="MCP")
axs[1, 0].plot([0, 0], [-2, 2], marker=".", **markerstyle)
axs[1, 0].set_title(r'$\lambda=2,a=3.7,\gamma=4.7$')
axs[1, 1].plot(x[1:-1].get(),
               class_temp.SCAD_grad(x, lambda_=2, a=3.7)[1:-1].get(),
               label="SCAD")
axs[1, 1].plot(x[1:-1].get(),
               class_temp.MCP_grad(x, lambda_=2, gamma=2.7)[1:-1].get(),
               label="MCP")
axs[1, 1].plot([0, 0], [-2, 2], marker=".", **markerstyle)
axs[1, 1].set_title(r'$\lambda=2,a=3.7,\gamma=2.7$')

for ax in axs.flat:
    ax.set(xlabel=r'$\theta$', ylabel='derivative')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.legend(loc='center left',
           bbox_to_anchor=(1, 1.05),
           ncol=1,
           fancybox=True,
           shadow=True)
plt.show()


# # Implementation
# 
# 
# The negative log-likelihood loss for logistic is: 
# 
# $$\frac{\left(\sum_{i}\log\left(\exp\left(\mathbf{X}\boldsymbol{\beta}\right)+\mathbf{1}\right)_{i}\right)-\mathbf{y}^{T}\mathbf{X}\boldsymbol{\beta}}{2N}$$
# 
# its gradient is then: 
# 
# $$\frac{\mathbf{X}^{T}\left(\sigma\left(\mathbf{X}\boldsymbol{\beta}\right)-\mathbf{y}\right)}{2N}$$

# In[ ]:


class logistic_SCAD_MCP(tensor_computation):
    '''
    This class performs SCAD/MCP pruning on linear models.
    '''
    def __init__(self,
                 design_matrix,
                 outcome,
                 penalty,
                 _lambda,
                 a=3.7,
                 gamma=2,
                 beta_0="NOT DECLARED",
                 tol=1e-4,
                 maxit=5000,
                 L_convex="NOT DECLARED"):
        '''
        Class constructor:
        design_matrix:           the design matrix for the linear models;
        outcome:                 the outcome for the linear model;
        penalty:                 "SCAD" or "MCP";
        _lambda:                 value for lambda; 
        a:                       value for a, only used when penalty set to be "SCAD";
        gamma:                   value for gamma, only used when penalty set to be "MCP";
        beta_0:                  initial values for beta;
        tol:                     tolerance parameter set for beta, this is for the maximum change of beta;
        maxit:                   maximum number of iterations allowed;
        '''
        assert penalty in (
            "SCAD", "MCP"), "Choose between \"SCAD\" or \"MCP\" for penalty"
        assert a > 2, "SCAD penalty parameter, a, needs to be greater than 2"
        assert gamma > 0, "MCP penalty parameter, gamma, needs to be positive"
        assert _lambda > 0, "penalty paramter, lambda_, needs to be positive"
        assert tol > 0, "tol should be postive"
        assert maxit > 0, "maxit is the maximum iteration allowed; which needs to be positive"
        # Construct self
        self.X = design_matrix
        self.y = outcome
        self.N = self.X.shape[0]
        #        cov = (self.y - cp.mean(self.y))@(self.X - cp.mean(self.X, 0).reshape(1,-1))
        if type(beta_0) == str:
            self.beta = cp.zeros(self.X.shape[1])  #cp.sign(cov)
        else:
            self.beta = beta_0
        # add design matrix column for the intercept, if it's not there already
        if cp.any(
                cp.all(self.X == self.X[0, :], 0)
        ):  # check if design matrix has included a column for intercept or not
            pass
        else:
            intercept_design = cp.ones(self.N).reshape(self.N, 1)
            self.X = cp.concatenate((intercept_design, self.X), 1)
            if type(beta_0) == str:
                self.beta = cp.concatenate((cp.array([0.]), self.beta))
        # passing other parameters
        self.tol = tol
        self.maxit = maxit
        self._lambda = _lambda
        self.penalty = penalty
        #        if penalty == "SCAD":
        self.a = a
        #        else:
        self.gamma = gamma
        self.p = self.X.shape[
            1]  # so here p includes the intercept design matrix column
        self.smooth_grad = cp.ones(self.p)
        self.beta_ag = self.beta.copy()
        self.beta_md = self.beta.copy()
        self.k = 0
        self.FISTA_k = 0
        self.converged = False
        self.obj_value = []
        self.obj_value_ORIGINAL = []
        self.obj_value_AG = []
        self.obj_coord_value = []
        self.opt_alpha = 1
        if type(L_convex) == str:
            self.L_convex = 1 / (8 * self.N) * cp.max(
                cp.linalg.eigh(self.X @ self.X.T)[0]).item()
        else:
            self.L_convex = L_convex
        self.FISTA_beta = cp.empty_like(self.beta)

    def update_smooth_grad_convex(self):
        '''
        Update the gradient of the smooth convex objective component.
        '''
        self.smooth_grad = (self.X.T @ (np.tanh(self.X @ self.beta_md / 2) / 2
                                        - self.y + .5)) / (2 * self.N)

    def update_smooth_grad_SCAD(self):
        '''
        Update the gradient of the smooth objective component for SCAD penalty.
        '''
        self.update_smooth_grad_convex()
        self.smooth_grad += self.SCAD_concave_grad(self.beta_md,
                                                   lambda_=self._lambda,
                                                   a=self.a)

    def update_smooth_grad_MCP(self):
        '''
        Update the gradient of the smooth objective component for MCP penalty.
        '''
        self.update_smooth_grad_convex()
        self.smooth_grad += self.MCP_concave_grad(self.beta_md,
                                                  lambda_=self._lambda,
                                                  gamma=self.gamma)

    def eval_obj_SCAD(self, x_temp, obj_value_name):
        '''
        evaluate value of the objective function.
        '''
        obj_value_name += [
            (-self.y.T @ self.X @ self.beta_md +
             cp.sum(cp.logaddexp(self.X @ self.beta_md, 0))) / (2 * self.N) +
            cp.sum(self.SCAD(self.beta_md, self._lambda, self.a))
        ]

    def eval_obj_MCP(self, x_temp, obj_value_name):
        '''
        evaluate value of the objective function.
        '''
        obj_value_name += [
            (-self.y.T @ self.X @ self.beta_md +
             cp.sum(cp.logaddexp(self.X @ self.beta_md, 0))) / (2 * self.N) +
            cp.sum(self.MCP(self.beta_md, self._lambda, self.a))
        ]

    def UAG_logistic_SCAD_MCP(self):
        '''
        Carry out the optimization.
        '''
        if self.penalty == "SCAD":
            L = max([self.L_convex, 1 / (self.a - 1)])
            self.opt_beta = .99 / L
            self.eval_obj_SCAD(self.beta_md, self.obj_value)
            self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    #self.opt_alpha = 2/(self.k+1)**0.3 #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    #self.opt_lambda = self.k/2*self.opt_beta #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    self.opt_alpha = 2 / (
                        1 + cp.sqrt(1 + 4 / self.opt_alpha**2)
                    )  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.opt_lambda = self.opt_beta / self.opt_alpha  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.beta_md = (
                        1 - self.opt_alpha
                    ) * self.beta_ag + self.opt_alpha * self.beta
                    self.update_smooth_grad_SCAD()
                    self.beta = self.soft_thresholding(
                        self.beta - self.opt_lambda * self.smooth_grad,
                        self.opt_lambda * self._lambda)
                    self.beta_ag = self.soft_thresholding(
                        self.beta_md - self.opt_beta * self.smooth_grad,
                        self.opt_beta * self._lambda)
                    self.converged = cp.all(
                        cp.max(
                            cp.abs(self.beta_md - self.beta_ag) /
                            self.opt_beta) < self.tol).item()
                    self.eval_obj_SCAD(self.beta_md, self.obj_value)
                    self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
                else:
                    break
        else:
            L = max([self.L_convex, 1 / self.gamma])
            self.opt_beta = .99 / L
            self.eval_obj_MCP(self.beta_md, self.obj_value)
            self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    #self.opt_alpha = 2/(self.k+1) #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    #self.opt_lambda = self.k/2*self.opt_beta #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    self.opt_alpha = 2 / (
                        1 + cp.sqrt(1 + 4 / self.opt_alpha**2)
                    )  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.opt_lambda = self.opt_beta / self.opt_alpha  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper
                    self.beta_md = (
                        1 - self.opt_alpha
                    ) * self.beta_ag + self.opt_alpha * self.beta
                    self.update_smooth_grad_MCP()
                    self.beta = self.soft_thresholding(
                        self.beta - self.opt_lambda * self.smooth_grad,
                        self.opt_lambda * self._lambda)
                    self.beta_ag = self.soft_thresholding(
                        self.beta_md - self.opt_beta * self.smooth_grad,
                        self.opt_beta * self._lambda)
                    self.converged = cp.all(
                        cp.max(cp.abs(self.beta_md - self.beta_ag)) /
                        self.opt_beta < self.tol).item()
                    self.eval_obj_MCP(self.beta_md, self.obj_value)
                    self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
                else:
                    break
        return self.report_results()

    def UAG_restarting_logistic_SCAD_MCP(self):
        '''
        Carry out the optimization.
        '''
        self.old_speed_norm = 0.
        self.speed_norm = 1.
        self.restart_k = 0
        if self.penalty == "SCAD":
            L = max([self.L_convex, 1 / (self.a - 1)])
            self.opt_beta = .99 / L
            self.eval_obj_SCAD(self.beta_md, self.obj_value)
            self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    if self.old_speed_norm > self.speed_norm and self.k - self.restart_k >= 3:
                        self.opt_alpha = 1.
                        self.restart_k = self.k
                        print("restarting at iteration:", self.restart_k)
                    else:
                        pass
                        #self.opt_alpha = 2/(self.k+1)**0.3 #parameter setting based on Ghadimi and Lan's exemplified Lemma
                        #self.opt_lambda = self.k/2*self.opt_beta #parameter setting based on Ghadimi and Lan's exemplified Lemma
                        self.opt_alpha = 2 / (
                            1 + cp.sqrt(1 + 4 / self.opt_alpha**2)
                        )  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.opt_lambda = self.opt_beta / self.opt_alpha  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.beta_md_old = self.beta_md.copy()
                    self.beta_md = (
                        1 - self.opt_alpha
                    ) * self.beta_ag + self.opt_alpha * self.beta
                    self.old_speed_norm = self.speed_norm
                    self.speed_norm = cp.linalg.norm(self.beta_md -
                                                     self.beta_md_old)
                    self.update_smooth_grad_SCAD()
                    self.beta = self.soft_thresholding(
                        self.beta - self.opt_lambda * self.smooth_grad,
                        self.opt_lambda * self._lambda)
                    self.beta_ag = self.soft_thresholding(
                        self.beta_md - self.opt_beta * self.smooth_grad,
                        self.opt_beta * self._lambda)
                    self.converged = cp.all(
                        cp.max(
                            cp.abs(self.beta_md - self.beta_ag) /
                            self.opt_beta) < self.tol).item()
                    self.eval_obj_SCAD(self.beta_md, self.obj_value)
                    self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
                else:
                    break
        else:
            L = max([self.L_convex, 1 / (self.a - 1)])
            self.opt_beta = .99 / L
            self.eval_obj_SCAD(self.beta_md, self.obj_value)
            self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    if self.old_speed_norm > self.speed_norm and self.k - self.restart_k > 3:
                        self.opt_alpha = 1.
                        self.restart_k = self.k
                        print("restarting at iteration:", self.restart_k)
                    else:
                        pass
                        #self.opt_alpha = 2/(self.k+1)**0.3 #parameter setting based on Ghadimi and Lan's exemplified Lemma
                        #self.opt_lambda = self.k/2*self.opt_beta #parameter setting based on Ghadimi and Lan's exemplified Lemma
                        self.opt_alpha = 2 / (
                            1 + cp.sqrt(1 + 4 / self.opt_alpha**2)
                        )  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.opt_lambda = self.opt_beta / self.opt_alpha  #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.beta_md_old = self.beta_md.copy()
                    self.beta_md = (
                        1 - self.opt_alpha
                    ) * self.beta_ag + self.opt_alpha * self.beta
                    self.old_speed_norm = self.speed_norm
                    self.speed_norm = cp.linalg.norm(self.beta_md -
                                                     self.beta_md_old)
                    self.update_smooth_grad_MCP()
                    self.beta = self.soft_thresholding(
                        self.beta - self.opt_lambda * self.smooth_grad,
                        self.opt_lambda * self._lambda)
                    self.beta_ag = self.soft_thresholding(
                        self.beta_md - self.opt_beta * self.smooth_grad,
                        self.opt_beta * self._lambda)
                    self.converged = cp.all(
                        cp.max(
                            cp.abs(self.beta_md - self.beta_ag) /
                            self.opt_beta) < self.tol).item()
                    self.eval_obj_MCP(self.beta_md, self.obj_value)
                    self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
                else:
                    break
        return self.report_results()

    def FISTA_logistic_SCAD_MCP(self):
        '''
        Carry out FISTA procedure to find out l1 penalized minimizer.
        '''
        FISTA_t_new = 1.
        FISTA_converged = False
        x_new, x_old = cp.empty_like(self.beta_md), cp.empty_like(
            self.beta_md
        )  # beta_md here is y; it plays a very different role here!
        self.update_smooth_grad_convex()
        x_new = self.soft_thresholding(
            self.beta_md - self.smooth_grad / self.L_convex,
            self._lambda / self.L_convex)
        if self.penalty == "SCAD":
            self.eval_obj_SCAD(x_new, self.obj_value)
        else:
            self.eval_obj_MCP(x_new, self.obj_value)
        while (not FISTA_converged) and self.FISTA_k <= self.maxit:
            self.FISTA_k += 1
            self.update_smooth_grad_convex()
            x_old = x_new.copy()
            x_new = self.soft_thresholding(
                self.beta_md - self.smooth_grad / self.L_convex,
                self._lambda / self.L_convex)
            FISTA_t_old = FISTA_t_new
            FISTA_t_new = (1 + cp.sqrt(1 + 4 * FISTA_t_new**2)) / 2
            diff_temp = x_new - x_old
            self.beta_md = x_new + (FISTA_t_old - 1) / FISTA_t_new * diff_temp
            FISTA_converged = cp.all(
                cp.max(cp.abs(diff_temp)) < self.tol
            ).item(
            ) and self.FISTA_k != 1  # since when FISTA_k=1, x_new and x_old are the same
            if self.penalty == "SCAD":
                self.eval_obj_SCAD(x_new, self.obj_value)
            else:
                self.eval_obj_MCP(x_new, self.obj_value)
        self.FISTA_beta = x_new  # because we used self.beta_md as y all the time, now it should be fixed
        self.beta = self.FISTA_beta.copy()
        self.beta_ag = self.FISTA_beta.copy()
        self.beta_md = self.FISTA_beta.copy()

    def Two_step_FISTA_UAG(self):
        '''
        Carry out the two step combining FISTA and Ghadimi's AG.
        '''
        self.FISTA_logistic_SCAD_MCP()
        self.k = self.FISTA_k  # so FISTA iterations will also count into the number of iterations
        self.beta = self.FISTA_beta.copy()
        self.beta_ag = self.FISTA_beta.copy()
        self.beta_md = self.FISTA_beta.copy()
        return self.UAG_logistic_SCAD_MCP()

    def Two_step_FISTA_ISTA(self):
        '''
        Carry out the two step combining FISTA and ISTA.
        '''
        self.FISTA_logistic_SCAD_MCP()
        self.k = self.FISTA_k  # so FISTA iterations will also count into the number of iterations
        self.beta = self.FISTA_beta.copy()
        self.beta_ag = self.FISTA_beta.copy()
        self.beta_md = self.FISTA_beta.copy()
        return self.vanilla_proximal()

    def Two_step_FISTA_Ghadimi(self):
        '''
        Carry out the two step combining FISTA and the suggested parameter settings.
        '''
        self.FISTA_logistic_SCAD_MCP()
        self.k = self.FISTA_k  # so FISTA iterations will also count into the number of iterations
        self.beta = self.FISTA_beta.copy()
        self.beta_ag = self.FISTA_beta.copy()
        self.beta_md = self.FISTA_beta.copy()
        return self.UAG_logistic_SCAD_MCP_Ghadimi_parameter()

    def UAG_logistic_SCAD_MCP_Ghadimi_parameter(self):
        '''
        Carry out the optimization.
        '''
        if self.penalty == "SCAD":
            L = max([self.L_convex, 1 / (self.a - 1)])
            self.opt_beta = .5 / L
            self.eval_obj_SCAD(self.beta_md, self.obj_value)
            self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    self.opt_alpha = 2 / (
                        self.k + 1
                    )  #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    self.opt_lambda = self.k / 2 * self.opt_beta  #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    #self.opt_alpha = 2/(1+cp.sqrt(1+4/self.opt_alpha**2)) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    #self.opt_lambda = self.opt_beta/self.opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    self.beta_md = (
                        1 - self.opt_alpha
                    ) * self.beta_ag + self.opt_alpha * self.beta
                    self.update_smooth_grad_SCAD()
                    self.beta = self.soft_thresholding(
                        self.beta - self.opt_lambda * self.smooth_grad,
                        self.opt_lambda * self._lambda)
                    self.beta_ag = self.soft_thresholding(
                        self.beta_md - self.opt_beta * self.smooth_grad,
                        self.opt_beta * self._lambda)
                    self.converged = cp.all(
                        cp.max(
                            cp.abs(self.beta_md - self.beta_ag) /
                            self.opt_beta) < self.tol).item()
                    self.eval_obj_SCAD(self.beta_md, self.obj_value)
                    self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
                else:
                    break
        else:
            L = max([self.L_convex, 1 / self.gamma])
            self.opt_beta = .5 / L
            self.eval_obj_MCP(self.beta_md, self.obj_value)
            self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    self.opt_alpha = 2 / (
                        self.k + 1
                    )  #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    self.opt_lambda = self.k / 2 * self.opt_beta  #parameter setting based on Ghadimi and Lan's exemplified Lemma
                    #self.opt_alpha = 2/(1+cp.sqrt(1+4/self.opt_alpha**2)) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                    #self.opt_lambda = self.opt_beta/self.opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper
                    self.beta_md = (
                        1 - self.opt_alpha
                    ) * self.beta_ag + self.opt_alpha * self.beta
                    self.update_smooth_grad_MCP()
                    self.beta = self.soft_thresholding(
                        self.beta - self.opt_lambda * self.smooth_grad,
                        self.opt_lambda * self._lambda)
                    self.beta_ag = self.soft_thresholding(
                        self.beta_md - self.opt_beta * self.smooth_grad,
                        self.opt_beta * self._lambda)
                    self.converged = cp.all(
                        cp.max(cp.abs(self.beta_md - self.beta_ag)) /
                        self.opt_beta < self.tol).item()
                    self.eval_obj_MCP(self.beta_md, self.obj_value)
                    self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
                else:
                    break
        return self.report_results()

    def vanilla_proximal(self):
        '''
        Carry out optimization using vanilla gradient descent.
        '''
        if self.penalty == "SCAD":
            L = max([self.L_convex, 1 / (self.a - 1)])
            self.vanilla_stepsize = 1 / L
            self.eval_obj_SCAD(self.beta_md, self.obj_value)
            self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
            self.old_beta = self.beta_md - 10.
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    self.update_smooth_grad_SCAD()
                    self.beta_md = self.soft_thresholding(
                        self.beta_md -
                        self.vanilla_stepsize * self.smooth_grad,
                        self.vanilla_stepsize * self._lambda)
                    self.converged = cp.all(
                        cp.max(cp.abs(self.beta_md -
                                      self.old_beta)) < self.tol).item()
                    self.old_beta = self.beta_md.copy()
                    self.eval_obj_SCAD(self.beta_md, self.obj_value)
                    self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
                else:
                    break
        else:
            L = max([self.L_convex, 1 / self.gamma])
            self.vanilla_stepsize = 1 / L
            self.eval_obj_MCP(self.beta_md, self.obj_value)
            self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
            self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
            self.old_beta = self.beta_md - 10.
            while not self.converged:
                self.k += 1
                if self.k <= self.maxit:
                    self.update_smooth_grad_MCP()
                    self.beta_md = self.soft_thresholding(
                        self.beta_md -
                        self.vanilla_stepsize * self.smooth_grad,
                        self.vanilla_stepsize * self._lambda)
                    self.converged = (cp.max(
                        cp.abs(self.beta_md - self.old_beta)) < self.tol)
                    self.old_beta = self.beta_md.copy()
                    self.eval_obj_MCP(self.beta_md, self.obj_value)
                    self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
                else:
                    break
        return self.report_results()

    def coordinate_descent(self):
        _Xbeta = self.X @ self.beta_md
        while not self.converged:
            self.k += 1
            if self.k <= self.maxit:
                self.beta_md_old = self.beta_md.copy()
                if self.penalty == "SCAD":
                    for j in range(self.p):
                        pi = np.tanh(_Xbeta / 2) / 2 + .5
                        W = pi * (1 - pi)
                        v_j = 1 / self.N * (self.X[:, j] * W) @ self.X[:, j]
                        z_j = 1 / self.N * self.X[:, j] @ (
                            self.y - pi) + v_j * self.beta_md[j]
                        if j != 0:
                            _Xbeta -= self.X[:, j] * self.beta_md[j]
                            self.beta_md[j] = cp.where(
                                cp.abs(z_j) <= self._lambda * (v_j + 1.),
                                self.soft_thresholding_scalar(
                                    z_j, self._lambda) / v_j,
                                cp.where(
                                    cp.abs(z_j) <= self._lambda * v_j * self.a,
                                    self.soft_thresholding_scalar(
                                        z_j, self.a * self._lambda /
                                        (self.a - 1.)) /
                                    (v_j - 1. / (self.a - 1)), z_j / v_j))
                            _Xbeta += self.X[:, j] * self.beta_md[j]
                    self.eval_obj_SCAD(self.beta_md, self.obj_coord_value)
                    self.eval_obj_SCAD(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_SCAD(self.beta_ag, self.obj_value_AG)
                else:
                    for j in range(self.p):
                        pi = np.tanh(_Xbeta / 2) / 2 + .5
                        W = pi * (1 - pi)
                        v_j = 1 / self.N * (self.X[:, j] * W) @ self.X[:, j]
                        z_j = 1 / self.N * self.X[:, j] @ (
                            self.y - pi) + v_j * self.beta_md[j]
                        if j != 0:
                            _Xbeta -= self.X[:, j] * self.beta_md[j]
                            self.beta_md[j] = cp.where(
                                cp.abs(z_j) <= self._lambda * self.gamma * v_j,
                                self.soft_thresholding_scalar(
                                    z_j, self._lambda) /
                                (v_j - 1. / self.gamma), z_j / v_j)
                            _Xbeta += self.X[:, j] * self.beta_md[j]
                    self.eval_obj_MCP(self.beta_md, self.obj_coord_value)
                    self.eval_obj_MCP(self.beta, self.obj_value_ORIGINAL)
                    self.eval_obj_MCP(self.beta_ag, self.obj_value_AG)
                self.converged = (cp.max(
                    cp.abs(self.beta_md - self.beta_md_old)) < self.tol)
            else:
                break
        return self.report_results()

    def report_results(self):
        '''
        A results reporting tool.
        '''
        #self.beta_md[cp.abs(self.beta_md)<self.tol] = 0 # for those estimates below tolerance parameter, set them to 0
        self.estimates_constructor = collections.namedtuple(
            'Estimates', [
                'beta_est', 'converged', 'num_of_iterations', 'obj_values',
                'obj_values_orignal', 'obj_values_AG', 'obj_coord_values',
                'FISTA_estimates'
            ])
        results = self.estimates_constructor(self.beta_md, self.converged,
                                             self.k, self.obj_value,
                                             self.obj_value_ORIGINAL,
                                             self.obj_value_AG,
                                             self.obj_coord_value,
                                             self.FISTA_beta)

        return results


# # Strong rule implementation

# In[ ]:


def lambda_max_logistic(X, y):
    X_temp = X.copy()
    X_temp = X_temp[:, 1:]
    X_temp -= cp.mean(X_temp, 0).reshape(1, -1)
    X_temp /= cp.std(X_temp, 0)
    grad_at_0 = cp.abs((y - cp.mean(y)) @ X_temp / (2 * len(y)))  #  or -.5???
    lambda_max = cp.max(grad_at_0[1:])
    return lambda_max


def strong_rule_seq_logistic(X, y, beta_old, lambda_new, lambda_old):
    # suppose that X is already standardized to make it faster, and X has intercept column
    X_temp = X.copy()
    X_temp = X_temp[:, 1:]
    #     X_temp -= cp.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= cp.std(X_temp,0)
    grad = cp.abs(
        (y - np.tanh(X @ beta_old / 2) / 2 - .5) @ X_temp / (2 * len(y)))
    eliminated = (grad < 2 * lambda_new - lambda_old
                  )  # True means the value gets eliminated
    eliminated = cp.concatenate(
        (cp.array([False]),
         eliminated))  # because intercept coefficient is not penalized
    return eliminated


def fit_logistic(X, y, lambda_seq, penalty, a=3.7, gamma=3., tol=1e-5):
    '''
    A function to fit SCAD/MCP penalized logistic with given lambda_seq (in a decreasing order), under strong rules; with X being standardized automatically (no intercept column); lambda_max will be calculated and added at the begining of lambda sequence.
    '''
    X_temp = X.copy()
    y_temp = y.copy()
    lambda_seq_temp = lambda_seq.copy()
    X_temp[:, 1:] -= cp.mean(X[:, 1:], 0).reshape(1, -1)
    X_temp[:, 1:] /= cp.std(X[:, 1:], 0)
    beta_est = cp.zeros((len(lambda_seq_temp) + 1, X.shape[1]))
    lambda_seq_temp = cp.concatenate(
        (cp.array([lambda_max_logistic(X=X_temp, y=y_temp)]), lambda_seq_temp))
    elim = cp.array([False] + [True] * (X.shape[1] - 1))
    for i in cp.arange(len(lambda_seq_temp) - 1):
        elim_temp = strong_rule_seq_logistic(X_temp,
                                             y_temp,
                                             beta_old=beta_est[i, :],
                                             lambda_new=lambda_seq_temp[i + 1],
                                             lambda_old=lambda_seq_temp[i])
        elim = cp.logical_and(elim, elim_temp) if i > 0 else cp.array(
            [False] + [True] * (X.shape[1] - 1)
        )  # because at lambda_max all penalized coefficinets should be eliminated, then when some coefficinets start not to be eliminated, it keeps in the estimates
        temp_beta = beta_est[i, :]
        cls = logistic_SCAD_MCP(
            design_matrix=X_temp[:, cp.invert(elim)],
            outcome=y_temp,
            penalty=penalty,
            _lambda=lambda_seq_temp[i + 1],  # .6 works
            a=a,
            gamma=gamma,
            beta_0=temp_beta[cp.invert(elim)],
            tol=tol,
            maxit=500)
        beta_temp = cp.zeros(X.shape[1])
        beta_temp[cp.invert(elim)] = cls.UAG_logistic_SCAD_MCP()[0]
        beta_est[i + 1, :] = beta_temp
    beta_est[:, 1:] /= (cp.std(X[:, 1:], 0).reshape(1, -1))
    return beta_est


def fit_logistic_coord(X, y, lambda_seq, penalty, a=3.7, gamma=3., tol=1e-5):
    '''
    A function to fit SCAD/MCP penalized logistic with given lambda_seq (in a decreasing order), under strong rules; with X being standardized automatically (no intercept column); lambda_max will be calculated and added at the begining of lambda sequence.
    '''
    X_temp = X.copy()
    y_temp = y.copy()
    lambda_seq_temp = lambda_seq.copy()
    X_temp[:, 1:] -= cp.mean(X[:, 1:], 0).reshape(1, -1)
    X_temp[:, 1:] /= cp.std(X[:, 1:], 0)
    beta_est = cp.zeros((len(lambda_seq_temp) + 1, X.shape[1]))
    lambda_seq_temp = cp.concatenate(
        (cp.array([lambda_max_logistic(X=X_temp, y=y_temp)]), lambda_seq_temp))
    elim = cp.array([False] + [True] * (X.shape[1] - 1))
    for i in cp.arange(len(lambda_seq_temp) - 1):
        elim_temp = strong_rule_seq_logistic(X_temp,
                                             y_temp,
                                             beta_old=beta_est[i, :],
                                             lambda_new=lambda_seq_temp[i + 1],
                                             lambda_old=lambda_seq_temp[i])
        elim = cp.logical_and(elim, elim_temp) if i > 0 else cp.array(
            [False] + [True] * (X.shape[1] - 1)
        )  # because at lambda_max all penalized coefficinets should be eliminated, then when some coefficinets start not to be eliminated, it keeps in the estimates
        temp_beta = beta_est[i, :]
        cls = logistic_SCAD_MCP(
            design_matrix=X_temp[:, cp.invert(elim)],
            outcome=y_temp,
            penalty=penalty,
            _lambda=lambda_seq_temp[i + 1],  # .6 works
            a=a,
            gamma=gamma,
            beta_0=temp_beta[cp.invert(elim)],
            tol=tol,
            maxit=500,
            L_convex=1.
        )  # for coordinate descent, L Lipschitz constant is not needed to compute
        beta_temp = cp.zeros(X.shape[1])
        beta_temp[cp.invert(elim)] = cls.coordinate_descent()[0]
        beta_est[i + 1, :] = beta_temp
    beta_est[:, 1:] /= (cp.std(X[:, 1:], 0).reshape(1, -1))
    return beta_est


# # Some simulations
# 

# ## comparison of number of iterations required for difference of objective values from optimal values to reach $e^{-4}$, with SNR=$5$, Toeplitz being $0.1,0.5,0.9$, and N=$200,500,1000,3000$, p=$2050$ with each $10$ coefficients simulated from $N(.5,1),N(.5,1),N(-.5,1),N(-.5,1),N(1,1)$, sparsely located in the array, with $500$ zeros in-between 
# 

# In[ ]:


def simulator(seed,
              SNR,
              Toeplitz,
              N,
              penalty,
              _lambda,
              a=3.7,
              gamma=2.,
              target=-7):
    cp.random.seed(seed)
    true_beta = cp.array(
        cp.random.normal(.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(-.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(-.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(1, 1, 10).tolist())
    X_cov = toeplitz(Toeplitz**cp.arange(2050))
    mean = cp.zeros(true_beta.shape[0])
    X = cp.random.multivariate_normal(mean, X_cov, N)
    X -= cp.mean(X, 0).reshape(1, -1)
    X /= cp.std(X, 0)
    intercept_design_column = cp.ones(N).reshape(N, 1)
    X_sim = cp.concatenate((intercept_design_column, X), 1)
    true_sigma_sim = cp.sqrt(true_beta.T @ X_cov @ true_beta / SNR)
    true_beta_intercept = cp.concatenate((cp.array([0.5]), true_beta))
    signal = X_sim @ true_beta_intercept + cp.random.normal(
        0, true_sigma_sim, N)
    y_sim = cp.random.binomial(1, cp.tanh(signal / 2) / 2 + .5)

    cls = logistic_SCAD_MCP(design_matrix=X_sim,
                            outcome=y_sim,
                            penalty=penalty,
                            _lambda=_lambda,
                            a=a,
                            gamma=gamma,
                            beta_0="NOT DECLARED",
                            tol=1e-4,
                            maxit=5000)
    obj_val_AG = cp.array(cls.UAG_logistic_SCAD_MCP()[3])
    s = """cls = logistic_SCAD_MCP(design_matrix = X_sim,outcome = y_sim,penalty = penalty,_lambda = _lambda,a=a,gamma=gamma,beta_0="NOT DECLARED",tol=1e-4,maxit=5000);cls.UAG_logistic_SCAD_MCP()"""
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    #     _, AG_time = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()

    cls = logistic_SCAD_MCP(design_matrix=X_sim,
                            outcome=y_sim,
                            penalty=penalty,
                            _lambda=_lambda,
                            a=a,
                            gamma=gamma,
                            beta_0="NOT DECLARED",
                            tol=1e-4,
                            maxit=5000)
    obj_val_ISTA = cp.array(cls.vanilla_proximal()[3])
    s = """cls = logistic_SCAD_MCP(design_matrix = X_sim,outcome = y_sim,penalty = penalty,_lambda = _lambda,a=a,gamma=gamma,beta_0="NOT DECLARED",tol=1e-4,maxit=5000);cls.vanilla_proximal()"""
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    #     _, ISTA_time = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()

    cls = logistic_SCAD_MCP(design_matrix=X_sim,
                            outcome=y_sim,
                            penalty=penalty,
                            _lambda=_lambda,
                            a=a,
                            gamma=gamma,
                            beta_0="NOT DECLARED",
                            tol=1e-4,
                            maxit=5000)
    obj_val_Ghadimi = cp.array(
        cls.UAG_logistic_SCAD_MCP_Ghadimi_parameter()[3])

    s = """cls = logistic_SCAD_MCP(design_matrix = X_sim,outcome = y_sim,penalty = penalty,_lambda = _lambda,a=a,gamma=gamma,beta_0="NOT DECLARED",tol=1e-4,maxit=5000,L_convex=1.);cls.coordinate_descent()"""
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    #     _, coord_time = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()

    obj_min_val = cp.min(
        cp.array([
            cp.min(obj_val_AG),
            cp.min(obj_val_ISTA),
            cp.min(obj_val_Ghadimi)
        ]))

    obj_val_AG -= obj_min_val
    obj_val_ISTA -= obj_min_val
    obj_val_Ghadimi -= obj_min_val

    #     obj_val_AG, obj_val_ISTA, obj_val_Ghadimi = cp.log(obj_val_AG), cp.log(obj_val_ISTA), cp.log(obj_val_Ghadimi)
    #     target_min = cp.min(cp.array([cp.min(obj_val_AG),cp.min(obj_val_ISTA),cp.min(obj_val_Ghadimi)]))
    results = cp.array([cp.inf] * 3)
    if cp.any(obj_val_AG <= target) == True:
        results[0] = cp.min(cp.where(obj_val_AG <= target)[0])
    if cp.any(obj_val_ISTA <= target) == True:
        results[1] = cp.min(cp.where(obj_val_ISTA <= target)[0])
    if cp.any(obj_val_Ghadimi <= target) == True:
        results[2] = cp.min(cp.where(obj_val_Ghadimi <= target)[0])


#     results[3] = AG_time
#     results[4] = ISTA_time
#     results[5] = coord_time
# returns number of iterations for AG, ISTA, Ghadimi settings
    return results


# In[1]:


def simulator_compute_time_AG(seed,
                              SNR,
                              Toeplitz,
                              N,
                              penalty,
                              _lambda,
                              a=3.7,
                              gamma=2.,
                              target=-7):
    cp.random.seed(seed)
    true_beta = cp.array(
        cp.random.normal(.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(-.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(-.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(1, 1, 10).tolist())
    X_cov = toeplitz(Toeplitz**cp.arange(2050))
    mean = cp.zeros(true_beta.shape[0])
    X = cp.random.multivariate_normal(mean, X_cov, N)
    X -= cp.mean(X, 0).reshape(1, -1)
    X /= cp.std(X, 0)
    intercept_design_column = cp.ones(N).reshape(N, 1)
    X_sim = cp.concatenate((intercept_design_column, X), 1)
    true_sigma_sim = cp.sqrt(true_beta.T @ X_cov @ true_beta / SNR)
    true_beta_intercept = cp.concatenate((cp.array([0.5]), true_beta))
    signal = X_sim @ true_beta_intercept + cp.random.normal(
        0, true_sigma_sim, N)
    y_sim = cp.random.binomial(1, cp.tanh(signal / 2) / 2 + .5)

    s = """cls = logistic_SCAD_MCP(design_matrix = X_sim,outcome = y_sim,penalty = penalty,_lambda = _lambda,a=a,gamma=gamma,beta_0="NOT DECLARED",tol=1e-4,maxit=5000);cls.UAG_logistic_SCAD_MCP()"""
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    _, AG_time = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()

    results = cp.array([cp.inf])
    results[0] = AG_time
    return results


# In[2]:


def simulator_compute_time_coord(seed,
                                 SNR,
                                 Toeplitz,
                                 N,
                                 penalty,
                                 _lambda,
                                 a=3.7,
                                 gamma=2.,
                                 target=-7):
    cp.random.seed(seed)
    true_beta = cp.array(
        cp.random.normal(.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(-.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(-.5, 1, 10).tolist() + [0.] * 500 +
        cp.random.normal(1, 1, 10).tolist())
    X_cov = toeplitz(Toeplitz**cp.arange(2050))
    mean = cp.zeros(true_beta.shape[0])
    X = cp.random.multivariate_normal(mean, X_cov, N)
    X -= cp.mean(X, 0).reshape(1, -1)
    X /= cp.std(X, 0)
    intercept_design_column = cp.ones(N).reshape(N, 1)
    X_sim = cp.concatenate((intercept_design_column, X), 1)
    true_sigma_sim = cp.sqrt(true_beta.T @ X_cov @ true_beta / SNR)
    true_beta_intercept = cp.concatenate((cp.array([0.5]), true_beta))
    signal = X_sim @ true_beta_intercept + cp.random.normal(
        0, true_sigma_sim, N)
    y_sim = cp.random.binomial(1, cp.tanh(signal / 2) / 2 + .5)

    s = """cls = logistic_SCAD_MCP(design_matrix = X_sim,outcome = y_sim,penalty = penalty,_lambda = _lambda,a=a,gamma=gamma,beta_0="NOT DECLARED",tol=1e-4,maxit=5000,L_convex=1.);cls.coordinate_descent()"""
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    _, coord_time = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()

    results = cp.array([cp.inf])
    results[0] = coord_time
    return results


# ### MCP `MCP_sim_results.npy`, `MCP_sim_results_AG_time`, `MCP_sim_results_coord_time`

# In[ ]:


MCP_sim_results_AG_time = cp.zeros((3, 4, 100, 1))

for i, j, seed in itertools.product(range(3), range(4), range(100)):
    MCP_sim_results_AG_time[i, j, seed, :] = simulator_compute_time_AG(
        seed=seed,
        SNR=5.,
        Toeplitz=[0.1, 0.5, 0.9][i],
        N=[200, 500, 1000, 3000][j],
        penalty="MCP",
        _lambda=.02,
        a=3.7,
        gamma=2.,
        target=cp.exp(-4))

cp.save("MCP_sim_results_AG_time", MCP_sim_results_AG_time)


# In[ ]:




