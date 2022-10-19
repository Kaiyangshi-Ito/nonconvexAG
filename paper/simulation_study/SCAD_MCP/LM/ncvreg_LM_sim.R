# install.packages(c("MASS","ncvreg","RcppCNPy"), repos='http://cran.us.r-project.org')
library("MASS")
library("ncvreg")
library("RcppCNPy")

options(nwarnings = 4*3*100*2)

setwd("/home/kyang/SCAD_MCP/LM/")

x = array(0, dim = c(4, 3, 100, 2000, 2052)) # first column will be y
true_betas = array(0, dim = c(4,3,100,2051))
for (i in 1:4){
  for (j in 1:3){
    for (seed in 1:100){
      set.seed(seed)
      true_beta = c(rnorm(10, .5, 1), rep(0,500), rnorm(10, 5, 2), rep(0,500), rnorm(10, 10, 3), rep(0,500), rnorm(10, 20, 4), rep(0,500), rnorm(10, 50, 5))
      true_betas[i,j,seed,-1] = true_beta
      X_cov = toeplitz((c(0.1,0.5,0.9)[j])^c(0:(length(true_beta)-1)))
      X_temp = mvrnorm(2000, rep(0,2050), X_cov)
      X_temp = scale(X_temp)
      x[i,j,seed,,-c(1,2)] = X_temp
      true_sigma_sim = sqrt(t(true_beta)%*%X_cov%*%true_beta/(c(1,3,7,10)[i]))
      y_temp = X_temp%*%true_beta + rnorm(2000,0,true_sigma_sim) # here I simply let the true intercept coefficient to be 0 
      x[i,j,seed,,2] = 1
      x[i,j,seed,,1] = y_temp
    }
  }
}

lambda_max_LM = function(X, y){
  grad_at_0 = y%*%X/length(y)
  lambda_max = max(grad_at_0)
  return(lambda_max)}

SCAD_sim = x
SCAD_true_beta = true_betas
results_SCAD_signal_recovery = array(0, dim = c(4,3,100,5))
for (i in c(1:4)){
  for (j in c(1:3)){
    for (seed in c(1:100)){
      X_sim = SCAD_sim[i,j,seed,1:1000,-1]
      y_sim = SCAD_sim[i,j,seed,1:1000,1]
      lambda_seq = seq(lambda_max_LM(X_sim[,-1], y_sim), 0, length.out=50+1)
      temp_fit = ncvreg(X_sim[,-1], y_sim, penalty = "SCAD", lambda=lambda_seq[-51])
      temp_beta = temp_fit$beta
      testing_X = SCAD_sim[i,j,seed,1001:2000,-1]
      testing_y = SCAD_sim[i,j,seed,1001:2000,1]
      testing_temp = testing_X%*%temp_beta - testing_y # each column is error for each beta
      testing_error = apply(testing_temp, 2, function(x) norm(x,type = "2"))
      beta_ind = which.min(testing_error)
      chosen_beta = temp_beta[,beta_ind]
      temp_true_beta = SCAD_true_beta[i,j,seed,]
      norm2_error = (norm(chosen_beta-temp_true_beta, type = "2")/norm(temp_true_beta, type = "2"))^2
      norminfity_error = norm(matrix(chosen_beta-temp_true_beta), type = "M")
      npv = sum((chosen_beta==0.&temp_true_beta==0.)*1.)/sum((chosen_beta==0.)*1.)
      ppv = sum((chosen_beta!=0.&temp_true_beta!=0.)*1.)/sum((chosen_beta!=0.)*1.)
      active_set_cardi = sum((chosen_beta!=0.)*1.)
      results_SCAD_signal_recovery[i,j,seed,1] = norm2_error
      results_SCAD_signal_recovery[i,j,seed,2] = norminfity_error
      results_SCAD_signal_recovery[i,j,seed,3] = ppv
      results_SCAD_signal_recovery[i,j,seed,4] = npv
      results_SCAD_signal_recovery[i,j,seed,5] = active_set_cardi
    }
  }
}

warnings()

npySave("R_results_SCAD_signal_recovery.npy", results_SCAD_signal_recovery)


MCP_sim = x
MCP_true_beta = true_betas
results_MCP_signal_recovery = array(0, dim = c(4,3,100,5))
for (i in c(1:4)){
  for (j in c(1:3)){
    for (seed in c(1:100)){
      X_sim = MCP_sim[i,j,seed,1:1000,-1]
      y_sim = MCP_sim[i,j,seed,1:1000,1]
      lambda_seq = seq(lambda_max_LM(X_sim[,-1], y_sim), 0, length.out=50+1)
      temp_fit = ncvreg(X_sim[,-1], y_sim, penalty = "MCP", lambda=lambda_seq[-51])
      temp_beta = temp_fit$beta
      testing_X = MCP_sim[i,j,seed,1001:2000,-1]
      testing_y = MCP_sim[i,j,seed,1001:2000,1]
      testing_temp = testing_X%*%temp_beta - testing_y # each column is error for each beta
      testing_error = apply(testing_temp, 2, function(x) norm(x,type = "2"))
      beta_ind = which.min(testing_error)
      chosen_beta = temp_beta[,beta_ind]
      temp_true_beta = MCP_true_beta[i,j,seed,]
      norm2_error = (norm(chosen_beta-temp_true_beta, type = "2")/norm(temp_true_beta, type = "2"))^2
      norminfity_error = norm(matrix(chosen_beta-temp_true_beta), type = "M")
      npv = sum((chosen_beta==0.&temp_true_beta==0.)*1.)/sum((chosen_beta==0.)*1.)
      ppv = sum((chosen_beta!=0.&temp_true_beta!=0.)*1.)/sum((chosen_beta!=0.)*1.)
      active_set_cardi = sum((chosen_beta!=0.)*1.)
      results_MCP_signal_recovery[i,j,seed,1] = norm2_error
      results_MCP_signal_recovery[i,j,seed,2] = norminfity_error
      results_MCP_signal_recovery[i,j,seed,3] = ppv
      results_MCP_signal_recovery[i,j,seed,4] = npv
      results_MCP_signal_recovery[i,j,seed,5] = active_set_cardi
    }
  }
}

warnings()

npySave("R_results_MCP_signal_recovery.npy", results_MCP_signal_recovery)
