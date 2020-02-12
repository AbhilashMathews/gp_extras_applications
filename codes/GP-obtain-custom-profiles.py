#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:56:39 2019

@author: mathewsa

This is an experimental script for testing different sampling distributions.
It permits applying a truncated Gaussian and/or arbitrary sampling choices as 
defined by the user (e.g. a linear function is applied below) in addition to 
standard normal distributions. This script is to be run only after first running 
and saving the GP after it has been trained upon the experimental data. This 
script displays the electron density and temperature (and their corresponding 
prediction intervals) with single sample realizations plotted in blue. 
"""

import sys
sys.path.append('C:/Users/mathewsa/') #provides path to gp_extras #provides path to gp_extras
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C 
from gp_extras.kernels import HeteroscedasticKernel, LocalLengthScalesKernel 
from scipy.optimize import differential_evolution
from scipy.linalg import cholesky, cho_solve
from scipy import stats 
import gp_extras
from numpy.linalg import inv
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18

time = 1.2 #in seconds, for single plot          
psi_min = 0.85 #lower limit you want for plotting x-axis
psi_max = 1.05 #upper limit you want for plotting y-axis
dpsi = 0.01 #normalized poloidal flux coordinate spacing you want
lower, upper = 1.0, 0.0 #lower and upper bounds for the truncated Gaussian
mu, sigma = 0.0, 1.0 #mean and standard deviation for the truncated Gaussian
n_sampling = 1000 #increase for cleaner statistics
file_path = '.../trainedGPs/saved_GP_1091016033/'

# --------------------------------------------------------------
#                       End of user inputs
# -------------------------------------------------------------- 

X_n = np.load(str(file_path)+'X_n.npy')
y_n_TS = np.load(str(file_path)+'y_n_TS.npy')
y_n_TS_err = np.load(str(file_path)+'y_n_TS_err.npy')
X_T = np.load(str(file_path)+'X_T.npy')
y_T_TS = np.load(str(file_path)+'y_T_TS.npy')
y_T_TS_err = np.load(str(file_path)+'y_T_TS_err.npy') 
n_max_iter = np.load(str(file_path)+'n_max_iter.npy')  

def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=n_max_iter, disp=False, polish=True)
    return res.x, obj_func(res.x, eval_gradient=False) 

gp = pickle.load(open(str(file_path)+"gp.dump","rb"))
gp_T = pickle.load(open(str(file_path)+"gp_T.dump","rb"))

inputs_x_array = np.arange(psi_min,psi_max + dpsi,dpsi)
a = np.ones((len(inputs_x_array),2))
a[:,0] = inputs_x_array
a[:,1] = a[:,1]*time
inputs_x_array = inputs_x_array_n = inputs_x_array_T = a 
  
lls_len_scale = gp.kernel_.k1.k2.theta_gp* 10**gp.kernel_.k1.k2.gp_l.predict(inputs_x_array)
m_lls_n = stats.mode(lls_len_scale)  
lls_len_scale_T = gp_T.kernel_.k1.k2.theta_gp* 10**gp_T.kernel_.k1.k2.gp_l.predict(inputs_x_array)
m_lls_T = stats.mode(lls_len_scale_T)
 
err_T = np.abs(lls_len_scale_T - m_lls_T[0][0]) #proxy for error 
err_n = np.abs(lls_len_scale - m_lls_n[0][0]) #proxy for error 

check_n = len(np.where(err_n != 0)[0])
check_T = len(np.where(err_T != 0)[0]) 
 
X_train = X_n
X_test = inputs_x_array
mean_y_arr = gp.predict(X_test, return_cov=False) 
mean_y_arr = mean_y_arr[:,0]

X_train_T = X_T
X_test_T = inputs_x_array
mean_y_arr_T = gp_T.predict(X_test_T, return_cov=False) 
mean_y_arr_T = mean_y_arr_T[:,0] 

n_samples = gp.sample_y(inputs_x_array,n_sampling)
T_samples = gp_T.sample_y(inputs_x_array,n_sampling) 

inputs_y = inputs_x_array
i_index = 0
mean_n = []
sigma_n = []
mean_T = []
sigma_T = []
while i_index < len(inputs_y):  
    mean_n.append(np.mean(n_samples[i_index]))
    sigma_n.append(np.std(n_samples[i_index]))
    mean_T.append(np.mean(T_samples[i_index]))
    sigma_T.append(np.std(T_samples[i_index]))
    i_index = i_index + 1

mean_n = np.array(mean_n)
sigma_n = np.array(sigma_n)
mean_T = np.array(mean_T)
sigma_T = np.array(sigma_T) 

mean_n_true = mean_n
sigma_n_true = sigma_n
mean_T_true = mean_T
sigma_T_true = sigma_T
            
#non-Heteroscedastic sampling   
X_train = X_n
K_trans1 = gp.kernel_(X_test, X_train)  
K = gp.kernel_(X_train) 
#inv_K = inv(gp.kernel_(X_train,X_train) + np.eye(len(X_train))*(y_n_TS_err)**2.)
 
K_trans1_T = gp_T.kernel_(X_test_T, X_train_T) 
K_T = gp_T.kernel_(X_train_T)
#inv_K_T = inv(gp_T.kernel_(X_train_T,X_train_T) + np.eye(len(X_train_T))*(y_T_TS_err)**2.) 
 
K_trans1_T = gp_T.kernel_(X_test_T, X_train_T) 
K_T = gp_T.kernel_(X_train_T)
#inv_K_T = inv(gp_T.kernel_(X_train_T,X_train_T) + np.eye(len(X_train_T))*(y_T_TS_err)**2.) 

from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrices with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        A = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    
    
K = gp.kernel_(X_train,X_train)
K = nearestPD(K)
L_ = cholesky(K, lower=True)
#L_1 = np.linalg.cholesky(K)
v1 = cho_solve((L_, True), K_trans1.T)  # Line 5
inv_K = inv(gp.kernel_(X_train,X_train) + np.eye(len(X_train))*(y_n_TS_err)**2.)
y_cov0 = gp.kernel_(X_test) - K_trans1.dot(v1) # this is code from gp_samples 
y_cov1 = gp.kernel_(X_test,X_test) - K_trans1.dot(v1) # this is best code and fix to from gp_samples 
y_cov2 = gp.kernel_(X_test,X_test) - np.dot(np.dot(gp.kernel_(X_test,X_train), inv_K),(gp.kernel_(X_test,X_train)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent)


j_K = nearestPD(y_cov2)
y_cov_L = cholesky(j_K, lower=True)
f_post = mean_n_true + np.dot(y_cov_L, np.random.normal(size=(len(X_test),1)))[:,0]
plt.figure()
plt.plot(X_test[:,0],f_post)
plt.plot(np.array(inputs_y)[:,0],mean_n_true,'r-') 
plt.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_n_true - 1.96*sigma_n_true,
                    (mean_n_true + 1.96*sigma_n_true)[::-1]]),
fc='r',ec='None',label='95% prediction interval',alpha=0.1)
plt.title('Normal GP sampling')
plt.xlabel(r"$\psi$")
plt.ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 
plt.show()
f_post = mean_n_true + np.dot(y_cov_L, -np.linspace(0.,1.,len(X_test)))
plt.figure()
plt.plot(X_test[:,0],f_post)
plt.plot(np.array(inputs_y)[:,0],mean_n_true,'r-') 
plt.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_n_true - 1.96*sigma_n_true,
                    (mean_n_true + 1.96*sigma_n_true)[::-1]]),
fc='r',ec='None',label='95% prediction interval',alpha=0.1)
plt.title('Arbitrary sampling using a decreasing linear function')
plt.xlabel(r"$\psi$")
plt.ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 
plt.show()
 

K_T = gp_T.kernel_(X_train_T,X_train_T)
K_T = nearestPD(K_T)
L__T = cholesky(K_T, lower=True) 
v1_T = cho_solve((L__T, True), K_trans1_T.T)  # Line 5
inv_K_T = inv(gp_T.kernel_(X_train_T,X_train_T) + np.eye(len(X_train_T))*(y_T_TS_err)**2.)
y_cov0_T = gp_T.kernel_(X_test) - K_trans1_T.dot(v1_T) # this is code from gp_samples 
y_cov1_T = gp_T.kernel_(X_test,X_test) - K_trans1_T.dot(v1_T) # this is best code and fix to from gp_samples 
y_cov2_T = gp_T.kernel_(X_test,X_test) - np.dot(np.dot(gp_T.kernel_(X_test,X_train_T), inv_K_T),(gp_T.kernel_(X_test,X_train_T)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent)
 
j_K_T = nearestPD(y_cov2_T)
y_cov_L_T = cholesky(j_K_T, lower=True)
f_post_T = mean_T_true + np.dot(y_cov_L_T, np.random.normal(size=(len(X_test),1)))[:,0]
plt.figure()
plt.plot(X_test[:,0],f_post_T)
plt.plot(np.array(inputs_y)[:,0],mean_T_true,'g-') 
plt.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_T_true - 1.96*sigma_T_true,
                    (mean_T_true + 1.96*sigma_T_true)[::-1]]),
fc='g',ec='None',label='95% prediction interval',alpha=0.1)
plt.xlabel(r"$\psi$")
plt.ylabel("T"+r"$_e$"+" (keV)",color='g')
plt.title('Normal GP sampling')
plt.show()

f_post_T = mean_T_true + np.dot(y_cov_L_T, -np.linspace(0.,0.5,len(X_test)))
plt.figure()
plt.plot(X_test[:,0],f_post_T)
plt.plot(np.array(inputs_y)[:,0],mean_T_true,'g-') 
plt.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_T_true - 1.96*sigma_T_true,
                    (mean_T_true + 1.96*sigma_T_true)[::-1]]),
fc='g',ec='None',label='95% prediction interval',alpha=0.1)
plt.xlabel(r"$\psi$")
plt.ylabel("T"+r"$_e$"+" (keV)",color='g')
plt.title('Arbitrary sampling using a decreasing linear function')
plt.show()

X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)
 
fig, ax = plt.subplots(2, sharex=True)
ax[0].set_title('Standard Gaussian applied for sampling')
ax[0].hist(X.rvs(10000), 50, normed=True)
ax[1].set_title('Truncated Gaussian applied for sampling')
ax[1].hist(N.rvs(10000), 50, normed=True)
plt.show()

f_post_T = mean_T_true + np.dot(y_cov_L_T, X.rvs(len(X_test)))
plt.figure()
plt.plot(X_test[:,0],f_post_T)
plt.plot(np.array(inputs_y)[:,0],mean_T_true,'g-') 
plt.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_T_true - 1.96*sigma_T_true,
                    (mean_T_true + 1.96*sigma_T_true)[::-1]]),
fc='g',ec='None',label='95% prediction interval',alpha=0.1)
plt.xlabel(r"$\psi$")
plt.ylabel("T"+r"$_e$"+" (keV)",color='g')
plt.title('GP sampling using a truncated Gaussian')
plt.show()  