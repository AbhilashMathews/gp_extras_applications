#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:56:39 2019

@author: mathewsa

This script is used for plotting electron density and temperature on the 
1D (i.e. radial at a specific time) domain specified by the user. This script is 
to be run only after first running and saving the GP after it has been trained
upon the experimental data. This script displays the electron density and 
temperature (and their corresponding prediction and confidence intervals). Both
'check_n' and 'check_T' will indicate if fit converged on these points or if
the time slice should be neglected in the fitted data. Derivatives are also plotted.
"""

import sys
sys.path.append('C:/Users/mathewsa/') #provides path to gp_extras  
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C 
from gp_extras.kernels import HeteroscedasticKernel, LocalLengthScalesKernel 
from scipy.optimize import differential_evolution
from scipy.linalg import cholesky, cho_solve, solve_triangular 
from scipy import stats 
import gp_extras
from mpl_toolkits.mplot3d import Axes3D  
from numpy.linalg import inv 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

time = 1.2 #in seconds, for single plot          
psi_min = 0.85 #lower limit you want for plotting x-axis
psi_max = 1.05 #upper limit you want for plotting y-axis
T_min = 0.0 #in keV, lower limit you want for plotting y-axis
T_max = 1.0 #in keV, upper limit you want for plotting y-axis
dpsi = psi_spacing = 0.01 #normalized poloidal flux coordinate spacing you want
file_path = '.../trainedGPs/saved_GP_1091016033/'
n_sampling = 10000 #increase for cleaner statistics 
time_spacing = 0.001 #seconds 
n_plots = 3 #number of samples to draw on plot; should be smaller than n_sampling

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

inputs_x_array = np.arange(psi_min,psi_max,dpsi)
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
 
fig2 = plt.figure(figsize=(10,6)) 
plt.clf()  
ax1 = fig2.add_subplot(111) 
ax1.plot(np.array(inputs_y)[:,0],mean_n,'r-')
ax1.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_n - 1.96*sigma_n,
                    (mean_n + 1.96*sigma_n)[::-1]]),
fc='r',ec='None',label='95% prediction interval',alpha=0.1)
ax1.set_xlabel(r"$\psi$")
ax1.set_ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r')  

ax2 = ax1.twinx() 
ax2.plot(np.array(inputs_y)[:,0],mean_T,'g-')
ax2.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_T - 1.96*sigma_T,
                    (mean_T + 1.96*sigma_T)[::-1]]),
fc='g',ec='None',label='95% prediction interval',alpha=0.1)
ax2.set_ylabel("T"+r"$_e$"+" (keV)",color='g')
plt.gca().set_ylim(bottom=0)  
alpha_n = X_n[:,1] - inputs_x_array[:,1][0]
alpha_n = 1./(alpha_n/time_spacing) 
alpha_n = np.abs(alpha_n)
alpha_n[alpha_n > 1] = 1.0
alpha_n[alpha_n < 0.01] = 0.0
rgba_colors = np.zeros((len(X_n),4)) 
rgba_colors[:,0] = 1.0 
rgba_colors[:, 3] = alpha_n 
plot_X_n = X_n[:,0]
plot_y_n_TS = y_n_TS
plot_y_n_TS_err = y_n_TS_err 

ax1.scatter(plot_X_n,plot_y_n_TS, c=rgba_colors, edgecolors=rgba_colors)
for pos, ypt, err, color in zip(plot_X_n, plot_y_n_TS, plot_y_n_TS_err, rgba_colors):
    plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
    
alpha_T = X_T[:,1] - inputs_x_array[:,1][0]
alpha_T = 1./(alpha_T/time_spacing) 
alpha_T = np.abs(alpha_T)
alpha_T[alpha_T > 1] = 1.0
alpha_T[alpha_T < 0.01] = 0.0
rgba_colors = np.zeros((len(X_T),4)) 
rgba_colors[:,1] = 1.0 
rgba_colors[:, 3] = alpha_T 
plot_X_T = X_T[:,0]
plot_y_T_TS = y_T_TS
plot_y_T_TS_err = y_T_TS_err 

ax2.scatter(plot_X_T,plot_y_T_TS, c=rgba_colors, edgecolors=rgba_colors)
for pos, ypt, err, color in zip(plot_X_T, plot_y_T_TS, plot_y_T_TS_err, rgba_colors):
    plotline, caplines, (barlinecols,) = ax2.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)

plt.title("Time is "+str(np.round(time,3))+"s")
plt.gca().set_ylim(bottom=0) 
ax2.set_ylim(T_min,T_max)
plt.xlim(psi_min,psi_max) 
plt.legend()
plt.show() 
  
#non-Heteroscedastic sampling   
X_train = X_n
K_trans1 = gp.kernel_(X_test, X_train)  
K = gp.kernel_(X_train) 

try:
    L_ = cholesky(K, lower=True)
    v1 = cho_solve((L_, True), K_trans1.T) 
    y_cov1 = gp.kernel_(X_test,X_test) - K_trans1.dot(v1) # this is best code and fix to from gp_samples  
    output_cov1 = np.random.multivariate_normal(mean_y_arr,y_cov1,n_sampling).T #(NH - non-heteroscedastic)
    n_samples_NH = output_cov1 
except np.linalg.LinAlgError:  
    print('Error in cholesky')
#    alternative formulation
#    L_ = cholesky(nearestPD(K), lower=True) #to ensure LinAlgError: 1303-th leading minor of the array is not positive definite is solved; https://github.com/rlabbe/filterpy/issues/62; https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite - mplementation of Higham’s 1988 paper
#    y_cov2 = gp.kernel_(X_test,X_test) - np.dot(np.dot(gp.kernel_(X_test,X_train), inv_K),(gp.kernel_(X_test,X_train)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent) 
#    output_cov2 = np.random.multivariate_normal(mean_y_arr,y_cov2,samples_cholesky).T #(NH - non-heteroscedastic)
#    n_samples_NH = output_cov2
except: 
    print('Unknown error in cholesky')
 
K_trans1_T = gp_T.kernel_(X_test_T, X_train_T) 
K_T = gp_T.kernel_(X_train_T)
#inv_K_T = inv(gp_T.kernel_(X_train_T,X_train_T) + np.eye(len(X_train_T))*(y_T_TS_err)**2.) 

try:
    L__T = cholesky(K_T, lower=True)
    v1_T = cho_solve((L__T, True), K_trans1_T.T) 
    y_cov1_T = gp_T.kernel_(X_test_T,X_test_T) - K_trans1_T.dot(v1_T) # this is best code and fix to from gp_samples  
    output_cov1_T = np.random.multivariate_normal(mean_y_arr_T,y_cov1_T,n_sampling).T #(NH - non-heteroscedastic)
    T_samples_NH = output_cov1_T
    
except np.linalg.LinAlgError:  
    print('Error in cholesky')
#    L__T = cholesky(nearestPD(K_T), lower=True) #to ensure LinAlgError: 1303-th leading minor of the array is not positive definite is solved; https://github.com/rlabbe/filterpy/issues/62; https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite - mplementation of Higham’s 1988 paper
#    y_cov2_T = gp_T.kernel_(X_test_T,X_test_T) - np.dot(np.dot(gp_T.kernel_(X_test_T,X_train_T), inv_K_T),(gp_T.kernel_(X_test_T,X_train_T)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent)
#    K_trans_T = gp_T.kernel_(X_test_T,X_train_T) 
#    output_cov2_T = np.random.multivariate_normal(mean_y_arr_T,y_cov2_T,samples_cholesky).T #(NH - non-heteroscedastic)
#    T_samples_NH = output_cov2_T
except: 
    print('Unknown error in cholesky')
 
i_index = 0
mean_n_NH = []
sigma_n_NH = []
mean_T_NH = []
sigma_T_NH = []
while i_index < len(inputs_x_array):  
    mean_n_NH.append(np.mean(n_samples_NH[i_index]))
    sigma_n_NH.append(np.std(n_samples_NH[i_index]))
    mean_T_NH.append(np.mean(T_samples_NH[i_index]))
    sigma_T_NH.append(np.std(T_samples_NH[i_index]))
    i_index = i_index + 1

mean_n_NH = np.array(mean_n_NH)
sigma_n_NH = np.array(sigma_n_NH)
mean_T_NH = np.array(mean_T_NH)
sigma_T_NH = np.array(sigma_T_NH) 
 
fig2 = plt.figure(figsize=(10,6)) 
plt.clf()  
ax1 = fig2.add_subplot(111) 
ax1.plot(np.array(inputs_y)[:,0],mean_n_NH,'r-')
ax1.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_n_NH - 1.96*sigma_n_NH,
                    (mean_n_NH + 1.96*sigma_n_NH)[::-1]]),
fc='r',ec='None',label='95% confidence interval',alpha=0.1)
ax1.set_xlabel(r"$\psi$")
ax1.set_ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 

ax2 = ax1.twinx() 
ax2.plot(np.array(inputs_y)[:,0],mean_T_NH,'g-')
ax2.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_T_NH - 1.96*sigma_T_NH,
                    (mean_T_NH + 1.96*sigma_T_NH)[::-1]]),
fc='g',ec='None',label='95% confidence interval',alpha=0.1)
ax2.set_ylabel("T"+r"$_e$"+" (keV)",color='g')
plt.gca().set_ylim(bottom=0)
 
alpha_n = X_n[:,1] - inputs_x_array[:,1][0]
alpha_n = 1./(alpha_n/time_spacing) 
alpha_n = np.abs(alpha_n)
alpha_n[alpha_n > 1] = 1.0
alpha_n[alpha_n < 0.01] = 0.0
rgba_colors = np.zeros((len(X_n),4)) 
rgba_colors[:,0] = 1.0 
rgba_colors[:, 3] = alpha_n 
plot_X_n = X_n[:,0]
plot_y_n_TS = y_n_TS
plot_y_n_TS_err = y_n_TS_err 

ax1.scatter(plot_X_n,plot_y_n_TS, c=rgba_colors, edgecolors=rgba_colors)
for pos, ypt, err, color in zip(plot_X_n, plot_y_n_TS, plot_y_n_TS_err, rgba_colors):
    plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
    
alpha_T = X_T[:,1] - inputs_x_array[:,1][0]
alpha_T = 1./(alpha_T/time_spacing) 
alpha_T = np.abs(alpha_T)
alpha_T[alpha_T > 1] = 1.0
alpha_T[alpha_T < 0.01] = 0.0
rgba_colors = np.zeros((len(X_T),4)) 
rgba_colors[:,1] = 1.0 
rgba_colors[:, 3] = alpha_T 
plot_X_T = X_T[:,0]
plot_y_T_TS = y_T_TS
plot_y_T_TS_err = y_T_TS_err

ax2.scatter(plot_X_T,plot_y_T_TS, c=rgba_colors, edgecolors=rgba_colors)
for pos, ypt, err, color in zip(plot_X_T, plot_y_T_TS, plot_y_T_TS_err, rgba_colors):
    plotline, caplines, (barlinecols,) = ax2.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)

plt.title("Time is "+str(np.round(time,3))+"s")
plt.gca().set_ylim(bottom=0) 
ax2.set_ylim(T_min,T_max)
plt.xlim(psi_min,psi_max) 
plt.legend()
plt.show() 
  
#Computing gradients 
inputs_dydx = []
time_dydx = time
psi = psi_min
while psi < psi_max + psi_spacing:
    inputs_dydx.append([psi,time_dydx])
    psi = psi + psi_spacing 
    
n_samples = gp.sample_y(inputs_dydx,n_sampling)
T_samples = gp_T.sample_y(inputs_dydx,n_sampling)

i_sample = 0 
dndx_samples = []
dTdx_samples = []
while i_sample < n_sampling:
    dndx_samples.append(np.gradient(n_samples[:,0][:,i_sample],np.array(inputs_dydx)[:,0]))
    dTdx_samples.append(np.gradient(T_samples[:,0][:,i_sample],np.array(inputs_dydx)[:,0]))
    i_sample = i_sample + 1
    
i_index = 0
mean_dndx = []
sigma_dndx = []
mean_dTdx = []
sigma_dTdx = []
while i_index < len(inputs_dydx):
    samples_dndx = []
    samples_dTdx = []
    for sample_j in dndx_samples:
        samples_dndx.append(sample_j[i_index])
    for sample_j in dTdx_samples:
        samples_dTdx.append(sample_j[i_index])
    mean_dndx.append(np.mean(samples_dndx))
    sigma_dndx.append(np.std(samples_dndx))
    mean_dTdx.append(np.mean(samples_dTdx))
    sigma_dTdx.append(np.std(samples_dTdx))
    i_index = i_index + 1

mean_dndx = np.array(mean_dndx)
sigma_dndx = np.array(sigma_dndx)
mean_dTdx = np.array(mean_dTdx)
sigma_dTdx = np.array(sigma_dTdx)

fig2 = plt.figure(figsize=(10,6))   
plt.clf() 
ax1 = fig2.add_subplot(111)
i_plot = 0
while i_plot < n_plots: 
    ax1.plot(np.array(inputs_dydx)[:,0],np.gradient(n_samples[:,0][:,i_plot],np.array(inputs_dydx)[:,0]),alpha=.5) 
    i_plot = i_plot + 1
ax1.plot(np.array(inputs_dydx)[:,0],mean_dndx,'r-')
ax1.fill(np.concatenate([np.array(inputs_dydx)[:,0],np.array(inputs_dydx)[:,0][::-1]]),
     np.concatenate([mean_dndx - 1.96*sigma_dndx,
                    (mean_dndx + 1.96*sigma_dndx)[::-1]]),
fc='r',ec='None',label='95% prediction interval',alpha=0.1)
ax1.set_xlabel(r"$\psi$")
ax1.set_ylabel(r"$\partial$"+"n"+r"$_e / \partial \psi \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 

ax2 = ax1.twinx() 
ax2.plot(np.array(inputs_dydx)[:,0],mean_dTdx,'g-')
ax2.fill(np.concatenate([np.array(inputs_dydx)[:,0],np.array(inputs_dydx)[:,0][::-1]]),
     np.concatenate([mean_dTdx - 1.96*sigma_dTdx,
                    (mean_dTdx + 1.96*sigma_dTdx)[::-1]]),
fc='g',ec='None',label='95% prediction interval',alpha=0.1)
ax2.set_ylabel(r"$\partial$"+"T"+r"$_e / \partial \psi$"+" (keV)",color='g') 
plt.xlim(psi_min,psi_max)                
plt.legend() 
plt.show()   

i_sample = 0 
dndx_samples_NH = []
dTdx_samples_NH = []
while i_sample < n_sampling:
    dndx_samples_NH.append(np.gradient(n_samples_NH[:,i_sample],np.array(inputs_dydx)[:,0]))
    dTdx_samples_NH.append(np.gradient(T_samples_NH[:,i_sample],np.array(inputs_dydx)[:,0]))
    i_sample = i_sample + 1
    
i_index = 0
mean_dndx_NH = []
sigma_dndx_NH = []
mean_dTdx_NH = []
sigma_dTdx_NH = []
while i_index < len(inputs_dydx):
    samples_dndx_NH = []
    samples_dTdx_NH = []
    for sample_j in dndx_samples_NH:
        samples_dndx_NH.append(sample_j[i_index])
    for sample_j in dTdx_samples_NH:
        samples_dTdx_NH.append(sample_j[i_index])
    mean_dndx_NH.append(np.mean(samples_dndx_NH))
    sigma_dndx_NH.append(np.std(samples_dndx_NH))
    mean_dTdx_NH.append(np.mean(samples_dTdx_NH))
    sigma_dTdx_NH.append(np.std(samples_dTdx_NH))
    i_index = i_index + 1

mean_dndx_NH = np.array(mean_dndx_NH)
sigma_dndx_NH = np.array(sigma_dndx_NH)
mean_dTdx_NH = np.array(mean_dTdx_NH)
sigma_dTdx_NH = np.array(sigma_dTdx_NH) 

fig2 = plt.figure(figsize=(10,6))   
plt.clf() 
ax1 = fig2.add_subplot(111)
i_plot = 0
while i_plot < n_plots: 
    ax1.plot(np.array(inputs_dydx)[:,0],np.gradient(n_samples_NH[:,i_plot],np.array(inputs_dydx)[:,0]),alpha=.5) 
    i_plot = i_plot + 1
ax1.plot(np.array(inputs_dydx)[:,0],mean_dndx_NH,'r-')
ax1.fill(np.concatenate([np.array(inputs_dydx)[:,0],np.array(inputs_dydx)[:,0][::-1]]),
     np.concatenate([mean_dndx_NH - 1.96*sigma_dndx_NH,
                    (mean_dndx_NH + 1.96*sigma_dndx_NH)[::-1]]),
fc='r',ec='None',label='95% confidence interval',alpha=0.1)
ax1.set_xlabel(r"$\psi$")
ax1.set_ylabel(r"$\partial$"+"n"+r"$_e / \partial \psi \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 

ax2 = ax1.twinx()
i_plot = 0
while i_plot < n_plots: 
    ax2.plot(np.array(inputs_dydx)[:,0],np.gradient(T_samples_NH[:,i_plot],np.array(inputs_dydx)[:,0]),alpha=.5) 
    i_plot = i_plot + 1
ax2.plot(np.array(inputs_dydx)[:,0],mean_dTdx_NH,'g-')
ax2.fill(np.concatenate([np.array(inputs_dydx)[:,0],np.array(inputs_dydx)[:,0][::-1]]),
     np.concatenate([mean_dTdx_NH - 1.96*sigma_dTdx_NH,
                    (mean_dTdx_NH + 1.96*sigma_dTdx_NH)[::-1]]),
fc='g',ec='None',label='95% confidence interval',alpha=0.1)
ax2.set_ylabel(r"$\partial$"+"T"+r"$_e / \partial \psi$"+" (keV)",color='g')
plt.xlim(psi_min,psi_max)  
plt.title("Time is "+str(time_dydx)+"s")              
plt.legend()      
plt.show()  