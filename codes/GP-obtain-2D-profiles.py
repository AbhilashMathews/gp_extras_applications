#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:56:39 2019

@author: mathewsa

This script is used for plotting electron density and temperature on the 
2D (i.e. radial and temporal) domain specified by the user. This script is to
be run only after first running and saving the GP after it has been trained
upon the experimental data. This script will also save the electron density
and temperature (and their corresponding uncertainties on the prediction 
interval) and the 2D domain itself for further quantitative analysis.
This code can be simply extended to higher dimensional cases, too.

(Note: all times with uncoverged points are removed/can be flagged and the 
'good times' are stored in the array 'inputs_t_array_good' for plotting.)
"""

import sys
sys.path.append('C:/Users/mathewsa/') #provides path to gp_extras
import gp_extras
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C 
from gp_extras.kernels import HeteroscedasticKernel, LocalLengthScalesKernel 
from scipy.optimize import differential_evolution
from scipy import stats 
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["font.family"] = "Times New Roman"
       
psi_min = 0.85 #lower limit you want for plotting x-axis
psi_max = 1.05 #upper limit you want for plotting y-axis
T_min = 0.0 #in keV, lower limit you want for plotting y-axis
T_max = 2.0 #in keV, upper limit you want for plotting y-axis
dpsi = psi_spacing = 0.01 #normalized poloidal flux coordinate spacing you specify
time_spacing = 0.01 #seconds; this is the grid spacing you specify
t_min = 0.4 #in seconds, lower limit for x-axis for 2d array/plot
t_max = 1.6 #in seconds, upper limit for x-axis for 2d array/plot  
n_sampling = 1000 #provides higher sampling count for profile statistics
file_path = '.../trainedGPs/saved_GP_1091016033/' #path to saved GP contents
#file_path is where the gp and its variables have been saved

# --------------------------------------------------------------
#                       End of user inputs
# --------------------------------------------------------------

def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=n_max_iter, disp=False, polish=True)
    return res.x, obj_func(res.x, eval_gradient=False) 

number_of_samples = 1
X_n = np.load(str(file_path)+'X_n.npy')
y_n_TS = np.load(str(file_path)+'y_n_TS.npy')
y_n_TS_err = np.load(str(file_path)+'y_n_TS_err.npy')
X_T = np.load(str(file_path)+'X_T.npy')
y_T_TS = np.load(str(file_path)+'y_T_TS.npy')
y_T_TS_err = np.load(str(file_path)+'y_T_TS_err.npy') 
n_max_iter = np.load(str(file_path)+'n_max_iter.npy') 

gp = pickle.load(open(str(file_path)+"gp.dump","rb"))
gp_T = pickle.load(open(str(file_path)+"gp_T.dump","rb"))

#checking for converged data points
inputs_psi_array = np.arange(psi_min,psi_max,dpsi)
inputs_t_array = np.arange(t_min,t_max + time_spacing,time_spacing)
a = np.ones((len(inputs_psi_array),2))
a[:,0] = inputs_psi_array

i = 0
good_times = []
while i < len(inputs_t_array):
    bad = 0
    a[:,1] = np.ones(len(a[:,0]))*inputs_t_array[i] 
    inputs_x_array_n = inputs_x_array_T = inputs_x_array = a
    lls_len_scale = gp.kernel_.k1.k2.theta_gp* 10**gp.kernel_.k1.k2.gp_l.predict(inputs_x_array)
    m_lls_n = stats.mode(lls_len_scale)  
    lls_len_scale_T = gp_T.kernel_.k1.k2.theta_gp* 10**gp_T.kernel_.k1.k2.gp_l.predict(inputs_x_array)
    m_lls_T = stats.mode(lls_len_scale_T)
    err_T = np.abs(lls_len_scale_T - m_lls_T[0][0]) #proxy for error 
    err_n = np.abs(lls_len_scale - m_lls_n[0][0]) #proxy for error 
    
    check_n = len(np.where(err_n != 0)[0])
    check_T = len(np.where(err_T != 0)[0]) 
    
    if check_n > 0:
        if check_n < (len(inputs_x_array) - 2):
            bad = 1
             
    if check_T > 0:
        if check_T < (len(inputs_x_array) - 2):
            bad = 1
             
    if bad == 0:
        good_times.append(inputs_t_array[i])
    
    i = i + 1

inputs_t_array_old = inputs_t_array #potentially has unconverged times
inputs_t_array_good = np.array(good_times)  

x1 = inputs_psi_array #radial coordinate
x2 = inputs_t_array_good #converged temporal coordinate
len_x1 = len(x1)
len_x2 = len(x2)   
            
i = 0 
inputs_x = []
while i < len(x1):
    j = 0
    while j < len(x2):
        inputs_x.append([x1[i],x2[j]]) 
        j = j + 1
    i = i + 1
inputs_x_array = np.array(inputs_x) 

y_pred_full = []
y_pred_sigma_full = []
i = 0 
inputs_x = []
while i < len(x1):
    j = 0
    while j < len(x2):
        inputs_x.append([x1[i],x2[j]])
        y_pred_index, y_pred_sigma_index = gp.predict(np.array(inputs_x[-1]).reshape(1, -1), return_std=True)
        y_pred_index = y_pred_index[:,0]
        y_pred_full.append(y_pred_index)
        y_pred_sigma_full.append(y_pred_sigma_index)
        j = j + 1
    i = i + 1 
    
inputs_x_array = np.array(inputs_x) 
arr_y = np.array(y_pred_full)
arr_y_sigma = np.array(y_pred_sigma_full)
if arr_y.ndim == 2:
    arr_y = arr_y[:,0]
if arr_y_sigma.ndim == 2:
    arr_y_sigma = arr_y_sigma[:,0]
y_pred_full_array = arr_y
y_pred_sigma_full_array = arr_y_sigma 
y_pred_full_array_zeros = y_pred_full_array.clip(min=0)
#
y_pred_T_full = []
y_pred_sigma_T_full = []
i = 0 
inputs_x = []
while i < len(x1):
    j = 0
    while j < len(x2):
        inputs_x.append([x1[i],x2[j]])
        y_pred_T_index, y_pred_sigma_T_index = gp_T.predict(np.array(inputs_x[-1]).reshape(1, -1), return_std=True)
        y_pred_T_index = y_pred_T_index[:,0]
        y_pred_T_full.append(y_pred_T_index)
        y_pred_sigma_T_full.append(y_pred_sigma_T_index)
        j = j + 1
    i = i + 1
 
arr_y_T = np.array(y_pred_T_full)
arr_y_T_sigma = np.array(y_pred_sigma_T_full)
if arr_y_T.ndim == 2:
    arr_y_T = arr_y_T[:,0]
if arr_y_T_sigma.ndim == 2:
    arr_y_T_sigma = arr_y_T_sigma[:,0]
y_pred_T_full_array = arr_y_T
y_pred_sigma_T_full_array = arr_y_T_sigma    

y_pred_T_full_array_zeros = y_pred_T_full_array.clip(min=0)

fig = plt.figure(figsize=(10,6)) 
cm = plt.cm.get_cmap('RdYlGn')
ax = fig.add_subplot(111, projection='3d')
c = ax.scatter(inputs_x_array[:,0],inputs_x_array[:,1],y_pred_full_array_zeros,c=y_pred_full_array_zeros,cmap=cm,alpha=0.9)
ax.set_xlabel(r"$\psi$",labelpad=20)
ax.set_ylabel('Time (s)',labelpad=20)
ax.set_zlabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$")
ax.set_xlim(psi_min-0.1,psi_max+0.1) 
fig.colorbar(c, ax=ax) 
plt.show() 

fig = plt.figure(figsize=(10,6)) 
cm = plt.cm.get_cmap('RdYlGn')
ax = fig.add_subplot(111, projection='3d')
c = ax.scatter(inputs_x_array[:,0],inputs_x_array[:,1],y_pred_T_full_array_zeros,c=y_pred_T_full_array_zeros,cmap=cm,alpha=0.9)
ax.set_xlabel(r"$\psi$",labelpad=20)
ax.set_ylabel('Time (s)',labelpad=20)
ax.set_zlabel("T"+r"$_e$"+" (keV)",labelpad=10)
ax.set_xlim(psi_min-0.1,psi_max+0.1) 
fig.colorbar(c, ax=ax) 
plt.show()  
 
#Computing gradients 
dt = time_spacing
n_plots = 3 
inputs_dydx_2d_array = []
time_dydx = t_min
dndx_mean_2d = []
dndx_sigma_2d = []
dTdx_mean_2d = []
dTdx_sigma_2d = []

while time_dydx < t_max + dt:
    psi = psi_min
    inputs_dydx = []
    while psi < psi_max + psi_spacing:
        inputs_dydx.append([psi,time_dydx])
        inputs_dydx_2d_array.append([psi,time_dydx])
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
    
    dndx_mean_2d.append(mean_dndx)
    dndx_sigma_2d.append(sigma_dndx)
    dTdx_mean_2d.append(mean_dTdx)
    dTdx_sigma_2d.append(sigma_dTdx)

    time_dydx = time_dydx + dt

dndx_mean_2d_array = dndx_mean_2d[0]
dndx_sigma_2d_array = dndx_sigma_2d[0]
dTdx_mean_2d_array = dTdx_mean_2d[0]
dTdx_sigma_2d_array = dTdx_sigma_2d[0]
i = 1
while i < len(dndx_mean_2d):
    dndx_mean_2d_array = np.concatenate((dndx_mean_2d_array,dndx_mean_2d[i]))
    dndx_sigma_2d_array = np.concatenate((dndx_sigma_2d_array,dndx_sigma_2d[i]))
    dTdx_mean_2d_array = np.concatenate((dTdx_mean_2d_array,dTdx_mean_2d[i]))
    dTdx_sigma_2d_array = np.concatenate((dTdx_sigma_2d_array,dTdx_sigma_2d[i]))
    i = i + 1
    
inputs_dydx_2d_array = np.array(inputs_dydx_2d_array) 

fig = plt.figure(figsize=(10,6)) 
cm = plt.cm.get_cmap('RdYlGn')
ax = fig.add_subplot(111, projection='3d')
c = ax.scatter(inputs_dydx_2d_array[:,0],inputs_dydx_2d_array[:,1],dndx_mean_2d_array,c=dndx_mean_2d_array,cmap=cm,alpha=0.9)
ax.set_xlabel(r"$\psi$",labelpad=20)
ax.set_ylabel('Time (s)',labelpad=20)
ax.set_zlabel(r"$\partial$"+"n"+r"$_e / \partial \psi \ (10^{20} \ $"+"m"+r"$^{-3})$")
ax.set_xlim(psi_min-0.1,psi_max+0.1) 
fig.colorbar(c, ax=ax) 
plt.show()  

fig = plt.figure(figsize=(10,6)) 
cm = plt.cm.get_cmap('RdYlGn')
ax = fig.add_subplot(111, projection='3d')
c = ax.scatter(inputs_dydx_2d_array[:,0],inputs_dydx_2d_array[:,1],dTdx_mean_2d_array,c=dTdx_mean_2d_array,cmap=cm,alpha=0.9)
ax.set_xlabel(r"$\psi$",labelpad=20)
ax.set_ylabel('Time (s)',labelpad=20)
ax.set_zlabel(r"$\partial$"+"T"+r"$_e / \partial \psi$"+" (keV)",labelpad=10)
ax.set_xlim(psi_min-0.1,psi_max+0.1) 
fig.colorbar(c, ax=ax) 
plt.show()      

#Files being saved as outputs
np.save('inputs_x_array.npy', inputs_x_array) #2d domain, i.e. time (s) and normalized poloidal flux coordinates for given input data
np.save('ne.npy', y_pred_full_array_zeros) #electron density (m^-3)
np.save('ne_1sigma.npy', y_pred_sigma_full_array) #prediction interval - 1 sigma
np.save('Te.npy', y_pred_T_full_array_zeros) #electron temperature (keV)
np.save('Te_1sigma.npy', y_pred_sigma_T_full_array) #prediction interval - 1 sigma