#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:56:39 2019

@author: mathewsa

This script is used for plotting the length scales learned by the GP across the 
2D (i.e. radial and temporal) domain specified by the user. This script is to
be run only after first running and saving the GP after it has been trained
upon the experimental data. Note that certain trained GPs may have trouble during
training to find good estimates of length scales across the domain, nevertheless
the fits to the original data may still be mostly all right, but checking for 
'good times' which are stored in the array 'inputs_t_array_good' should be 
performed as described in the script 'GP-obtain-2D-profiles.py'.
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
plt.rcParams['font.size'] = 18
       
psi_min = 0.85 #lower limit you want for plotting x-axis
psi_max = 1.05 #upper limit you want for plotting y-axis
T_min = 0.0 #in keV, lower limit you want for plotting y-axis
T_max = 2.0 #in keV, upper limit you want for plotting y-axis
dpsi = 0.01 #normalized poloidal flux coordinate spacing you specify
dt = 0.001 #seconds; this is the grid spacing you specify
t_min = 0.4 #in seconds, lower limit for x-axis for 2d array/plot
t_max = 1.58 #in seconds, upper limit for x-axis for 2d array/plot  
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
n_max_iter = np.load(str(file_path)+'n_max_iter.npy') 

gp = pickle.load(open(str(file_path)+"gp.dump","rb")) 
 
x1 = np.arange(psi_min,psi_max,dpsi) #radial coordinate
x2 = np.arange(t_min,t_max,dt) #temporal coordinate
            
i = 0 
inputs_x = []
while i < len(x1):
    j = 0
    while j < len(x2):
        inputs_x.append([x1[i],x2[j]]) 
        j = j + 1
    i = i + 1 

inputs_x_array = np.array(inputs_x)

lls_len_scale = []
i = 0  
while i < len(inputs_x_array): 
    lls_len_scale_i = gp.kernel_.k1.k2.theta_gp* 10**gp.kernel_.k1.k2.gp_l.predict(inputs_x_array[i].reshape(1, -1))[0] 
    lls_len_scale.append(lls_len_scale_i) 
    i = i + 1 

lls_len_scale = np.array(lls_len_scale) 

fig = plt.figure(figsize=(16,6)) 
cm = plt.cm.get_cmap('RdYlGn')
ax = fig.add_subplot(111, projection='3d')
c = ax.scatter(inputs_x_array[:,0],inputs_x_array[:,1],lls_len_scale,c=lls_len_scale[:,0],cmap=cm,alpha=0.3) 
ax.set_xlabel(r"$\psi$",labelpad=20)
ax.set_ylabel('Time (s)',labelpad=27.5) 
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('GP LLS',labelpad=5,rotation=90) 
ax.set_xlim(0.8,1.1) 
ax.set_ylim(0.4,1.55) 
fig.colorbar(c, ax=ax) 
ax.azim = 25
ax.elev = 20
plt.show() 