#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:56:39 2019

This script provides an examples on how to train the adaptive heteroscedastic
Gaussian process on experimental measurements of electron density and temperature,
which are provided by the user, over the 2D (i.e. radial and temporal) grid. 

The GP and its necessary contents are then pickled and saved for future usage. 

@author: mathewsa
"""
import os
import sys
sys.path.append('C:/Users/mathewsa/') #provides path to gp_extras 
import pickle
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C 
from gp_extras.kernels import HeteroscedasticKernel, LocalLengthScalesKernel 
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans
import gp_extras 
import timeit #simply used for testing time performance
   
lower_l = 0.05 #lower length scale as specified by the user
upper_l = 5.0 #upper length scale as specified by the user
n_max_iter = 20 #number of training iterations
N_clusters = 20 #number of cluster centers for the k-means algorithm
file_path = '.../trainedGPs/saved_GP_1091016033/' #source of experimental data which is loaded below
newpath_save = '.../trainedGPs/saved_GP_1091016033_v2' #save data to user specified folder

#Loading experimental Thomson diagnostic data for training GP
X_n = np.load(str(file_path)+'X_n.npy')
y_n_TS = np.load(str(file_path)+'y_n_TS.npy')
y_n_TS_err = np.load(str(file_path)+'y_n_TS_err.npy')
X_T = np.load(str(file_path)+'X_T.npy')
y_T_TS = np.load(str(file_path)+'y_T_TS.npy')
y_T_TS_err = np.load(str(file_path)+'y_T_TS_err.npy')

# --------------------------------------------------------------
#                       End of user inputs
# -------------------------------------------------------------- 
                
prototypes = KMeans(n_clusters=N_clusters).fit(X_n).cluster_centers_ 

def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=n_max_iter, disp=False, polish=True)
    return res.x, obj_func(res.x, eval_gradient=False)

kernel_lls = C(1.0, (1e-10, 1000)) \
  * LocalLengthScalesKernel.construct(X_n, l_L=lower_l, l_U=upper_l, l_samples=10)\
      + HeteroscedasticKernel.construct(prototypes, 1e-1, (1e-5, 50.0),
              gamma=1.0, gamma_bounds="fixed")
 
finish_training_n = 0
while finish_training_n == 0:
    print("Start GPR training - density of length ="+str(len(X_n)))
    start = timeit.default_timer()
    #density fitting 
    gp = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer, alpha = (y_n_TS_err)**2.)
     
    try:
        gp.fit(X_n, y_n_TS.reshape(-1,1)) 
        finish_training_n = 1
    except MemoryError:
        print('Memory error')  
    except:
        print('Non-memory error')  
    
    stop = timeit.default_timer()
    print('Time: ', stop - start) 

prototypes = KMeans(n_clusters=N_clusters).fit(X_T).cluster_centers_ 
kernel_lls_T = C(1.0, (1e-10, 1000)) \
  * LocalLengthScalesKernel.construct(X_T, l_L=lower_l, l_U=upper_l, l_samples=10)\
      + HeteroscedasticKernel.construct(prototypes, 1e-1, (1e-5, 50.0),
              gamma=1.0, gamma_bounds="fixed")

finish_training_T = 0
while finish_training_T == 0: 
    print("Start GPR training - temperature of length ="+str(len(X_T)))
    start = timeit.default_timer()
         
    gp_T = GaussianProcessRegressor(kernel=kernel_lls_T, optimizer=de_optimizer, alpha = (y_T_TS_err)**2.)
    
    try:
        gp_T.fit(X_T, y_T_TS.reshape(-1,1))
        finish_training_T = 1
    except MemoryError:
        print('Memory error')  
    except:
        print('Non-memory error')  
        
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
                 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder(newpath_save) 
 
with open(str(newpath_save)+'/'+"gp.dump" , "wb") as f:
     pickle.dump(gp, f)  
with open(str(newpath_save)+'/'+"gp_T.dump" , "wb") as f:
     pickle.dump(gp_T, f)   

np.save(str(newpath_save)+'/'+'X_n.npy', X_n)
np.save(str(newpath_save)+'/'+'y_n_TS.npy', y_n_TS)
np.save(str(newpath_save)+'/'+'y_n_TS_err.npy', y_n_TS_err)
np.save(str(newpath_save)+'/'+'X_T.npy', X_T)
np.save(str(newpath_save)+'/'+'y_T_TS.npy', y_T_TS)
np.save(str(newpath_save)+'/'+'y_T_TS_err.npy', y_T_TS_err) 
np.save(str(newpath_save)+'/'+'n_max_iter.npy', n_max_iter) 