#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:56:39 2019

@author: mathewsa
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
from scipy.linalg import cholesky, cho_solve, solve_triangular
import statistics
from scipy import stats 
import gp_extras
from numpy.linalg import inv
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18

def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=n_max_iter, disp=False, polish=True)
    return res.x, obj_func(res.x, eval_gradient=False) 

time = 1.2  #s          
psi_min = 0.85
psi_max = 1.05
T_min = 0.0
T_max = 1.0 #keV
dpsi = 0.01
skip = 0
n_sampling = 1000
number_of_samples = 1
file_path = 'C:/Users/mathewsa/Downloads/USE_THIS_N_GREAT_n_GREAT_1091016033-80.751.250.910.31.7lower-l-0.05upper-l-5.01e-10N_clusters10n_max_iter20TS_only_2D_GPR1578949919.4504375/'#'C:/Users/mathewsa/Downloads/USE_THIS_BEST_N_good_n1091016033-80.751.250.910.31.7lower-l-0.05upper-l-5.01e-10N_clusters10n_max_iter20TS_only_2D_GPR1578948613.1040285/'#'C:/Users/mathewsa/Downloads/1091016033-0.61.32Danalysis1_WORKED__NEW23_psi_minmax0.81.1lower-l-0.05upper-l-5.0BOTH_LLS_WORKEDN_clusters10n_max_iter201578614055.5881824/'#'C:/Users/mathewsa/Downloads/1101014029-0.61.32Danalysis1_NEW1_psi_minmax0.81.1lower-l-0.05upper-l-5.0BOTH_LLS_WORKEDN_clusters10n_max_iter20/'#'C:/Users/mathewsa/Downloads/custom8-0.01.152Danalysis/'#'/home/mathewsa/Desktop/1160718018-0.60.62Danalysis/'#'/home/mathewsa/Desktop/1160930033-0.60.62Danalysis/'#'/home/mathewsa/Desktop/confinement_table/codes+/'#'/home/mathewsa/Desktop/1160708024-analysis/time=096-100/' #'/home/mathewsa/Desktop/confinement_table/codes+/y_T_TS.npy'
  
X_n = np.load(str(file_path)+'X_n.npy')
y_n_TS = np.load(str(file_path)+'y_n_TS.npy')
y_n_TS_err = np.load(str(file_path)+'y_n_TS_err.npy')
X_T = np.load(str(file_path)+'X_T.npy')
y_T_TS = np.load(str(file_path)+'y_T_TS.npy')
y_T_TS_err = np.load(str(file_path)+'y_T_TS_err.npy') 
n_max_iter = np.load(str(file_path)+'n_max_iter.npy') 
time_spacing = 0.001

gp = pickle.load(open(str(file_path)+"gp.dump","rb"))
gp_T = pickle.load(open(str(file_path)+"gp_T.dump","rb"))

inputs_x_array = np.arange(psi_min,psi_max + dpsi,dpsi)
a = np.ones((len(inputs_x_array),2))
a[:,0] = inputs_x_array
a[:,1] = a[:,1]*time
inputs_x_array = inputs_x_array_n = inputs_x_array_T = a 

if gp.err_gp == 0:
    if gp_T.err_gp == 0:
        if skip == 0: 
            
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
            #plt.gca().set_ylim(bottom=0)
            
            ax2 = ax1.twinx() 
            ax2.plot(np.array(inputs_y)[:,0],mean_T,'g-')
            ax2.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
                 np.concatenate([mean_T - 1.96*sigma_T,
                                (mean_T + 1.96*sigma_T)[::-1]]),
            fc='g',ec='None',label='95% prediction interval',alpha=0.1)
            ax2.set_ylabel("T"+r"$_e$"+" (keV)",color='g')
            plt.gca().set_ylim(bottom=0)  
            alpha_n = X_n[:,1] - inputs_x_array[:,1][0]
            alpha_n = 1./(alpha_n/time_spacing)#0.1*np.exp(-(alpha_n**8))
            alpha_n = np.abs(alpha_n)
            alpha_n[alpha_n > 1] = 1.0
            alpha_n[alpha_n < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_n),4)) 
            rgba_colors[:,0] = 1.0 
            rgba_colors[:, 3] = alpha_n 
            plot_X_n = X_n[:,0]
            plot_y_n_TS = y_n_TS
            plot_y_n_TS_err = y_n_TS_err
            #                ax1.errorbar(plot_X_n,plot_y_n_TS,1.96*plot_y_n_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15)
            
            ax1.scatter(plot_X_n,plot_y_n_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_n, plot_y_n_TS, plot_y_n_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
                
            alpha_T = X_T[:,1] - inputs_x_array[:,1][0]
            alpha_T = 1./(alpha_T/time_spacing)#0.1*np.exp(-(alpha_T**8))
            alpha_T = np.abs(alpha_T)
            alpha_T[alpha_T > 1] = 1.0
            alpha_T[alpha_T < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_T),4)) 
            rgba_colors[:,1] = 1.0 
            rgba_colors[:, 3] = alpha_T 
            plot_X_T = X_T[:,0]
            plot_y_T_TS = y_T_TS
            plot_y_T_TS_err = y_T_TS_err
            #                ax2.errorbar(plot_X_T[:,0],plot_y_T_TS,1.96*plot_y_T_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15) 
            
            ax2.scatter(plot_X_T,plot_y_T_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_T, plot_y_T_TS, plot_y_T_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax2.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
            
            plt.title("Time is "+str(np.round(time,3))+"s")
            plt.gca().set_ylim(bottom=0)
            plt.legend()
            ax2.set_ylim(T_min,T_max)
            plt.xlim(psi_min,psi_max) 
            
#non-Heteroscedastic sampling   
X_train = X_n
K_trans1 = gp.kernel_(X_test, X_train)  
K = gp.kernel_(X_train) 
#                            inv_K = inv(gp.kernel_(X_train,X_train) + np.eye(len(X_train))*(y_n_TS_err)**2.)
 
try:
    L_ = cholesky(K, lower=True)
    v1 = cho_solve((L_, True), K_trans1.T) 
    y_cov1 = gp.kernel_(X_test,X_test) - K_trans1.dot(v1) # this is best code and fix to from gp_samples  
    output_cov1 = np.random.multivariate_normal(mean_y_arr,y_cov1,n_sampling).T #(NH - non-heteroscedastic)
    n_samples_NH = output_cov1 
except np.linalg.LinAlgError: 
    skip = 1
    print('Error in cholesky')
#                            L_ = cholesky(nearestPD(K), lower=True) #to ensure LinAlgError: 1303-th leading minor of the array is not positive definite is solved; https://github.com/rlabbe/filterpy/issues/62; https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite - mplementation of Higham’s 1988 paper
#                            y_cov2 = gp.kernel_(X_test,X_test) - np.dot(np.dot(gp.kernel_(X_test,X_train), inv_K),(gp.kernel_(X_test,X_train)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent) 
#                            output_cov2 = np.random.multivariate_normal(mean_y_arr,y_cov2,samples_cholesky).T #(NH - non-heteroscedastic)
#                            n_samples_NH = output_cov2
except:
    skip = 1
    print('Unknown error in cholesky')
 
K_trans1_T = gp_T.kernel_(X_test_T, X_train_T) 
K_T = gp_T.kernel_(X_train_T)
#                            inv_K_T = inv(gp_T.kernel_(X_train_T,X_train_T) + np.eye(len(X_train_T))*(y_T_TS_err)**2.) 

try:
    L__T = cholesky(K_T, lower=True)
    v1_T = cho_solve((L__T, True), K_trans1_T.T) 
    y_cov1_T = gp_T.kernel_(X_test_T,X_test_T) - K_trans1_T.dot(v1_T) # this is best code and fix to from gp_samples  
    output_cov1_T = np.random.multivariate_normal(mean_y_arr_T,y_cov1_T,n_sampling).T #(NH - non-heteroscedastic)
    T_samples_NH = output_cov1_T
    
except np.linalg.LinAlgError: 
    skip = 1
    print('Error in cholesky')
#                            L__T = cholesky(nearestPD(K_T), lower=True) #to ensure LinAlgError: 1303-th leading minor of the array is not positive definite is solved; https://github.com/rlabbe/filterpy/issues/62; https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite - mplementation of Higham’s 1988 paper
#                            y_cov2_T = gp_T.kernel_(X_test_T,X_test_T) - np.dot(np.dot(gp_T.kernel_(X_test_T,X_train_T), inv_K_T),(gp_T.kernel_(X_test_T,X_train_T)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent)
#                            K_trans_T = gp_T.kernel_(X_test_T,X_train_T) 
#                            output_cov2_T = np.random.multivariate_normal(mean_y_arr_T,y_cov2_T,samples_cholesky).T #(NH - non-heteroscedastic)
#                            T_samples_NH = output_cov2_T
except:
    skip = 1
    print('Unknown error in cholesky')

if gp.err_gp == 0:
    if gp_T.err_gp == 0:
        if skip == 0:   
             
            i_index = 0
            mean_n_NH = []
            sigma_n_NH = []
            mean_T_NH = []
            sigma_T_NH = []
            while i_index < len(inputs_y):  
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
            #plt.gca().set_ylim(bottom=0)
            
            ax2 = ax1.twinx() 
            ax2.plot(np.array(inputs_y)[:,0],mean_T_NH,'g-')
            ax2.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
                 np.concatenate([mean_T_NH - 1.96*sigma_T_NH,
                                (mean_T_NH + 1.96*sigma_T_NH)[::-1]]),
            fc='g',ec='None',label='95% confidence interval',alpha=0.1)
            ax2.set_ylabel("T"+r"$_e$"+" (keV)",color='g')
            plt.gca().set_ylim(bottom=0)
             
            alpha_n = X_n[:,1] - inputs_x_array[:,1][0]
            alpha_n = 1./(alpha_n/time_spacing)#0.1*np.exp(-(alpha_n**8))
            alpha_n = np.abs(alpha_n)
            alpha_n[alpha_n > 1] = 1.0
            alpha_n[alpha_n < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_n),4)) 
            rgba_colors[:,0] = 1.0 
            rgba_colors[:, 3] = alpha_n 
            plot_X_n = X_n[:,0]
            plot_y_n_TS = y_n_TS
            plot_y_n_TS_err = y_n_TS_err
            #                ax1.errorbar(plot_X_n,plot_y_n_TS,1.96*plot_y_n_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15)
            
            ax1.scatter(plot_X_n,plot_y_n_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_n, plot_y_n_TS, plot_y_n_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
                
            alpha_T = X_T[:,1] - inputs_x_array[:,1][0]
            alpha_T = 1./(alpha_T/time_spacing)#0.1*np.exp(-(alpha_T**8))
            alpha_T = np.abs(alpha_T)
            alpha_T[alpha_T > 1] = 1.0
            alpha_T[alpha_T < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_T),4)) 
            rgba_colors[:,1] = 1.0 
            rgba_colors[:, 3] = alpha_T 
            plot_X_T = X_T[:,0]
            plot_y_T_TS = y_T_TS
            plot_y_T_TS_err = y_T_TS_err
            #                ax2.errorbar(plot_X_T[:,0],plot_y_T_TS,1.96*plot_y_T_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15) 
            
            ax2.scatter(plot_X_T,plot_y_T_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_T, plot_y_T_TS, plot_y_T_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax2.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
            
            plt.title("Time is "+str(np.round(time,3))+"s")
            plt.gca().set_ylim(bottom=0)
            plt.legend()
            ax2.set_ylim(T_min,T_max)
            plt.xlim(psi_min,psi_max) 
            
             
if gp.err_gp == 0:
    if gp_T.err_gp == 0:
        if skip == 0: 
            
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
            
            n_samples = gp.sample_y(inputs_x_array,number_of_samples)
            T_samples = gp_T.sample_y(inputs_x_array,number_of_samples) 
            
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
            
            n_samples = n_samples[:,0][:,0]
            T_samples = T_samples[:,0][:,0]
            fig2 = plt.figure(figsize=(10,6))
            plt.clf()  
            ax1 = fig2.add_subplot(111) 
            ax1.plot(inputs_x_array[:,0],n_samples,'r-') 
            ax1.set_xlabel(r"$\psi$")
            ax1.set_ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 
            plt.gca().set_ylim(bottom=0)
            
            ax2 = ax1.twinx()
            ax2.plot(inputs_x_array[:,0],T_samples,'g-') 
            ax2.set_ylabel("T"+r"$_e$"+" (keV)",color='g')
            plt.gca().set_ylim(bottom=0)  
#            plt.clf()  
#            ax1 = fig2.add_subplot(111) 
#            ax1.plot(np.array(inputs_y)[:,0],mean_n,'r-')
#            ax1.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
#                 np.concatenate([mean_n - 1.96*sigma_n,
#                                (mean_n + 1.96*sigma_n)[::-1]]),
#            fc='r',ec='None',label='95% confidence interval',alpha=0.1)
#            ax1.set_xlabel(r"$\psi$")
#            ax1.set_ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 
#            #plt.gca().set_ylim(bottom=0)
#            
#            ax2 = ax1.twinx() 
#            ax2.plot(np.array(inputs_y)[:,0],mean_T,'g-')
#            ax2.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
#                 np.concatenate([mean_T - 1.96*sigma_T,
#                                (mean_T + 1.96*sigma_T)[::-1]]),
#            fc='g',ec='None',label='95% confidence interval',alpha=0.1)
#            ax2.set_ylabel("T"+r"$_e$"+" (keV)",color='g')
            #plt.gca().set_ylim(bottom=0)  
            alpha_n = X_n[:,1] - inputs_x_array[:,1][0]
            alpha_n = 1./(alpha_n/time_spacing)#0.1*np.exp(-(alpha_n**8))
            alpha_n = np.abs(alpha_n)
            alpha_n[alpha_n > 1] = 1.0
            alpha_n[alpha_n < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_n),4)) 
            rgba_colors[:,0] = 1.0 
            rgba_colors[:, 3] = alpha_n 
            plot_X_n = X_n[:,0]
            plot_y_n_TS = y_n_TS
            plot_y_n_TS_err = y_n_TS_err
            #                ax1.errorbar(plot_X_n,plot_y_n_TS,1.96*plot_y_n_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15)
            
            ax1.scatter(plot_X_n,plot_y_n_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_n, plot_y_n_TS, plot_y_n_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
                
            alpha_T = X_T[:,1] - inputs_x_array[:,1][0]
            alpha_T = 1./(alpha_T/time_spacing)#0.1*np.exp(-(alpha_T**8))
            alpha_T = np.abs(alpha_T)
            alpha_T[alpha_T > 1] = 1.0
            alpha_T[alpha_T < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_T),4)) 
            rgba_colors[:,1] = 1.0 
            rgba_colors[:, 3] = alpha_T 
            plot_X_T = X_T[:,0]
            plot_y_T_TS = y_T_TS
            plot_y_T_TS_err = y_T_TS_err
            #                ax2.errorbar(plot_X_T[:,0],plot_y_T_TS,1.96*plot_y_T_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15) 
            
            ax2.scatter(plot_X_T,plot_y_T_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_T, plot_y_T_TS, plot_y_T_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax2.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
            
            plt.title("Time is "+str(np.round(time,3))+"s")
            plt.gca().set_ylim(bottom=0)
            plt.legend()
            ax2.set_ylim(T_min,T_max)
            plt.xlim(psi_min,psi_max) 
               
#non-Heteroscedastic sample   
X_train = X_n
K_trans1 = gp.kernel_(X_test, X_train)  
K = gp.kernel_(X_train) 
#                            inv_K = inv(gp.kernel_(X_train,X_train) + np.eye(len(X_train))*(y_n_TS_err)**2.)
 
try:
    L_ = cholesky(K, lower=True)
    v1 = cho_solve((L_, True), K_trans1.T) 
    y_cov1 = gp.kernel_(X_test,X_test) - K_trans1.dot(v1) # this is best code and fix to from gp_samples  
    output_cov1 = np.random.multivariate_normal(mean_y_arr,y_cov1,number_of_samples).T #(NH - non-heteroscedastic)
    n_samples_NH = output_cov1 
except np.linalg.LinAlgError: 
    skip = 1
    print('Error in cholesky')
#                            L_ = cholesky(nearestPD(K), lower=True) #to ensure LinAlgError: 1303-th leading minor of the array is not positive definite is solved; https://github.com/rlabbe/filterpy/issues/62; https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite - mplementation of Higham’s 1988 paper
#                            y_cov2 = gp.kernel_(X_test,X_test) - np.dot(np.dot(gp.kernel_(X_test,X_train), inv_K),(gp.kernel_(X_test,X_train)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent) 
#                            output_cov2 = np.random.multivariate_normal(mean_y_arr,y_cov2,samples_cholesky).T #(NH - non-heteroscedastic)
#                            n_samples_NH = output_cov2
except:
    skip = 1
    print('Unknown error in cholesky')
 
K_trans1_T = gp_T.kernel_(X_test_T, X_train_T) 
K_T = gp_T.kernel_(X_train_T)
#                            inv_K_T = inv(gp_T.kernel_(X_train_T,X_train_T) + np.eye(len(X_train_T))*(y_T_TS_err)**2.) 

try:
    L__T = cholesky(K_T, lower=True)
    v1_T = cho_solve((L__T, True), K_trans1_T.T) 
    y_cov1_T = gp_T.kernel_(X_test_T,X_test_T) - K_trans1_T.dot(v1_T) # this is best code and fix to from gp_samples  
    output_cov1_T = np.random.multivariate_normal(mean_y_arr_T,y_cov1_T,number_of_samples).T #(NH - non-heteroscedastic)
    T_samples_NH = output_cov1_T
    
except np.linalg.LinAlgError: 
    skip = 1
    print('Error in cholesky')
#                            L__T = cholesky(nearestPD(K_T), lower=True) #to ensure LinAlgError: 1303-th leading minor of the array is not positive definite is solved; https://github.com/rlabbe/filterpy/issues/62; https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite - mplementation of Higham’s 1988 paper
#                            y_cov2_T = gp_T.kernel_(X_test_T,X_test_T) - np.dot(np.dot(gp_T.kernel_(X_test_T,X_train_T), inv_K_T),(gp_T.kernel_(X_test_T,X_train_T)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent)
#                            K_trans_T = gp_T.kernel_(X_test_T,X_train_T) 
#                            output_cov2_T = np.random.multivariate_normal(mean_y_arr_T,y_cov2_T,samples_cholesky).T #(NH - non-heteroscedastic)
#                            T_samples_NH = output_cov2_T
except:
    skip = 1
    print('Unknown error in cholesky')

if gp.err_gp == 0:
    if gp_T.err_gp == 0:
        if skip == 0:  
            fig2 = plt.figure(figsize=(10,6))  
            plt.clf() 
            ax1 = fig2.add_subplot(111)
            ax1.plot(inputs_x_array_n[:,0],n_samples_NH,'r-') 
            ax1.set_xlabel(r"$\psi$")
            ax1.set_ylabel("n"+r"$_e \ (10^{20} \ $"+"m"+r"$^{-3})$",color='r') 
            plt.gca().set_ylim(bottom=0)
            
            ax2 = ax1.twinx()
            ax2.plot(inputs_x_array_T[:,0],T_samples_NH,'g-') 
            ax2.set_ylabel("T"+r"$_e$"+" (keV)",color='g')
            #ax2.set_ylim(0.,1.4)
             
            alpha_n = X_n[:,1] - inputs_x_array[:,1][0]
            alpha_n = 1./(alpha_n/time_spacing)#0.1*np.exp(-(alpha_n**8))
            alpha_n = np.abs(alpha_n)
            alpha_n[alpha_n > 1] = 1.0
            alpha_n[alpha_n < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_n),4)) 
            rgba_colors[:,0] = 1.0 
            rgba_colors[:, 3] = alpha_n 
            plot_X_n = X_n[:,0]
            plot_y_n_TS = y_n_TS
            plot_y_n_TS_err = y_n_TS_err
            #                ax1.errorbar(plot_X_n,plot_y_n_TS,1.96*plot_y_n_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15)
            
            ax1.scatter(plot_X_n,plot_y_n_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_n, plot_y_n_TS, plot_y_n_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax1.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
                
            alpha_T = X_T[:,1] - inputs_x_array[:,1][0]
            alpha_T = 1./(alpha_T/time_spacing)#0.1*np.exp(-(alpha_T**8))
            alpha_T = np.abs(alpha_T)
            alpha_T[alpha_T > 1] = 1.0
            alpha_T[alpha_T < 0.01] = 0.0
            rgba_colors = np.zeros((len(X_T),4)) 
            rgba_colors[:,1] = 1.0 
            rgba_colors[:, 3] = alpha_T 
            plot_X_T = X_T[:,0]
            plot_y_T_TS = y_T_TS
            plot_y_T_TS_err = y_T_TS_err
            #                ax2.errorbar(plot_X_T[:,0],plot_y_T_TS,1.96*plot_y_T_TS_err, markerfacecolor=rgba_colors, fmt='.', markersize=15) 
            
            ax2.scatter(plot_X_T,plot_y_T_TS, c=rgba_colors, edgecolors=rgba_colors)
            for pos, ypt, err, color in zip(plot_X_T, plot_y_T_TS, plot_y_T_TS_err, rgba_colors):
                plotline, caplines, (barlinecols,) = ax2.errorbar(pos, ypt, err, lw=2, color=color, capsize=5, capthick=2)
            
            plt.title("Time is "+str(np.round(time,3))+"s")
            plt.gca().set_ylim(bottom=0)
            plt.legend()
            ax2.set_ylim(T_min,T_max)
            plt.xlim(psi_min,psi_max) 
            
#import MDSplus 
#from MDSplus import *
#import numpy as np   
#import os
#import sys
#from os import getenv   
#tree = Tree('cmod', shot) 
#spectroscopy = MDSplus.Tree('SPECTROSCOPY', shot) 
#z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).data() 
#time_z_ave = (spectroscopy.getNode('\SPECTROSCOPY::z_ave')).dim_of().data()
#plt.figure()
#plt.plot(time_z_ave,z_ave)
#plt.xlabel('Time (s)')
#plt.ylabel('Z')
#plt.show()
            
            
            
#K_trans1 = gp.kernel_(X_test, X_train)
#K_ss0 = gp.kernel_(X_test) 
#K_ss1 = gp.kernel_(X_test,X_test) 
#K = gp.kernel_(X_train)
#L_ = cholesky(K, lower=True)
##L_1 = np.linalg.cholesky(K)
#v1 = cho_solve((L_, True), K_trans1.T)  # Line 5
#inv_K = inv(gp.kernel_(X_train,X_train) + np.eye(len(X_train))*(y_n_TS_err)**2.)
#y_cov0 = gp.kernel_(X_test) - K_trans1.dot(v1) # this is code from gp_samples 
#y_cov1 = gp.kernel_(X_test,X_test) - K_trans1.dot(v1) # this is best code and fix to from gp_samples 
#y_cov2 = gp.kernel_(X_test,X_test) - np.dot(np.dot(gp.kernel_(X_test,X_train), inv_K),(gp.kernel_(X_test,X_train)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent)

#L_init = np.linalg.cholesky(K + 0.00005*np.eye(len(X_train)))
#K_s = gp.kernel_(X_train, X_test)
#Lk = np.linalg.solve(L_init, K_s)
#L = np.linalg.cholesky(K_ss0 + 1e-6*np.eye(len(X_test)) - np.dot(Lk.T, Lk))
#f_post = mean_n + np.dot(L, np.random.normal(size=(len(X_test),1)))[:,0]
#plt.figure()
#plt.plot(X_test[:,0],f_post)
#plt.show()
#f_post = mean_n + np.dot(L, np.linspace(0.,1.,len(L)))
#plt.figure()
#plt.plot(X_test[:,0],f_post)
#plt.show()


#K_trans1_T = gp_T.kernel_(X_test, X_train_T) 
#K_T = gp_T.kernel_(X_train_T)
#L__T = cholesky(K_T, lower=True)
##L_1 = np.linalg.cholesky(K)
#v1_T = cho_solve((L__T, True), K_trans1_T.T)  # Line 5
#inv_K_T = inv(gp_T.kernel_(X_train_T,X_train_T) + np.eye(len(X_train_T))*(y_T_TS_err)**2.)
#y_cov0_T = gp_T.kernel_(X_test) - K_trans1_T.dot(v1_T) # this is code from gp_samples 
#y_cov1_T = gp_T.kernel_(X_test,X_test) - K_trans1_T.dot(v1_T) # this is best code and fix to from gp_samples 
#y_cov2_T = gp_T.kernel_(X_test,X_test) - np.dot(np.dot(gp_T.kernel_(X_test,X_train_T), inv_K_T),(gp_T.kernel_(X_test,X_train_T)).T) # this is code I created, where there seems to be small deviation from RHS terms (i,e, second terms in equation should be equivalent)


#output_cov0 = np.random.multivariate_normal(mean_y_arr,y_cov0,samples_cholesky).T 
#output_cov1 = np.random.multivariate_normal(mean_y_arr,y_cov1,samples_cholesky).T 
#output_cov2 = np.random.multivariate_normal(mean_y_arr,y_cov2,samples_cholesky).T
#
#output_cov0_T = np.random.multivariate_normal(mean_y_arr_T,y_cov0_T,samples_cholesky).T 
#output_cov1_T = np.random.multivariate_normal(mean_y_arr_T,y_cov1_T,samples_cholesky).T 
#output_cov2_T = np.random.multivariate_normal(mean_y_arr_T,y_cov2_T,samples_cholesky).T

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
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
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
        _ = la.cholesky(B)
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
plt.show()
f_post = mean_n_true + np.dot(y_cov_L, -np.linspace(0.,1.,len(X_test)))
plt.figure()
plt.plot(X_test[:,0],f_post)
plt.plot(np.array(inputs_y)[:,0],mean_n_true,'r-') 
plt.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_n_true - 1.96*sigma_n_true,
                    (mean_n_true + 1.96*sigma_n_true)[::-1]]),
fc='r',ec='None',label='95% prediction interval',alpha=0.1)
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
plt.show()
f_post_T = mean_T_true + np.dot(y_cov_L_T, -np.linspace(0.,0.5,len(X_test)))
plt.figure()
plt.plot(X_test[:,0],f_post_T)
plt.plot(np.array(inputs_y)[:,0],mean_T_true,'g-') 
plt.fill(np.concatenate([np.array(inputs_y)[:,0],np.array(inputs_y)[:,0][::-1]]),
     np.concatenate([mean_T_true - 1.96*sigma_T_true,
                    (mean_T_true + 1.96*sigma_T_true)[::-1]]),
fc='g',ec='None',label='95% prediction interval',alpha=0.1)
plt.show()

lower, upper = 1.0, 0.0
mu, sigma = 0.0, 1.0
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)
 
fig, ax = plt.subplots(2, sharex=True)
ax[0].hist(X.rvs(10000), 50, normed=True)
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
plt.show()  



x1new = np.arange(0.85,1.05,0.01) #radial coordinate
x2new = np.arange(0.4,1.58,0.001) #temporal coordinate
            
i = 0 
inputs_xnew = []
while i < len(x1new):
    j = 0
    while j < len(x2new):
        inputs_xnew.append([x1new[i],x2new[j]]) 
        j = j + 1
    i = i + 1 

inputs_xnew_array = np.array(inputs_xnew)

lls_len_scalenew = []
i = 0  
while i < len(inputs_xnew_array): 
    lls_len_scalenew_i = gp.kernel_.k1.k2.theta_gp* 10**gp.kernel_.k1.k2.gp_l.predict(inputs_xnew_array[i].reshape(1, -1))[0] 
    lls_len_scalenew.append(lls_len_scalenew_i) 
    i = i + 1 

lls_len_scalenew = np.array(lls_len_scalenew)

from mpl_toolkits.mplot3d import Axes3D  

fig = plt.figure(figsize=(16,6)) 
cm = plt.cm.get_cmap('RdYlGn')
ax = fig.add_subplot(111, projection='3d')
c = ax.scatter(inputs_xnew_array[:,0],inputs_xnew_array[:,1],lls_len_scalenew,c=lls_len_scalenew[:,0],cmap=cm,alpha=0.3) 
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


import h5py
h5f = h5py.File('Figure6.h5', 'w') 
h5f.create_dataset('Figure6x', data=inputs_xnew_array[:,0])  
h5f.create_dataset('Figure6y', data=inputs_xnew_array[:,1])  
h5f.create_dataset('Figure6z', data=lls_len_scalenew)  
h5f.close()