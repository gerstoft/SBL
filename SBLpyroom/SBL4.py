#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BL_v3p12( A , Y, options )
# 
# SBL reformulates the parameter estimation problem as an underdetermined,
# linear problem. The variables in the problem are treated as Gaussian
# random vectors and evidence maximization is performed using Bayesian
# analysis to obtain a sparse solution. SBL version 3.12 can accomodate  
# multiple snapshots and multiple frequencies. 
#
# Advatages: SBL
# - can handle coherent sources,
# - behaves similar to an adaptive processor,
# - is robust to mismatch,
# - determines sparsity automatically, 
# - is computationally efficient when compared to L1 methods.
#
# Ideally, SBL requires the number of sources to be known for its
# estimate of the noise. 
#
## ----------------------------Inputs:------------------------------------- 
#
# Attention: If Y is single snapshot (and single frequency), it needs to be
# a row vector.
#
# Inputs:
#
# A - Multiple frequency augmented dictionary <f , n , m>
#     f: number of frequencies
#     n: number of sensors
#     m: number of replicas
#   Note: if f==1, A = < n , m >
#
# Y - Multiple snapshot multiple frequency observations <f , n , L>
#     f: number of frequencies
#     n: number of sensors
#     L: number of snapshots
#   Note: if f==1, Y = < n , L >
#
# options - generate with SBLset.m 
#
# Outputs:
#
# gamma <m , 1> - vector containing source power
#
# report - various report options
#
#
#---------------------------Changelog:------------------------------------- 
# Version 1.0
# Code originally written by P. Gerstoft (see http://noiselab.ucsd.edu/)
#
# Version 2.23 (5/16/16)
# Edited to include multiple frequency support: 
#
# Version 3.1 (8/11/16)
# added additional convergence norms 
# A and Y are 3D dimensions - code now works for beamforming or matched
# field processing (A has all replicas or atoms)
# added posterior unbiased mean
# added single snapshot, single frequency support
#
# Version 3.12 (9/21/16)
# improvement on execution time
#
#
# Code written by
# Kay L Gemba & Santosh Nannuru
# gemba@ucsd.edu & snannuru@ucsd.edu
#
# Marine Physical Laboratory
# Scripps Institution of Oceanography
# University of California, San Diego
#
# Selected Publications:

# [1] S. Nannuru, K. L. Gemba, P. Gerstoft, W. S. Hodgkiss, and
# C. F. Mecklenbräuker, “Sparse Bayesian learning with multiple
# dictionaries,” Signal Process., vol. 159, pp. 159–170, Jun. 2019.

# 
#-------------------------------------------------------------------------- 
# Code released under GNU General Public License v3.0
#-------------------------------------------------------------------------- 
from scipy.signal import find_peaks
import numpy as np
from SBLpeaks_1D import *
#import cmath 
#import time
from Options import * 
 
        
#options = Options(10 ** (-6), 200, 3000, 25, 150, 0.1, 1, 3, 0, 1);
def SBL(A, Y, options):
    #number of frequencies
    Nfreq = A.shape[2];

    #slicing
    Nsource = options.Nsource;

 #   if options.tic == 1:
 #      t = time.time();

    # Initialize variables
    # number of sensors
    Nsensor = A.shape[0];
    # number of dictionary entries
    Ntheta = A.shape[1];
    # number of snapshots in the data covariance
    Nsnapshot = Y.shape[1];
    # noise power initializtion
    sigc = np.ones((Nfreq, 1)) ; #* options.noisepower_guess;
    # posterior
 #   x_post = np.zeros((Ntheta, Nsnapshot,Nfreq), dtype = complex);
    # minimum (global) gamma
 #   gmin_global = 10** 10;
    #L1 error
    errornorm = np.zeros((options.convergence_maxiter, 1));

    #Initialize equal and uncorrelated weights
    gamma = np.ones((Ntheta));
    gamma_num = np.zeros(( Ntheta,Nfreq));
    gamma_denum = np.zeros((Ntheta,Nfreq));

    # Sample Covariance Matrix
    SCM = np.zeros(( Nsensor, Nsensor, Nfreq), dtype = complex);

    for iF in range(Nfreq):
        SCM[ :, :,iF] = np.matmul(Y[ :, :,iF], np.conj(Y[ :, :, iF].T))/Nsnapshot;   
        # noise power initializtion
        sigc[iF] = np.real(np.trace(np.squeeze(SCM[:,:,iF]))/Nsensor)
    print('SBL initialized.');

    for j1 in range(options.convergence_maxiter):
        gammaOld = gamma.copy();
        Itheta= np.argwhere(gamma>max(gamma)*options.gamma_range)
        Itheta=Itheta.reshape(len(Itheta))
        gammaSmall=gamma[Itheta]
#        print('Dic size',len(Itheta))
        # gamma update

        for iF in range(Nfreq):
            Af = np.squeeze(A[ :, Itheta,iF]);
            Af_H = np.conj(Af.T);
#            gamma_replica = np.tile(gammaSmall[:,np.newaxis], (1, Nsensor));
#            ApSigmaYinv = Af_H @ np.linalg.inv(sigc[iF] * np.eye(Nsensor) + Af @ (gamma_replica * Af_H));
            ApSigmaYinv = Af_H @ np.linalg.inv(sigc[iF] * np.eye(Nsensor) + Af @ (np.tile(gammaSmall[:,np.newaxis], (1, Nsensor)) * Af_H));
#            ApSigmaYinv = np.conj(Af.T)@ np.linalg.inv(sigc[iF] * np.eye(Nsensor) + Af @ (np.tile(gammaSmall[:,np.newaxis], (1, Nsensor)) * np.conj(Af.T)));
            # Sum over snapshots and normalize, abs for roundoff errors
            gamma_num[ Itheta,iF] = np.sum(np.abs( (ApSigmaYinv @ Y[ :, :,iF]))**2 , axis = 1) / Nsnapshot;
            # postive def quantity, abs for roundoff errors
            gamma_denum[ Itheta,iF] = np.abs(np.sum( ApSigmaYinv.T * Af , axis = 0) );

        # Fixed point Eq. Update
        gamma[Itheta] = gamma[Itheta] * ((np.sum(gamma_num[ Itheta,:], axis = 1) / np.sum(gamma_denum[ Itheta,:], axis=1) ) ** (1/options.fixedpoint))

        ## sigma and L2 error using unbiased posterior update 

        # locate same peaks for all frequencies
        # this takes a long time (replace by max is using Nsource=1)
        # indexes, _ = find_peaks(gamma.reshape(Ntheta));
        # gamma_index = np.zeros(len(indexes));
        # for ii in range(len(indexes)):
        #     gamma_index[ii] = gamma[indexes[ii]];

        # sort = np.argsort(gamma_index);
        # sort_flip = np.flip(sort);
        # Ilocs = indexes[sort_flip[0:(Nsource)]];
        # print(Ilocs)
     #   pks, Ilocs = SBLpeaks_1D.SBLpeaks_1D(gamma,Nsource)
        _, Ilocs = SBLpeaks_1D(gamma, Nsource);
  #      Ilocs = Ilocs.reshape(Nsource).astype(int);
       
  #      print('2',Ilocs)
        Apeak = A[:,  Ilocs,:];

        for iF in range(Nfreq):
            # only active replicas
            Am = Apeak[:,:,iF];
            # noise estimate
            sigc[iF] = np.real(np.matrix.trace( (np.eye(Nsensor) - Am @ np.linalg.pinv(Am)) @ np.squeeze(SCM[:, :,iF])/ (Nsensor - Nsource)))*1.1;

        ## Convergance
        # checks convergance and displays status reports

        # convergance indicator
        errornorm[j1] = np.linalg.norm(gamma - gammaOld, ord = 1) / np.linalg.norm(gamma, ord = 1);

        # look into the past and find best error since then
        # if j1 > options.convergence_min_iteration  and  errornorm[j1] < gmin_global:
        #     gmin_global = errornorm[j1];
        #     gamma_min = gamma;
        #     iteration_L1 = j1;

        if j1 > options.convergence_min_iteration and (errornorm[j1] < options.convergence_error ):

            if options.flag == 1:
                print('Solution converged. Iteration:', str(j1),' Dic size',len(Itheta), '.Error:', str(errornorm[j1]));
            break;

            # not convereged
        elif j1 == options.convergence_maxiter:
            print('Solution not converged. Error: ' + str(errornorm(j1)) + '.');

        elif j1 != options.convergence_maxiter and options.flag == 1 and np.mod(j1, options.status_report) == 0:
            print('Iteration: ', str(j1),' Dic size',len(Itheta),' Error: ', str(errornorm[j1]));

    # Posterior Distribution
    # x_post - posterior unbiased mean
    # for iF in range(Nfreq):
    #     Af = np.squeeze(A[:,:,iF]);
    #     Af_H = np.conj(Af.T);
    #     gamma_replica_sensor = np.tile(gamma[:,np.newaxis], (1, Nsensor));
    #     gamma_replica_snap = np.tile(gamma[:,np.newaxis], (1, Nsnapshot));
    #     x_post[:,:,iF] = gamma_replica_snap * ((Af_H @ np.linalg.inv(sigc[iF] * np.eye(Nsensor) + Af @ (gamma_replica_sensor * Af_H))) @ (Y[:,:,iF]));
 
    # function reture
    #global minimum
  #  gamma = gamma_min
  #  report = Report(results_error = errornorm, results_iteration_L1 = j1 , results_iteration = j1, results_noisepower = sigc,  results_gamma = gamma, results_x_post = x_post, options = options);
    report = Report(results_error = errornorm, results_iteration_L1 = j1 , results_iteration = j1, results_noisepower = sigc,  results_gamma = gamma, options = options);
    return gamma, report
