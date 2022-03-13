import numpy as np

class Options:
    def __init__(self, 
            convergence_error, 
            gamma_range, 
            convergence_maxiter, 
            convergence_min_iteration, 
            status_report,  
            fixedpoint, 
            Nsource, 
            flag,
            ):
        self.convergence_error = convergence_error
#        self.convergence_delay = convergence_delay
        self.gamma_range = gamma_range
        self.convergence_maxiter = convergence_maxiter
        self.convergence_min_iteration = convergence_min_iteration
        self.status_report = status_report
#        self.noisepower_guess = noisepower_guess
        self.fixedpoint = fixedpoint
        self.Nsource = Nsource
        self.flag = flag
#        self.tic = tic

class Report:
    def __init__(self, results_error, results_iteration_L1, results_iteration, results_noisepower, results_gamma,  options):
        self.results_error = results_error
        self.results_iteration_L1 = results_iteration_L1
        self.results_iteration = results_iteration
        self.results_noisepower = results_noisepower
        self.results_gamma = results_gamma
     #   self.results_x_post = results_x_post
        self.options = options;              
        
def SBLpeaks_1D(gamma, Nsources):
    pks    = np.zeros((Nsources));
    locs   = np.zeros((Nsources),dtype = int);
    Ntheta = len(gamma)
    gamma  = gamma.reshape(Ntheta)
    gamma_new = np.zeros((Ntheta+2)); # zero padding on the boundary
    gamma_new[1:Ntheta+1] = gamma;
    Ilocs  = np.flip(gamma.argsort(axis = 0));
    npeaks = 0;         # current number of peaks found
    local_patch=np.zeros((Nsources));
    for ii in range(Ntheta):
        # local patch area surrounding the current array entry i.e. (r,c)
        # local_patch = gamma_new[(Ilocs[ii]):(Ilocs[ii]+3)];
        # zero the center
        # local_patch[1] = 0;
        local_patch = [gamma_new[(Ilocs[ii])], 0, gamma_new[(Ilocs[ii]+2)]];
        # zero the center
        if sum(gamma[Ilocs[ii]] > local_patch) == 3:
            pks[npeaks] = gamma[Ilocs[ii]];
            locs[npeaks] = Ilocs[ii];
            npeaks = npeaks + 1;
            # if found sufficient peaks, break
            if npeaks == Nsources:
                break;

    return pks, locs
        
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
# Version 4 (unknown)
#
# Version 4.01 (September 21)
# code cleanup and reorg mha
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
def sbl_func(x,A,options): 
    # wrapper for multiprocessing via partial function
    return SBL(A,x,options)[0]
 
#options = Options(10 ** (-6), 200, 3000, 25, 150, 0.1, 1, 3, 0, 1)
def SBL(A, Y, options):
    # Initialize variables
    Nsource   = options.Nsource
    Nfreq     = A.shape[2];# number of frequencies
    Nsensor   = A.shape[0];# number of sensors
    Ntheta    = A.shape[1];# number of dictionary entries
    Nsnapshot = Y.shape[1];# number of snapshots in the data covariance
    errornorm = np.zeros(options.convergence_maxiter); #L1 error

    #Initialize equal and uncorrelated weights
    # for iF in range(Nfreq):
    #     Af_H = np.conj(np.squeeze(A[ :, :,iF])).T;    
    #     Yf = np.squeeze(Y[ :, :,iF])
    #     gamma=gamma+ np.sum(np.abs(Af_H @ Yf)**2) / Nsnapshot; #CBF sum.
    gamma       = np.ones((Ntheta))

    # gamma_init = conventional beamformer
    SCM   = np.einsum('nsf,msf->nmf',Y,np.conj(Y),optimize=True)/Nsnapshot
    AHA   = np.einsum('ndf,mdf->nmdf',np.conj(A),A,optimize=True)
    gamma = np.real(np.einsum('nmf,nmdf->d',SCM,AHA,optimize=True))

    print('{:_<30}'.format(' > SBL with {:d} thetas'.format(Ntheta)),end="")
    maxnoise = np.real(np.trace(SCM))/Nsensor # noise power initializtion
    sigc  = maxnoise
    # print('SBL initialized.')
    for j1 in range(options.convergence_maxiter):
        gammaOld = gamma.copy()

        Itheta   = np.argwhere(gamma>max(gamma)*options.gamma_range)
        Itheta   = Itheta.reshape(len(Itheta))
        gammaSmall = gamma[Itheta]

        gamma_num   = np.zeros((Ntheta,Nfreq))
        gamma_denum = np.zeros((Ntheta,Nfreq))
        for iF in range(Nfreq):
            Af   = np.squeeze(A[ :, Itheta,iF])
            if len(Itheta)==1:
                Af = np.expand_dims(Af,1)
            ApSigmaYinv = np.conj(Af.T) @ np.linalg.inv(sigc[iF] * np.eye(Nsensor) + Af @ (gammaSmall[:,np.newaxis] * np.conj(Af.T)))
            # Sum over snapshots and normalize, abs for roundoff errors
            gamma_num  [ Itheta,iF] = np.sum(np.abs(ApSigmaYinv @ Y[ :, :,iF])**2 , axis = 1) / Nsnapshot
            # postive def quantity, abs for roundoff errors
            gamma_denum[ Itheta,iF] = np.abs(np.sum(ApSigmaYinv.T * Af , axis = 0))


        gamma[Itheta] *= (np.sum(gamma_num[Itheta],1) / np.sum(gamma_denum[Itheta],1))**(1/options.fixedpoint)
        # Fixed point Eq. Update
        # gamma[Itheta] *= (np.sum(gamma_num,1) / np.sum(gamma_denum,1)**(1/options.fixedpoint)[Itheta]

        ## sigma and L2 error using unbiased posterior update 

        # locate same peaks for all frequencies
        # this takes a long time (replace by max is using Nsource=1)
        # from scipy.signal import find_peaks
        # indexes, _ = find_peaks(gamma.reshape(Ntheta))
        # gamma_index = np.zeros(len(indexes))
        # for ii in range(len(indexes)):
        #     gamma_index[ii] = gamma[indexes[ii]]

        # sort = np.argsort(gamma_index)
        # sort_flip = np.flip(sort)
        # Ilocs = indexes[sort_flip[0:(Nsource)]]
        # print(Ilocs)
        _, Ilocs = SBLpeaks_1D(gamma, Nsource)
        # Ilocs = Ilocs.reshape(Nsource).astype(int)
        # Apeak = A[:,  Ilocs,:]
        # print('2',Ilocs)

        for iF in range(Nfreq):
            # noise estimate, only active replicas
            sigc[iF] = np.real(np.matrix.trace( (np.eye(Nsensor) - A[:,Ilocs,iF] @ np.linalg.pinv(A[:,Ilocs,iF])) @ np.squeeze(SCM[:, :,iF])/ (Nsensor - Nsource)))
            sigc[iF] = np.minimum(sigc[iF],maxnoise[iF]);           #cant be larger than signal+noise.
            sigc[iF] = np.maximum(sigc[iF],maxnoise[iF]*10**(-10)); #snr>100 is unlikely larger than signal.

        ## Convergence
        # checks convergence and displays status reports
        errornorm[j1] = np.linalg.norm(gamma - gammaOld, ord = 1) / np.linalg.norm(gamma, ord = 1)

        # look into the past and find best error since then
        # if j1 > options.convergence_min_iteration  and  errornorm[j1] < gmin_global:
        #     gmin_global = errornorm[j1];
        #     gamma_min = gamma;
        #     iteration_L1 = j1;

        if j1 > options.convergence_min_iteration and (errornorm[j1] < options.convergence_error ):
            if options.flag == 1:
                print('Solution converged. Iteration:', str(j1),' Dic size',len(Itheta), '.Error:', str(errornorm[j1]))
            break
        # not convereged
        elif j1 == options.convergence_maxiter:
            print('Solution not converged. Error: ' + str(errornorm[j1]) + '.')
        elif j1 != options.convergence_maxiter and options.flag == 1 and np.mod(j1, options.status_report) == 0:
            print('Iteration: ', str(j1),' Dic size',len(Itheta),' Error: ', str(errornorm[j1]))

    # Posterior Distribution
    # x_post = np.zeros((Ntheta, Nsnapshot,Nfreq), dtype = complex); # posterior unbiased mean
    # for iF in range(Nfreq):
    #     Af = np.squeeze(A[:,:,iF])
    #     Af_H = np.conj(Af.T)
    #     gamma_replica_sensor = np.tile(gamma[:,np.newaxis], (1, Nsensor))
    #     gamma_replica_snap = np.tile(gamma[:,np.newaxis], (1, Nsnapshot))
    #     x_post[:,:,iF] = gamma_replica_snap * ((Af_H @ np.linalg.inv(sigc[iF] * np.eye(Nsensor) + Af @ (gamma_replica_sensor * Af_H))) @ (Y[:,:,iF]))
 
    # function reture
    # global minimum
    # gamma = gamma_min
    print('{:_<20}'.format('Error {:.6f} #Iter {:d}'.format(errornorm[j1],j1)))
    report = Report(results_error = errornorm, results_iteration_L1 = j1 , 
            results_iteration = j1, results_noisepower = sigc,  
            results_gamma = gamma, options = options)
    return gamma, report
