# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:04:49 2020

@author: lenovo
"""

import numpy as np
# fast alternative for findpeaks in 1D case

def SBLpeaks_1D(gamma, Nsources):
    # [pks, locs] = SBLpeaks_1D(gamma, Nsources)
    # output variables
    pks = np.zeros((Nsources));
    locs = np.zeros((Nsources),dtype=int);
    Ntheta=len(gamma)
    gamma=gamma.reshape(Ntheta)
    # zero padding on the boundary
    gamma_new = np.zeros((Ntheta+2));
    gamma_new[1:Ntheta+1] = gamma;
    Ilocs = np.flip(gamma.argsort(axis = 0));
    # current number of peaks found
    npeaks = 0;
    local_patch=np.zeros((Nsources));
    for ii in range(Ntheta):
        # local patch area surrounding the current array entry i.e. (r,c)
 #       local_patch = gamma_new[(Ilocs[ii]):(Ilocs[ii]+3)];
        # zero the center
   #     local_patch[1] = 0;
        local_patch = [gamma_new[(Ilocs[ii])], 0, gamma_new[(Ilocs[ii]+2)]];
        # zero the center
        if sum(gamma[Ilocs[ii]] > local_patch) == 3:
            pks[npeaks] = gamma[Ilocs[ii]];
            locs[npeaks] = Ilocs[ii];
            npeaks = npeaks + 1;
            # if found sufficient peaks, break
            if npeaks == Nsources:
                break;
    
    # if Nsources not found
#    if npeaks != Nsources:
#        pks[npeaks+1 : Nsources] = [];
#        locs[npeaks+1 : Nsources] = [];
        
    return pks, locs