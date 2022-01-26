import numpy as np
import matplotlib.pyplot as plt
import cmath
#import SBL
#import SBL4
import time
from sbl import SBL, Report, Options,  SBLpeaks_1D
#from Options import Options
#from Options import Report
# Parameter
c      = 1500;       # speed of sound
f      = [200, 250, 300, 350, 400];  #frequency
#f      = [200];  #frequency
lambda_vec = np.zeros(len(f));
for iF in range(len(f)):
    lambda_vec[iF] = c/f[iF];

# ULA-horizontal array configuration
Nsensor = 20;                  # number of sensors
d       = 1/2*lambda_vec[0];       # intersensor spacing for 200 Hz
q       = np.arange(Nsensor);
xq      = (q-(Nsensor-1)/2)*d; #sensor locations
xq = xq.reshape(xq.shape[0], 1)
# sensor configuration structure
Nlambda = len(lambda_vec);

# SBL options structure for version 3 !!!
#options = SBLSet();

# signal generation parameters
SNR = 40 ;-10;
# total number of snapshots
Nsnapshot = 1;
# number of random repetitions to calculate the average performance
Nsim = 1;
np.random.seed(23)
# range of angle space
thetalim         = [-90, 90];
theta_separation = 0.1;

# Bearing grid
theta = np.arange(thetalim[0], thetalim[1]+theta_separation, theta_separation)
Ntheta = len(theta)
theta = theta[:, np.newaxis]

# Design/steering matrix
sin_theta = np.sin(theta/180 * np.pi);

# Multi-F dictionary / replicas
A = np.zeros(( Nsensor, Ntheta,Nlambda), dtype = complex);


for iF in range(Nlambda):
    A[:,:,iF] = np.exp(-1j*2*np.pi/lambda_vec[iF]* (xq @ sin_theta.T))/np.sqrt(Nsensor);

def generate_signal(Nsensor, lambda_vec, xq, Nsnapshot, SNR, isim):
    # three DOA example
    source_theta = np.array([-2, 3, 75]) * np.pi / 180; source_amp = np.array([0.5, 13, 1]);
     # One DOA example
   # source_theta = np.array([-20]) * np.pi / 180;   source_amp = np.array([1]);
    source_theta = source_theta.reshape(source_theta.shape[0], 1)
    source_amp   = source_amp.reshape(source_amp.shape[0], 1)
    Nsource = len(source_theta);
    Nlambda = len(lambda_vec);
    
    Ysignal = np.zeros((Nsensor, Nsnapshot, Nlambda), dtype=complex)
    # random number seed

    for iF in range(Nlambda):
        for t in range(Nsnapshot):
            phase = np.random.randn(Nsource,1);
            Xsource = source_amp * np.exp(1j * 2 * np.pi * phase);
       #     Xsource =abs(Xsource)
            # Represenation matrix (steering matrix)
            u = np.sin(source_theta);
            A = np.exp(-1j * 2 * np.pi / lambda_vec[iF] * xq @ u.T) / np.sqrt(Nsensor);
            # Signal without noise
            Ysignal[:, t, iF] = np.sum(A @ Xsource, axis=1);
            # add noise to the signals
            rnl = 10 ** (-SNR / 20) * np.linalg.norm(Xsource);
            e = (np.random.randn(Nsensor, 1) + 1j*np.random.randn(Nsensor, 1) ) / np.sqrt(2 * Nsensor) * rnl;
            if (SNR==100):
                Ysignal[:, t, iF] = Ysignal[:, t, iF];
            else:
                Ysignal[:, t, iF] = Ysignal[:, t, iF] + e.reshape(e.shape[0]);
            
    return Ysignal, source_theta

t = time.time()
for isim in range(Nsim):
    Ysignal, source_theta = generate_signal(Nsensor, lambda_vec, xq, Nsnapshot, SNR, isim);

    # run CBF
    print('Running CBF code');
    CBF = np.zeros((Ntheta,Nlambda), dtype=float); 
    SCM = np.zeros((Nsensor, Nsensor,Nlambda), dtype=complex);

    for iF in range(Nlambda):
        SCM[ :, :,iF] = np.matmul(Ysignal[ :, :,iF], np.conj(Ysignal[ :, :, iF].T))/Nsnapshot;    
        CBF[:, iF] = np.real(np.diag(np.conj(np.transpose(A[ :, :,iF])) @ SCM[ :, :,iF] @ A[ :, :,iF]));

   # run SBL
   #(self, convergence_error, gamma_range, convergence_maxiter, convergence_min_iteration, status_report,  fixedpoint, Nsource, flag):
     #   options = Options(10 ** (-6), 200, 3000, 25, 150, 0.1, 1, 3, 0, 1);
        options = Options(10 ** (-8), 10 ** (-4), 3000, 2, 5, 1, 3, 1);# wors well for noise free
        options = Options(10 ** (-4), 10 ** (-4), 3000, 1, 1, 1, 3, 1);
    print('Running SBL code');
    gamma, report = SBL(A, Ysignal, options);

elapsed = time.time() - t; print ('Elapsed time',elapsed)

maxc = np.zeros((1, Nlambda));
for iF in range(Nlambda):
    CBF_dB = 10 * np.log10(np.abs(np.reshape(CBF[ :,iF], Ntheta)));
    plt.plot(theta, CBF_dB, linewidth=2);
    maxc[:, iF] = np.max(CBF_dB);

plt.plot(source_theta[0:] * 180 / np.pi, -5 * np.ones(len(source_theta)), 'r+', markersize=20);
plt.title('Single Freq. CBF');
plt.xlabel('Theta [deg.]');
plt.ylabel('Power [dB]');
plt.xlim([thetalim[0], thetalim[1]]);
plt.ylim([-40, np.ceil(np.max(maxc))]);
plt.show()

plt.plot(source_theta[0:]*180/np.pi, -0.5 * np.ones(len(source_theta)), 'r+', markersize = 20);
CBF_dB_sum = 10 * np.log10(np.sum(np.abs(CBF), axis = 1)/Nlambda);
plt.plot(theta, CBF_dB_sum, linewidth = 2);
plt.title('Incoherent Multi-Freq. CBF');
plt.xlabel('Theta [deg.]');
plt.ylabel('Power [dB]');
plt.xlim([thetalim[0], thetalim[1]]);
plt.ylim([-40, np.ceil(np.max(maxc))]);
plt.show()

plt.plot(source_theta[0:]*180/np.pi, -0.5 * np.ones(len(source_theta)), 'r+', markersize = 20);
plt.plot(theta, 10 * np.log10(gamma+0.000001), linewidth = 2);
maxg = np.max(10 * np.log10(gamma+0.000001));
plt.title('SBL');
plt.xlabel('Theta [deg.]');
plt.ylabel('Power [dB]');
plt.xlim([thetalim[0], thetalim[1]]);
plt.ylim([-65, np.ceil(np.max(maxg))]);
plt.show();


