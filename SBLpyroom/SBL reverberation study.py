import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import scipy.io as scio
import IPython
import pyroomacoustics as pra
from SBL4 import *
from Options import *
import time

dataFile = 'Y_test_beta.mat'
data = scio.loadmat(dataFile)
Y = data['Y']
Y = Y.T;
dataFile2 = 'echo.mat'
data2 = scio.loadmat(dataFile2)
echo = data2['r_temp'];
echo = echo.T;
echo = echo[0:2,:]
j = cmath.sqrt(-1);
N = 8;
r = 37.5e-3;
# Location of sources
azimuth = np.array([45]) / 180. * np.pi
azimuth_deg = azimuth * 180/ np.pi;
#elevation = np.array([90]) / 180. * np.pi
distance = 1.0  # meters

#amplitude of the sources
source_amp = np.array([1])

c = 343.    # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]

theta = 90 / 180 * np.pi;
phi = np.linspace(-90, 90, 361) /180 * np.pi;

Nphi = len(phi);

n = np.linspace(0, N-1, N);
n = n.reshape((N, 1));
n_grid = 360;

snr_db = np.array([15]);
beta = np.linspace(0.2, 1.5, 14);
phi_MUSIC = np.ones(len(beta));
phi_SBL = np.ones(len(beta));
phi_CBF = np.ones(len(beta));
phi_SRP = np.ones(len(beta));
RMSE_MUSIC = np.ones(len(beta));
RMSE_SBL = np.ones(len(beta));
RMSE_CBF = np.ones(len(beta));
RMSE_SRP = np.ones(len(beta));

t = time.time()    
for beta_index in range(len(beta)):    
    snr = snr_db[0];
    X = pra.transform.stft.analysis(Y[:, :, beta_index].T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])
    X_noise = np.random.randn(X.shape[0], X.shape[1], X.shape[2]);
    sigma = 10 ** (-snr/10);
    X_temp = X + sigma * X_noise
    spatial_resp = dict()
    algo_name = 'MUSIC'
    doa = pra.doa.algorithms[algo_name](echo, fs, nfft = nfft, c=c, num_src=len(azimuth), n_grid = n_grid, dim = 2, max_four=4)
    #doa.locate_sources(X_temp, freq_range=freq_range)
    doa.locate_sources(X_temp, freq_range=freq_range)
    phi_MUSIC[beta_index] = doa.azimuth_recon * 180/np.pi;
    RMSE_MUSIC[beta_index] = np.abs(phi_MUSIC[beta_index] - azimuth_deg);
elapsed = time.time() - t; print ('Music Elapsed time',elapsed)    

t = time.time()    

for beta_index in range(len(beta)):    
    snr = snr_db[0];
    X = pra.transform.stft.analysis(Y[:, :, beta_index].T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])
    X_noise = np.random.randn(X.shape[0], X.shape[1], X.shape[2]);
    sigma = 10 ** (-snr/10);
    X_temp = X + sigma * X_noise
    spatial_resp = dict()
    algo_name = 'SRP'
    doa = pra.doa.algorithms[algo_name](echo, fs, nfft = nfft, c=c, num_src=len(azimuth), n_grid = n_grid, dim = 2, max_four=4)
    #doa.locate_sources(X_temp, freq_range=freq_range)
    doa.locate_sources(X_temp, freq_range=freq_range)
    phi_SRP[beta_index] = doa.azimuth_recon * 180/np.pi;
    RMSE_SRP[beta_index] = np.abs(phi_SRP[beta_index] - azimuth_deg);
elapsed = time.time() - t; print ('SRP Elapsed time',elapsed)    

t = time.time()    
for beta_index in range(len(beta)):
    snr = snr_db[0];
    X = pra.transform.stft.analysis(Y[:, :, beta_index].T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])
    X_noise = np.random.randn(X.shape[0], X.shape[1], X.shape[2]);
    sigma = 10 ** (-snr/10);
    X_temp2 = X + sigma * X_noise;
    # freq_range = [int(np.round(f / fs * nfft))
    #                           for f in freq_range]
    # freq_bins_index = np.arange(freq_range[0], freq_range[1], dtype=np.int);
    # freq_bins = freq_bins_index * fs / nfft;
    # Nfreq = len(freq_bins)
    X_signal = np.transpose(X_temp2, axes=[2, 1, 0]);
    X_signal = X_signal[..., :, :].transpose([2, 0, 1]); # dim: #freq_bin * Nsensor * Nsnapshot
    mod_vec = np.transpose(np.array(doa.mode_vec[:,:,:]),axes=[1, 2, 0]);
#    options = Options(800*10 ** (-2), 10 ** (-1), 500, 0, 20, 1, len(azimuth), 1);
    options = Options(10 ** (-0), 10 ** (-2), 3000, 0, 2, 1, len(azimuth), 1);
    #    options = Options(10 ** (-5), 10 ** (-5), 3000, 2, 1000, 1, len(azimuth), 1);   #options = Options(10 ** (-6), 200, 3000, 25, 150, 0.1, 1, len(azimuth), 0, 1);
    gamma, report = SBL(mod_vec, X_signal, options);
    gamma_list = gamma.tolist();
    gamma_list_max = max(gamma_list[1:]);
    max_id = gamma_list.index(gamma_list_max);
    phi_2 = np.linspace(0, 2*np.pi, n_grid, endpoint=False);
    phi_SBL[beta_index] = phi_2[max_id] / np.pi * 180;
    RMSE_SBL[beta_index] = np.abs(phi_SBL[beta_index] - azimuth_deg);
    Nfreq = mod_vec.shape[2]
    SCM = np.zeros(( N, N, Nfreq), dtype = complex);
    CBF = np.zeros(( n_grid,  Nfreq));
    Nsnapshot = X_signal.shape[1]
    for iF in range(Nfreq):
        SCM[ :, :,iF] = np.matmul(X_signal[ :, :,iF], np.conj(X_signal[ :, :, iF].T))/Nsnapshot;   
        # noise power initializtion
        CBF[:, iF] = np.real(np.diag(np.conj(np.transpose(mod_vec[ :, :,iF])) @ SCM[ :, :,iF] @ mod_vec[ :, :,iF])); 
    gamma= np.sum(CBF,axis=1)
    gamma_list = gamma.tolist();
    gamma_list_max = max(gamma_list[1:]);
    max_id = gamma_list.index(gamma_list_max);
    phi_2 = np.linspace(0, 2*np.pi, n_grid, endpoint=False);
    phi_CBF[beta_index] = phi_2[max_id] / np.pi * 180;
    RMSE_CBF[beta_index] = np.abs(phi_CBF[beta_index] - azimuth_deg);
        
        
elapsed = time.time() - t; print ('SBL Elapsed time',elapsed)    
plt.plot(beta, RMSE_MUSIC, label = 'MUSIC');
plt.plot(beta, RMSE_SBL, label = 'SBL');
plt.plot(beta, RMSE_SRP, label = 'SRP');
plt.plot(beta, RMSE_CBF, label = 'CBF');
plt.legend();
plt.xlabel('RT_60/s');
plt.ylabel('RMSE');
plt.show();

plt.plot(phi_2/ np.pi * 180, gamma, label = 'CBF');
plt.show();