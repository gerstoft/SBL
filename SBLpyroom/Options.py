#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Options:
    def __init__(self, convergence_error, gamma_range, convergence_maxiter, convergence_min_iteration, status_report,  fixedpoint, Nsource, flag):
#(self, convergence_error, convergence_delay, convergence_maxiter, convergence_min_iteration, status_report, noisepower_guess, fixedpoint, Nsource, flag, tic):
        self.convergence_error = convergence_error; 
#        self.convergence_delay = convergence_delay; 
        self.gamma_range = gamma_range; 
        self.convergence_maxiter = convergence_maxiter; 
        self.convergence_min_iteration = convergence_min_iteration; 
        self.status_report = status_report;
#        self.noisepower_guess = noisepower_guess;
        self.fixedpoint = fixedpoint;
        self.Nsource = Nsource;
        self.flag = flag;
#        self.tic = tic;

class Report:
    def __init__(self, results_error, results_iteration_L1, results_iteration, results_noisepower, results_gamma, options):
        self.results_error = results_error;
        self.results_iteration_L1 = results_iteration_L1;
        self.results_iteration = results_iteration;
        self.results_noisepower = results_noisepower;
        self.results_gamma = results_gamma;
#        self.results_x_post = results_x_post;
        self.options = options;              
        


# In[ ]:




