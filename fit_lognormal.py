#!/usr/bin/env python
# coding: utf-8


#functions to fit a lognormal distribution to a series of event times in minutes
#based on the code from https://arxiv.org/pdf/1801.10082.pdf, with modifications

#fitted function follows the form of the pdf given here: https://en.wikipedia.org/wiki/Log-normal_distribution

#parameters:
#   mu      mean
#   sigma   variance


import numpy as np
from scipy.optimize import minimize
from scipy.special import erf


#given a list of event times, fit a lognormal function, returning parameters mu and sigma
#use random perturbation to correct for poor initial guesses a maximum of <runs> times
#if no good fit found, return None indicating failure
def lognorm_parameters_estimation(event_times, runs = 10, large_params = [20, 20], start_params = [4.,2.]):

    #given lognormal parameters mu and sigma, return the value of the negative 
    #loglikelihood function across all time values 
    def lognorm_loglikelihood(var):     #var = (mu, sigma)
        #extract parameters from tuple
        mu = var[0]
        sigma = var[1]

        t_n = event_times[-1]   #grab last event tim

        f = (-1/2 - (1/2) * erf((np.log(t_n)-mu) / (np.sqrt(2)*sigma)) ) + len(event_times) * np.log(1 / (sigma * np.sqrt(2 * np.pi)))
        for t in event_times:
            f += -(np.log(t) - mu)**2 / (2 * sigma**2) - np.log(t)
        return (-1) * f       #convert from maximum likelihood to negative likelihood (for minimization)
    #end lognorm_loglikelihood
    
    param_set = np.asarray(start_params)        #convert start params to np array

    for i in range(runs):
        result = minimize(lognorm_loglikelihood, param_set, method = 'L-BFGS-B', bounds = ((0.0001,None), (0.0001,None)))
        fit_params = list(result.get('x'))

        #if bad fit, perturb the initial guess and re-fit
        if fit_params[0] > large_params[0] or fit_params[1] > large_params[1]:
            param_set += np.array([np.random.normal(0, start_params[0]/10), np.random.normal(0, start_params[1]/10)])

            #if too many bad results, return fail
            if i == runs-1:
                fit_params = None
        #good fit, quit
        else:
            break

    return fit_params    #[mu, sigma]
#end lognorm_parameters_estimation


#given lognormal parameters, return the value of the lognormal pdf at time t
#both parameters must be > 0
def lognorm_func(t, mu, sigma):
    if t > 0:
        res = 1 / (sigma * t * np.sqrt(2 * np.pi)) * np.exp(-((np.log(t) - mu)**2) / (2 * sigma**2))
    else:
        res = 0
    return res
#end lognorm_func


#given event times, fit the lognormal distribution
#if fit fails, return None
#otherwise, returns mu and lambda paramters
def fit_lognormal(event_times, display = True):
    params = lognorm_parameters_estimation(event_times)     #try loglikelihood estimation first

    if display:
        if params == None:
            print("\nlognormal fit failed\n")
        else:
            print("\nLog-normal params:", "\n   mu\t\t", params[0], "\n   sigma\t", params[1], "\n")

    return params   #(mu, sigma)
#end fit_lognormal


#Usage example
'''
from fit_lognormal import *
params = fit_lognormal(comment_times)
mu, sigma = fit_lognormal(comment_times)
'''







