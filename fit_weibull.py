#!/usr/bin/env python
# coding: utf-8


#functions to fit a Weibull distribution to a series of event times in minutes
#based on the code from https://arxiv.org/pdf/1801.10082.pdf, with modifications

#fitted function follows the form given here: https://en.wikipedia.org/wiki/Weibull_distribution,
#with the addition of a scalar multiplier a as in the paper

#parameters:
#   k           shape parameter (alpha in the source paper)
#   lambda      scale parameter (b in the source paper)
#   a           scalar multiplier (directly from the paper)


import numpy as np
from scipy.optimize import curve_fit, minimize


#given a list of event times, fit a weibull function, returning parameters a, lbd, and k
#use random perturbation to correct for poor initial guesses a maximum of <runs> times
#if no good fit found, return None indicating failure
def weibull_parameters_estimation(event_times, runs = 10, large_params = [1000, 10000, 20], start_params = [20, 500, 2.3]):

    #given weibull parameters a, k, and lambda (lbd), return the value of the negative 
    #loglikelihood function across all time values 
    def weib_loglikelihood(var):    #var = (a, lbd, k)
        #extract params from tuple
        a = var[0]
        lbd = var[1]
        k = var[2]

        t_n = event_times[-1]       #grab last event time

        f = -a * (1 - np.exp(-(t_n/lbd)**(k))) + len(event_times) * (np.log(a) + np.log(k) - (k)*np.log(lbd))
        for t in event_times:
            f += (k-1) * np.log(t) - (t/lbd)**(k)
        return (-1) * f     #convert from maximum likelihood to negative likelihood (for minimization)
    #end weib_loglikelihood
           
    param_set = np.asarray(start_params)    #convert start params to np array

    for i in range(runs):
        result = minimize(weib_loglikelihood, param_set, method = 'L-BFGS-B', bounds = ((0.0001, None), (0.0001, None),(0.0001, None)))
        fit_params = list(result.get('x'))

        #if bad fit, perturb the initial guess and re-fit
        if fit_params[0] > large_params[0] or fit_params[1] > large_params[1] or fit_params[2] > large_params[2]:
            param_set += np.array([np.random.normal(0, start_params[0]/10), np.random.normal(0, start_params[1]/10),
                                  np.random.normal(0, start_params[2]/10)])

            #if too many bad results, return fail
            if i == runs-1:
                fit_params = None
        #good fit, quit
        else:
            break

    return fit_params     #[a, lbd, k]
#end weibull_parameters_estimation


#perform a curve_fit of the weibull function (less precise than parameter estimation method)
#unmodified from original, exept for comments
def func_fit_weibull(event_times, res=60, runs = 10, T_max = 3*1440, large_params = [1000, 10000, 20], start_params = [50, 400, 2.3]):
    
    bins = np.arange(0, max([T_max, max(event_times)]), res)
    hist, bins = np.histogram(event_times, bins)  #construct histogram of the root comments appearance 
    center_bins = [b+res/2 for b in bins[:-1]]
    
    param_set = np.asarray(start_params)
    for i in range(runs):
        fit_params, pcov = curve_fit(weib_func, xdata = center_bins, ydata = hist/res, p0 = param_set, 
                                     bounds = (0.0001, 100000))
        #if bad fit, perturb the initial guess and re-fit
        if fit_params[0] > large_params[0] or fit_params[1] > large_params[1] or fit_params[2] > large_params[2]:
            param_set += np.array([np.random.normal(0, start_params[0]/10), np.random.normal(0, start_params[1]/10), np.random.normal(0, start_params[2]/4)])
            #if too many bad results, return fail
            if i == runs-1:
                fit_params = [None, None, None]
        #good fit, quit
        else:
            break

    return fit_params     # [a, lbd, k]
#end func_fit_weibull


#given Weibull parameters, return the value of the Weibull pdf at time t
#all parameters must be > 0
def weib_func(t, a, lbd, k):
		return (a* k/lbd) * (t/lbd)**(k-1) * np.exp(-(t/lbd)**k)
#end weib_fuct


#given event times, fit the weibull function
#if both methods fail, returns None for all parameters
#otherwise, returns a, lambda, and k paramters
def fit_weibull(event_times, display = True):
    params = weibull_parameters_estimation(event_times)     #try loglikelihood estimation first
    #if that fails, use curve fit
    if params == None:
        params = func_fit_weibull(event_times)

    if display:
        if params[0] == None:
            print("Weibull fit failed")
        else:
            print("Weibull params:", "\n   a\t\t", params[0], "\n   lambda\t", params[1], "\n   k\t\t", params[2], "\n")

    return params   #(a, lambda, k)
#end fit_weibull


#Usage example
'''
from fit_weibull import *
params = fit_weibull(comment_times)
a, lbd, k = fit_weibull(comment_times)     
'''   

