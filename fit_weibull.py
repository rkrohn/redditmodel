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
import warnings
import random
warnings.filterwarnings("ignore")   #hide the warnings that optimize sometimes throws


#HARDCODED PARAMS - only used when fit/estimation fails
#not used literally, small random perturbation applied

DEFAULT_WEIBULL_NONE = [1, 1, 0.15]     #param results if post has NO comments to fit
                                        #force a distribution heavily weighted towards the left, then decreasing

DEFAULT_WEIBULL_SINGLE = [1, 2, 0.75]   #param result if post has ONE comment and other fit methods fail
                                        #force a distribution heavily weighted towards the left, then decreasing
                                        #use this same hardcode for other fit failures, but set a equal to the number of replies

DEFAULT_QUALITY = 0.45                   #default quality if any hardcode param is used

DEFAULT_DELTA = 0.15            #default maximum delta percentage for random hardcoded param perturbations


#given a list of event times, fit a weibull function, returning parameters a, lbd, and k
#use random perturbation to correct for poor initial guesses a maximum of <runs> times
#if no good fit found, return None indicating failure
def weibull_parameters_estimation(event_times, runs = 20, large_params = [1000, 10000, 20], start_params = [20, 500, 2.3], max_iter = False, display = False):

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

    #update large_params limit for <a> parameter based on number of events
    large_params[0] = 1.45 * len(event_times)
           
    param_set = np.asarray(start_params)    #convert start params to np array

    for i in range(runs):
        #run the minimize call, limiting iterations if given
        if max_iter == False:
            result = minimize(weib_loglikelihood, param_set, method = 'L-BFGS-B', bounds = ((0.0001, None), (0.0001, None),(0.0001, None)))
        else:
            result = minimize(weib_loglikelihood, param_set, method = 'L-BFGS-B', bounds = ((0.0001, None), (0.0001, None),(0.0001, None)), options = {'maxiter': max_iter})
        fit_params = list(result.get('x'))

        #if bad fit (or failed fit, that returned initial params), perturb the initial guess and re-fit
        #if you get the initial params back, fit failed
        if (fit_params[0] > large_params[0] or fit_params[1] > large_params[1] or fit_params[2] > large_params[2]) or (fit_params[0] == param_set[0] and fit_params[1] == param_set[1] and fit_params[2] == param_set[2]):
            param_set += np.array([np.random.normal(0, start_params[0]/10), np.random.normal(0, start_params[1]/10),
                                  np.random.normal(0, start_params[2]/10)])

            #if too many bad results, return fail
            if i == runs-1:
                fit_params = None
        #good fit, quit
        else:
            if display:
                print("Converged in", result.get('nit'), "iterations")
            break

    return fit_params     #[a, lbd, k]
#end weibull_parameters_estimation


#perform a curve_fit of the weibull function (less precise than parameter estimation method)
#unmodified from original, exept for comments
def func_fit_weibull(event_times, res=60, runs = 20, T_max = 3*1440, large_params = [1000, 10000, 20], start_params = [50, 400, 2.3], display=False):
    
    bins = np.arange(0, max([T_max, max(event_times)]), res)
    hist, bins = np.histogram(event_times, bins)  #construct histogram of the root comments appearance 
    center_bins = [b+res/2 for b in bins[:-1]]
    
    param_set = np.asarray(start_params)
    for i in range(runs):
        fail = False        #boolean failure flag

        try:
            fit_params, pcov = curve_fit(weib_func, xdata = center_bins, ydata = hist/res, p0 = param_set, bounds = (0.00001, 1000000), maxfev = 1000000)
        except Exception as e:
            #catch the ValueError that sometimes occurs when fitting few events
            #really shouldn't happen, but just in case...
            if display:
                print("Error encountered in func_fit_weibull for", len(event_times), "events:", e)
            fail = True     #set flag to trigger a perturbation
            
        #if bad fit, perturb the initial guess and re-fit
        if fail or fit_params[0] > large_params[0] or fit_params[1] > large_params[1] or fit_params[2] > large_params[2]:
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
    if t == 0 and k < 1:
        return float('Inf')
    return (a* k/lbd) * (t/lbd)**(k-1) * np.exp(-(t/lbd)**k)
#end weib_fuct


#given a list (of params), and a list of corresponding delta limits,
#apply a random perturbation to each value with max delta of value * DEFAULT_DELTA
def perturb(data):
    for i in range(len(data)):
        max_delta = data[i] * DEFAULT_DELTA
        data[i] += random.uniform(-1 * max_delta, max_delta)
    return data
#end perturb


#given event times for a partial cascade, fit the weibull function
#slightly different in functionality than standard fit_weibull
#if both fit methods fail, or there are not enough events, return False as signal to use inferred params
#if fit succeeds, returns a, lambda, and k parameters (no quality)
def fit_partial_weibull(event_times, param_guess = False, max_iter = False, display = False):

    #no events to fit, return False
    if len(event_times) == 0:
        if display:
            print("No events to fit Weibull, returning")
        return False

    params = weibull_parameters_estimation(event_times, start_params=param_guess, max_iter=max_iter, display=display)     #try loglikelihood estimation first

    #if that fails, use curve fit if more than one item
    if params == None:
        if len(event_times) != 1:
            params = func_fit_weibull(event_times, start_params=param_guess)
        else:
            params = [None, None, None]     #next if will catch this

    #if both fail, return False - will use inferred params
    if params[0] == None: 
        if display:
            print("Weibull fit failed, returning")
        return False

    if display:
        print("Weibull params: \n   a\t\t", params[0], "\n   lambda\t", params[1], "\n   k\t\t", params[2], "\n")

    return params   #(a, lambda, k)
#end fit_partial_weibull


#given event times, fit the weibull function
#if both fit methods fail or no events, use either perturbed hardcoded defaults or return all False
#otherwise, returns a, lambda, and k parameters, and quality measure
def fit_weibull(event_times, display = False, hardcode_default = True):

    #no events to fit
    if len(event_times) == 0:
        #hardcode
        if hardcode_default:
            params = perturb(list(DEFAULT_WEIBULL_NONE))    #small random perturbation of hardcoded vals
            if display:
                print("No events to fit, setting Weibull params: (quality", str(DEFAULT_QUALITY) + ")\n   a\t\t", params[0], "\n   lambda\t", params[1], "\n   k\t\t", params[2], "\n")
            return params + [DEFAULT_QUALITY]
        #no hardcode
        else:
            return [False, False, False, False]

    params = weibull_parameters_estimation(event_times)     #try loglikelihood estimation first
    quality = 0.95

    #if that fails, use curve fit if more than one item
    if params == None:
        if len(event_times) != 1:
            params = func_fit_weibull(event_times)
            quality = 0.8
        else:
            params = [None, None, None]     #next if will catch this

    #if both fail, hardcode or do nothing - not sure what else to do
    if params[0] == None: 
        if hardcode_default:
            params = list(DEFAULT_WEIBULL_SINGLE)
            if len(event_times) > 1:
                params[0] = len(event_times)    
            params = perturb(params)        #small random perturbation
            if display:
                print("Fit failed, setting Weibull params: (quality", str(DEFAULT_QUALITY) + ")", "\n   a\t\t", params[0], "\n   lambda\t", params[1], "\n   k\t\t", params[2], "\n")
            return params + [DEFAULT_QUALITY]
        else:
            return [False, False, False, False]

    if display:
        if params[0] == None:
            print("Weibull fit failed\n")
        else:
            print("Weibull params: (quality", str(quality) + ")\n   a\t\t", params[0], "\n   lambda\t", params[1], "\n   k\t\t", params[2], "\n")

    return params + [quality]   #(a, lambda, k, quality)
#end fit_weibull


#Usage example
'''
from fit_weibull import *
params = fit_weibull(comment_times)
a, lbd, k = fit_weibull(comment_times)     
'''   

