#!/usr/bin/env python
# coding: utf-8


#simulate a reddit cascade
#root comments based on a Weibull distribution
#deeper comments follow a log-normal distribution
#see fit_weibull.py and fit_lognormal.py for more details on fitting process

#based on the code from https://arxiv.org/pdf/1801.10082.pdf, with modifications


'''
import json
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import OrderedDict
import gc
from itertools import islice
import time
from scipy.optimize import curve_fit, minimize
from scipy.special import erf, gamma
from IPython.display import clear_output
import warnings
from operator import itemgetter
import pickle
import multiprocessing as mp
from copy import deepcopy
import os
from collections import defaultdict
import math
import warnings
warnings.filterwarnings("ignore")
'''

from fit_weibull import *
from fit_lognormal import *
import math


#HARDCODED VALUES
INF_INTENSITY = 2.25    #value used for intensity if function returns infinity (occurs when t = 0 and k < 1)


#given current time t, max inter-event arrival time T, max number of events N_max, 
#and acceptance flag (True if t was accepted as an event time), and list of generated
#event times, determine if simulation has reached stopping condition
#returns True if simulation should stop, False otherwise
def stop_sim(t, T, N_max, accept, new_event_times, display = False):
    #quit if generated too many events
    if len(new_event_times) > N_max:
        if display:
            print("Stopping simulation: max total events reached")
        return True

    #quit if reached inter-arrival threshold
    #three conditions cause this to trigger: 
    #   no events and current time too high
    #   t was added as an event and occurred >T units after the previous event
    #   t was not accepted, and is >T units after the previous event
    if (len(new_event_times) == 0 and t > T) or (accept and len(new_event_times) > 1 and t - new_event_times[-2] > T) or (accept == False and len(new_event_times) > 0 and t - new_event_times[-1] > T):
        if display:
            print("Stopping simulation: max inter-event arrival time reached")
        return True

    #if no conditions tripped, not time to stop, return False
    return False
#end stop_sim


#evaluate weibull at time t, return result - with special check for infinite values
#all paramaters must be > 0
def weib_intensity(t, a, lbd, k):
    f = weib_func(t, a, lbd, k)     #get value of weibull function at time t
    #special-case to prevent infinte returns
    if math.isinf(f):
        f = INF_INTENSITY
    return f


#given weibull parameters (see fit_weibull.py for param details), simulate new event times following this distribution
#
#parameters:
#   a           scalar multiplier (directly from the paper)
#   lambda      scale parameter (b in the source paper)
#   k           shape parameter (alpha in the source paper)
#
#essentially a homogenous poisson process, but with a variable base intensity mu defined by the weibull kernel
#bounded by inter-event time of 7200 minutes = 120 hours or max number of events 2000 
def generate_weibull_times(params, start_time = 0, T = 7200, N_max = 2000):   #params = [a, lbd, k]
    (a, lbd, k) = params    #unpack parameters

    new_event_times = []    #build list of new event times
    t = start_time          #init t

    #set base intensity for current time
    #intensity = lambda* = thinning rate = upper bound of mu(t) for current time interval = rate at which to generate top-level replies

    #shape param > 1, bell-shape curve
    if k > 1:
        if t > lbd * ((k-1) / k)**(1/k):
            intensity = weib_intensity(t, a, lbd, k)
        else:
            intensity = (a * k/lbd) * ((k-1) / k)**(1 - 1/k) * np.exp(-((k-1) / k))
    #shape param between 0 and 1, decreasing curve
    elif k > 0:
        intensity = weib_intensity(t,a,lbd,k)		#call method to get weibull at time t


    #thinning algorithm to generate times - see page 10 of Hawkes chapter for more details
    while True:
        accept = False      #boolean flag used in stoppig condition logic

        e = np.random.uniform(low=0, high=1)		#sample between 0 and 1 uniform distribution
        t += -np.log(e)/intensity 					#compute waiting time = -ln(e)/lambda, add to current time t

        #accept or reject t as event time?
        s = np.random.uniform(low=0, high=1)		#draw another value
        #if random value less than ratio of true event rate to thinning rate, accept
        if s < weib_intensity(t,a,lbd, k) / intensity:
            new_event_times.append(t)		#save this event time
            accept = True

            #update thinning rate if necessary for next event generation
            if (k > 1 and t > lbd*((k-1)/k)**(1/k)) or (k > 0 and k < 1):
                intensity = weib_intensity(t,a,lbd, k)

        #quit if reached inter-arrival threshold
        if stop_sim(t, T, N_max, accept, new_event_times):
            break

    return new_event_times
#end generate_weibull_times


#evaluate log-normal distribution at time t, including branching factor scaling to get intensity
#requires fitted params mu, sigma, and n_b
def lognorm_intensity(t, mu, sigma, n_b):
    return n_b * lognorm_func(t, mu, sigma)
#end lognorm_intensity


#given log-normal parameters (see fit_lognormal.py for param details), simulate new event times following this distribution
#
#parameters:
#   mu      mean
#   sigma   variance
#
#essentially a homogenous poisson process, but with a variable base intensity mu defined by the log-normal distribution
#bounded by inter-event time of 7200 minutes = 120 hours or max number of events 200
#also utilizes the estimated branching factor 
def generate_lognorm_times(params, n_b, start_time = 0, T = 7200, N_max = 200): # Log-normal kernel
    (mu, sigma) = params    #unpack the parameters

    new_event_times = []    #build list of new event times
    t = start_time          #init time

    #set base intensity for current time
    #get intensity = lambda* = thinning rate = event rate = upper bound of intensity
    if t > np.exp(mu - (sigma**2)):  
        intensity = lognorm_intensity(t, mu, sigma, n_b)
    else:
        intensity = n_b * (np.exp(sigma**2 - mu) / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(sigma**2) / 2)

    #thinning process to generate events - see page 10 of Hawkes chapter for more details
    while True:
        accept = False          #boolean flag used in stopping logic

        e = np.random.uniform(low=0, high=1)	#sample from uniform distribution
        t += -np.log(e)/intensity 				#compute inter-event time and add to current time t

        #accept or reject current time as event time?
        s = np.random.uniform(low=0, high=1)	#sample again
        #if sample less than ratio of true event rate to thinning rate, accept
        if s < lognorm_intensity(t, mu, sigma, n_b) / intensity:
            new_event_times.append(t)		#accept this event time
            accept = True
            #update lambda* for next generation cycle, if required
            if t > np.exp(mu - (sigma**2)):
                intensity = lognorm_intensity(t, mu, sigma, n_b)

        #quit if reached inter-arrival threshold
        if stop_sim(t, T, N_max, accept, new_event_times):
            break

    return new_event_times
#end generate_lognorm_times


#overall simulation function, work from root down, generating as we go
#generate root comments based on fitted weibull, comment replies based on fitted log-normal
#requires fitted weibull and log-normal distribution to define hawkes process
#model params = [a, lbd, k, mu, sigma, n_b] (first 3 weibull, next 2 lognorm, last branching factor)
def simulate_comment_tree(model_params, display = True):
    weibull_params, lognorm_params, n_b = unpack_params(model_params)   #unfold parameters

    #simulate from empty starting tree (just the root, no observed comments
    #get root comment times
    root_comment_times = generate_weibull_times(weibull_params)
    if display:
        print("new root comments:", root_comment_times, "\n")

    #generate deeper comments for each root comment (and each of their comments, etc)
    needs_replies = [] + root_comment_times
    all_replies = [] + root_comment_times
    while len(needs_replies) != 0:
        comment = needs_replies.pop()
        reply_times = generate_lognorm_times(lognorm_params, n_b, start_time = comment)
        all_replies.extend(reply_times)
        needs_replies.extend(reply_times)
        print("   ", "*" if comment in root_comment_times else "", comment, ":", reply_times)
    #need to run this deeper, but good enough for today

    if display:
        print("Simulated cascade has", len(root_comment_times), "replies and", len(all_replies), "total comments")

    return sorted(all_replies)  #for now, just return sorted list of all reply times
#end simulate_comment_tree


#helper function to unpack parameters as given in tensor to separate items
def unpack_params(params):
    weibull_params = params[:3]
    lognorm_params = params[3:5]
    n_b = params[5]

    return weibull_params, lognorm_params, n_b
#end unpack_params