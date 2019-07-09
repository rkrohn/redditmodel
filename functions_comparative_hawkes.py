#functions for comparative_hawkes.py
#unmodified, apart from removal of some lingering jupyter tags

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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


# ## Inference of parameters of a process

# Estimate parameters of $\mu(t)$ and $\phi(t)$ given the time stamps. Initial guess for parameters may be bad for convergence (one of the parameters is larger than in *large_parameters*), thus a random perturbation is introduced (maximum *runs* times).

#fits weibull for top-level comments, returns params a, b, alpha
def mu_parameters_estimation(hawkes_root, runs = 10, large_params = [1000, 10000, 20], start_params = [20, 500, 2.3]):
    
    def weib_loglikelihood(var):  # var = (a,b,alpha)
        t_n = hawkes_root[-1]
        f = -var[0]*(1-np.exp(-(t_n/var[1])**(var[2]))) + len(hawkes_root)*(np.log(var[0])+np.log(var[2])-(var[2])*np.log(var[1]))
        for t in hawkes_root:
            f+= (var[2]-1)*np.log(t)-(t/var[1])**(var[2])
        return (-1)*f
        
    param_set = np.asarray(start_params)
    for i in range(runs):
        result = minimize(weib_loglikelihood, param_set, method = 'L-BFGS-B', 
                      bounds = ((0.0001,None), (0.0001,None),(0.0001,None)))
        fit_params = list(result.get('x'))
        if fit_params[0] > large_params[0] or fit_params[1] > large_params[1] or fit_params[2] > large_params[2]:
            param_set += np.array([np.random.normal(0, start_params[0]/10), np.random.normal(0, start_params[1]/10),
                                  np.random.normal(0, start_params[2]/10)])
            if i == runs-1:
                fit_params = None
            continue
        else:
            break
    return fit_params     # [a,b,alpha]


#fits lognormal to rest of comments, returns params mu, sigma
def phi_parameters_estimation(hawkes_others, runs = 10, large_params = [20, 20], start_params = [4.,2.]):

    def lognorm_loglikelihood(var): # var = [mu,sigma]
        t_n = hawkes_others[-1]
        f = (-1/2-(1/2)*erf((np.log(t_n)-var[0])/(np.sqrt(2)*var[1]))) + len(hawkes_others)*np.log(1/(var[1]*np.sqrt(2*np.pi)))
        for t in hawkes_others:
            f+= -(np.log(t)-var[0])**2/(2*var[1]**2)-np.log(t)
        return (-1)*f
    
    param_set = np.asarray(start_params)
    for i in range(runs):
        result = minimize(lognorm_loglikelihood, param_set, 
                                        method = 'L-BFGS-B', 
                                        bounds = ((0.0001,None), (0.0001,None)))
        fit_params = list(result.get('x'))
#         print("Current params:", param_set, 
#                   "fit_params:", fit_params,
#                   "L=", lognorm_loglikelihood(fit_params))
        if fit_params[0] > large_params[0] or fit_params[1] > large_params[1]:
            param_set += np.array([np.random.normal(0, start_params[0]/10), np.random.normal(0, start_params[1]/10)])
            if i == runs-1:
                fit_params = None
            continue
        else:
            break
    return fit_params  # [mu, sigma]


#estimate branching number n_b: 1 - (root degree / (total nodes - 1))
def nb_parameters_estimation(tree, root):
    f = 1-tree.degree(root)/(nx.number_of_nodes(tree)-1)
    return f


# # Prediction functions

# In the next cell we present functions that generate the Poisson process with intensities $\mu(t)$ and $\phi(t)$ starting from the *start_time* with *params*. The process ends either 1) when the gap between consecutive events is more than *T* minutes, or 2) when the number of generated events is greater than upper bound *N_max*.

#mu poisson times - generated based on root-children weibull distribution (need a, b, alpha params)
#bounded by inter-event time of 7200 minutes = 120 hours or max number of children 2000 

#homogenous poisson process, but with a variable base intensity mu defined by the weibull kernel
def generate_mu_poisson_times(start_time, params, T = 7200, N_max = 2000):   # Weibull kernel
    (a, b, alpha) = params

    #evaluate weibull at time t, return result
    def mu_func(t, a, b, alpha): # a>0, b>0, alpha>0  --- Weibull
        f = (a*alpha/b)*(t/b)**(alpha-1)*np.exp(-(t/b)**alpha)
        return f

    thin_poisson_times = []
    t = start_time

    #shape param > 1, bell-shape curve
    if alpha>1:
        if t>b*((alpha-1)/alpha)**(1/alpha):		#??
            lbd = mu_func(t,a,b, alpha)
        else:
            lbd = (a*alpha/b)*((alpha-1)/alpha)**(1-1/alpha)*np.exp(-((alpha-1)/alpha))

    #shape param between 0 and 1, decreasing curve
    if alpha>0 and alpha<1:
        lbd = mu_func(t,a,b,alpha)		#call method to get weibull at time t

    #lbd = lambda* = thinning rate = upper bound of mu(t) for current time interval = rate at which to generate top-level replies

    #thinning algorithm to generate times - see page 10 of Hawkes chapter
    while True:
        e = np.random.uniform(low=0, high=1)		#sample between 0 and 1 uniform distribution
        t += -np.log(e)/lbd 						#compute waiting time -ln(e)/lambda, add to current time t

        #accept or reject?
        U = np.random.uniform(low=0, high=1)		#draw another value (s in chapter)
        #if random value less than ratio of true event rate to thinning rate, accept
        if U < mu_func(t,a,b, alpha) / lbd:
            thin_poisson_times.append(t)		#save this event time

            #update thinning rate if necessary for next event generation
            if (alpha>1 and t>b*((alpha-1)/alpha)**(1/alpha)) or (alpha>0 and alpha<1):
                lbd = mu_func(t,a,b, alpha)

        #quit if reached inter-arrival threshold
        if len(thin_poisson_times) > 0:
            if t-thin_poisson_times[-1]>T:
                break
        else:
            if t>T:
                break
        #quit if generated too many events
        if len(thin_poisson_times)>N_max:
            return thin_poisson_times

    return thin_poisson_times

#generate lower-level comments, with intensity defined by log-normal distribution
def generate_phi_poisson_times(start_time, params, n_b, T = 7200, N_max = 200): # Log-normal kernel
    (mu, sigma) = params

    #evaluate log-normal distribution at time t (requires fitted params mu, sigma, and n_b)
    def intensity(t, mu, sigma, n_b):
        if t>0:
            lbd = n_b*(1/(sigma*t*np.sqrt(2*np.pi)))*np.exp(-((np.log(t)-mu)**2)/(2*sigma**2))
        else:
            lbd = 0
        return lbd

    thin_poisson_times = []
    t = start_time

    #get lambda* = thinning rate = event rate = upper bound of intensity
    if t>np.exp(mu-(sigma**2)):  
        lbd = intensity(t, mu, sigma, n_b)
    else:
        lbd = n_b*(np.exp(sigma**2-mu)/(sigma*np.sqrt(2*np.pi)))*np.exp(-(sigma**2)/2)

    #thinning process to generate events
    while True:
        e = np.random.uniform(low=0, high=1)	#sample from uniform distribution
        t += -np.log(e)/lbd 					#compute inter-event time and add to current time t

        #accept or reject?
        U = np.random.uniform(low=0, high=1)	#sample again
        #if sample less than ratio of true event rate to thinning rate, accept
        if U < intensity(t, mu, sigma, n_b)/lbd:
            thin_poisson_times.append(t)		#accept this event time
            #update lambda* for next generation cycle, if required
            if t>np.exp(mu-(sigma**2)):
                lbd = intensity(t, mu, sigma, n_b)

        #stop if events too far apart
        if len(thin_poisson_times)>0:
            if t-thin_poisson_times[-1]>T:
                break
        else:
            if t>T:
                break

        #stop if too many events
        if len(thin_poisson_times)>N_max:
            return thin_poisson_times
    return thin_poisson_times


# The function below simulates the discussion tree from the initially observed subtree *given_tree*. *Start_time* is the age of the *given_tree* (or t_learn). First we generate comments to already existing comments, then generate further possible comments to the post and their further comments. Tree simulation is successful if the total number of nodes is less than *N_max*.

#overall simulation function, work from root down, generating as we go
#starts from existing tree, at age of that tree
#requires fitted weibull and log-normal distribution to define hawkes process
def simulate_comment_tree(given_tree, start_time, params_mu, params_phi, n_b, N_max = 2000):

	#stopping conditions: max inter-event time and max number of events
    T = 7200  # hard set: maximum possible time gap between consecutive events for root comments
    T2 = 7200  # hard set: -//- for comments to comments
    #also cap number of comments (default function argument)

    #copy tree
    g = nx.Graph()
    g = given_tree.copy()

    root, _ = get_root(g)		#fetch root
    
    node_index = max(g.nodes())		#starting counter for new node indices
    
    further_comment_times = []		#used to hold replies of a single node

    #get list of existing comments to root
    root_node_list = []
    for u in g.neighbors(root):
        root_node_list.append(u)

    #get list of existing comments to other comments
    comment_node_list = []
    for v in list(g.nodes())[1:]:
        if v not in root_node_list:
            comment_node_list.append(v)

    #process all existing comment replies
    while len(comment_node_list) > 0:
    	#grab current comment (reply to another comment), and remove from list
        comment_node = deepcopy(comment_node_list[0])	#copy node
        del comment_node_list[0]						#remove from list
        comment_time = g.node[comment_node]['created']	#grab comment time

        #generate reply events for this comment
        further_comment_times.clear()
        further_comment_times = generate_phi_poisson_times(float(start_time-comment_time), params_phi, n_b, T2 )
        further_comment_times = [i+comment_time for i in further_comment_times]

        tree_node_list = []

        #process all generated replies
        if len(further_comment_times) > 0:
            for t in further_comment_times:
            	#create node, add to graph and list
                node_index += 1
                g.add_node(node_index, created = t, root = False)
                g.add_edge(comment_node, node_index)	#new node is reply to current comment
                tree_node_list.append(node_index)

        #process all newly-generated nodes
        while len(tree_node_list) != 0:
        	#pull  current node, delete from list
            current_node = deepcopy(tree_node_list[0])
            del tree_node_list[0]
            comment_time = g.node[current_node]['created']

            #generate replies to this comment (deeper level now)
            further_comment_times = generate_phi_poisson_times(float(start_time-comment_time), params_phi, n_b, T2 )
            further_comment_times = [i+comment_time for i in further_comment_times]

            #process all newly-generated replies
            if len(further_comment_times) > 0:
                for t2 in further_comment_times:
                	#create node, add to graph and list
                    node_index += 1
                    g.add_node(node_index, created = t2, root = False)
                    g.add_edge(current_node, node_index)	#reply to current generated comment
                    tree_node_list.append(node_index)
                    node_index+=1

            #if too many nodes, quit
            if nx.number_of_nodes(g) > N_max:
                return g, False   
    #end processing comments

    #generate new top-level comments, process each individually
    new_root_comment_times = generate_mu_poisson_times(float(start_time), params_mu, T)
    for t in new_root_comment_times:
    	#create new node, add to graph
        node_index += 1
        g.add_node(node_index, created = t, root = False)
        g.add_edge(root, node_index)

        tree_node_list = []		#clear this list

        #generate replies to this new comment
        new_further_comment_times = generate_phi_poisson_times(0, params_phi, n_b, T2 )
        new_further_comment_times = [i+t for i in new_further_comment_times]
        if len(new_further_comment_times) > 0:
            for t2 in new_further_comment_times:
            	#create node for each, add to graph and list
                node_index += 1
                g.add_node(node_index, created = t2, root = False)
                g.add_edge(node_index, node_index)
                tree_node_list.append(node_index)

        #process the reply-replies, same as above
        while len(tree_node_list) != 0:
            current_node = tree_node_list[0]
            del tree_node_list[0]
            t_offspring = g.node[current_node]['created']
            new_further_comment_times = generate_phi_poisson_times(0, params_phi, n_b, T2 )
            new_further_comment_times = [i+t_offspring for i in new_further_comment_times]
            if len(new_further_comment_times) > 0:
                for t2 in new_further_comment_times:
                    node_index += 1
                    g.add_node(node_index, created = t2, root = False)
                    g.add_edge(current_node, node_index)
                    tree_node_list.append(node_index)

            #quit if too many nodes
            if nx.number_of_nodes(g)>N_max:
                return g, False
    #end processing new root comments

    #finished, return results
    return g, True


# # Technical functions

# Curve_fit of $\mu(t)$ (less precise for prediction)

def mu_func_fit_weibull(list_times, res=60, runs = 10, T_max = 3*1440, large_params = [1000, 10000, 20], start_params = [50, 400, 2.3]):
    def weib_func(t, a, b, alpha): # a>0, b>0, alpha>0  --- Weibull pdf
        return (a*alpha/b)*(t/b)**(alpha-1)*np.exp(-(t/b)**alpha)
    
    bins = np.arange(0, max([T_max, max(list_times)]), res)
    hist, bins = np.histogram(list_times, bins)  # construct histogram of the root comments appearance 
    center_bins = [b+res/2 for b in bins[:-1]]
    
    param_set = np.asarray(start_params)
    print("Start curve_fit estimation:")
    for i in range(runs):
        fit_params, pcov = curve_fit(weib_func, xdata = center_bins, ydata = hist/res, p0 = param_set, 
                                     bounds = (0.0001, 100000))
        if fit_params[0] > large_params[0] or fit_params[1] > large_params[1] or fit_params[2] > large_params[2]:
            print("Current params:", param_set, "fit_params:", fit_params)
            param_set += np.array([np.random.normal(0, start_params[0]/10), np.random.normal(0, start_params[1]/10),
                                  np.random.normal(0, start_params[2]/4)])
            if i == runs-1:
                fit_params = [None, None, None]
            continue
        else:
            break
    return fit_params     # [a,b,alpha]


#given a tree (graph), return the root and it's creation time
def get_root(g):
    for u in g.nodes():
        if (g.node[u]['root']):
            return u, g.node[u]['created']
    return None


#given a tree, return sorted list of comment reply times in minutes (originally stored in seconds)
def get_root_comment_times(tree):
    r, root_time = get_root(tree)
    root_comment_times = []
    for u in tree.neighbors(r):
        root_comment_times.append((tree.node[u]['created'] - root_time)/60)  # in minutes
    root_comment_times.sort()
    return root_comment_times


#get list of comment times of all comments not on root, in minutes
def get_other_comment_times(tree):
    hawkes_others = []
    r, root_time = get_root(tree)
    sh_paths_dict = nx.shortest_path_length(tree, source=r)		#get distance from root to all nodes
    for u, d in sh_paths_dict.items():	#loop (node, distance) pairs
    	#if node is not root and has a neighbor
        if d > 0:
            if tree.degree(u) > 1:
                time_to_add = []
                for v in tree.neighbors(u):
                    time_to_add.append(tree.node[v]['created'])		#add all neighbor times to list
                time_to_add.sort()		#sort
                del time_to_add[0]		#remove first item - parent is not a reply
                time_to_add = [(t-tree.node[u]['created'])/60 for t in time_to_add]		#convert to minutes
                hawkes_others = hawkes_others + time_to_add		#add to overall list
    hawkes_others.sort()	#sort
    return hawkes_others


#given a tree and a time, filter the tree to only nodes created within that time span after the root
#essentially, trunc_value is the age of the tree, and we filter out anything that is too new
def get_trunc_tree(tree, trunc_value):
	#copy tree
    g = nx.Graph()
    g = tree.copy()

    #find root
    for u in g.nodes():
        if (g.node[u]['root']):
            root = u
            break

    #build list of nodes to remove from graph
    nodes_to_delete = []
    #process all nodes
    for u in g.nodes():
        t = (g.node[u]['created']-g.node[root]['created'])/60
        #if node creation time too late, delete it
        if t > trunc_value:
            nodes_to_delete.append(u)

    #delete all flagged nodes
    for u in nodes_to_delete:
        g.remove_node(u)
    g_out = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')

    #convert all node times to minutes since root
    for u in list(g_out.nodes())[1:]:
        g_out.node[u]['created'] = float(g_out.node[u]['created']-g_out.node[0]['created'])/60
    g_out.node[0]['created'] = 0.0
    return g_out


#same as above, but does not relabel nodes and does not convert timestamps to minutes
def get_trunc_tree_no_relabel(tree, trunc_value):
    g = nx.Graph()
    g = tree.copy()
    r, root_creation_time = get_root(tree)
    nodes_to_delete = []
    for u in g.nodes():
        t = (g.node[u]['created']-root_creation_time)/60
        if t > trunc_value:
            nodes_to_delete.append(u)
    for u in nodes_to_delete:
        g.remove_node(u)
    return g


#get size of a tree as total number of nodes
def get_size_tree(tree):
    return nx.number_of_nodes(tree)