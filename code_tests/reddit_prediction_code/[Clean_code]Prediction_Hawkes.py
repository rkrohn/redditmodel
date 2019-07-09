#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # Upload a set of trees

# In this work the preprocessed set of trees is used. Each tree is asssumed to be a networkx Graph (undirected) with key node attributes: 'root': Bool - if the node is the post; 'created': int, POSIX timestamp -- creation time of the node. 

# In[2]:


dump_filename = './sample_trees.dump'
final_tree_list = pickle.load(open (dump_filename, "rb"))


# Create a list of all trees sorted by its size

# In[3]:


Otrees_list = sorted(final_tree_list, key=lambda t: nx.number_of_nodes(t), reverse=False)
print(len(Otrees_list))


# ## Inference of parameters of a process

# Estimate parameters of $\mu(t)$ and $\phi(t)$ given the time stamps. Initial guess for parameters may be bad for convergence (one of the parameters is larger than in *large_parameters*), thus a random perturbation is introduced (maximum *runs* times).

# In[4]:


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


# In[5]:


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


# In[6]:


def nb_parameters_estimation(tree, root):
    f = 1-tree.degree(root)/(nx.number_of_nodes(tree)-1)
    return f


# # Prediction functions

# In the next cell we present functions that generate the Poisson process with intensities $\mu(t)$ and $\phi(t)$ starting from the *start_time* with *params*. The process ends either 1) when the gap between consecutive events is more than *T* minutes, or 2) when the number of generated events is greater than upper bound *N_max*.

# In[7]:


def generate_mu_poisson_times(start_time, params, T = 7200, N_max = 2000):   # Weibull kernel
    (a, b, alpha) = params
    def mu_func(t, a, b, alpha): # a>0, b>0, alpha>0  --- Weibull
        f = (a*alpha/b)*(t/b)**(alpha-1)*np.exp(-(t/b)**alpha)
        return f

    thin_poisson_times = []
    t = start_time
    if alpha>1:
        if t>b*((alpha-1)/alpha)**(1/alpha):
            lbd = mu_func(t,a,b, alpha)
        else:
            lbd = (a*alpha/b)*((alpha-1)/alpha)**(1-1/alpha)*np.exp(-((alpha-1)/alpha))
    if alpha>0 and alpha<1:
        lbd = mu_func(t,a,b,alpha)
    while True:
        e = np.random.uniform(low=0, high=1)
        t += -np.log(e)/lbd
        U = np.random.uniform(low=0, high=1)
        if U<mu_func(t,a,b, alpha)/lbd:
            thin_poisson_times.append(t)
            if (alpha>1 and t>b*((alpha-1)/alpha)**(1/alpha)) or (alpha>0 and alpha<1):
                lbd = mu_func(t,a,b, alpha)
        if len(thin_poisson_times)>0:
            if t-thin_poisson_times[-1]>T:
                break
        else:
            if t>T:
                break
        if len(thin_poisson_times)>N_max:
            return thin_poisson_times
    return thin_poisson_times

def generate_phi_poisson_times(start_time, params, n_b, T = 7200, N_max = 200): # Log-normal kernel
    (mu, sigma) = params
    def intensity(t, mu, sigma, n_b):
        if t>0:
            lbd = n_b*(1/(sigma*t*np.sqrt(2*np.pi)))*np.exp(-((np.log(t)-mu)**2)/(2*sigma**2))
        else:
            lbd = 0
        return lbd
    thin_poisson_times = []
    t = start_time
    if t>np.exp(mu-(sigma**2)):  
        lbd = intensity(t, mu, sigma, n_b)
    else:
        lbd = n_b*(np.exp(sigma**2-mu)/(sigma*np.sqrt(2*np.pi)))*np.exp(-(sigma**2)/2)
    while True:
        e = np.random.uniform(low=0, high=1)
        t += -np.log(e)/lbd
        U = np.random.uniform(low=0, high=1)
        if U<intensity(t, mu, sigma, n_b)/lbd:
            thin_poisson_times.append(t)
            if t>np.exp(mu-(sigma**2)):
                lbd = intensity(t, mu, sigma, n_b)
        if len(thin_poisson_times)>0:
            if t-thin_poisson_times[-1]>T:
                break
        else:
            if t>T:
                break
        if len(thin_poisson_times)>N_max:
            return thin_poisson_times
    return thin_poisson_times


# The function below simulates the discussion tree from the initially observed subtree *given_tree*. *Start_time* is the age of the *given_tree* (or t_learn). First we generate comments to already existing comments, then generate further possible comments to the post and their further comments. Tree simulation is successful if the total number of nodes is less than *N_max*.

# In[8]:


def simulate_comment_tree(given_tree, start_time, params_mu, params_phi, n_b, N_max = 2000):
    T = 7200  # hard set: maximum possible time gap between consecutive events for root comments
    T2 = 7200  # hard set: -//- for comments to comments
    g = nx.Graph()
    g = given_tree.copy()
    root, _ = get_root(g)
    root_node_list = []
    node_index = max(g.nodes())
    comment_node_list = []
    further_comment_times = []
    for u in g.neighbors(root):
        root_node_list.append(u)
    for v in g.nodes()[1:]:
        if v not in root_node_list:
            comment_node_list.append(v)
    while len(comment_node_list)>0:
        comment_node = deepcopy(comment_node_list[0])
        del comment_node_list[0]
        comment_time = g.node[comment_node]['created']
        further_comment_times.clear()
        further_comment_times = generate_phi_poisson_times(float(start_time-comment_time), params_phi, n_b, T2 )
        further_comment_times = [i+comment_time for i in further_comment_times]
        tree_node_list = []
        if len(further_comment_times)>0:
            for t in further_comment_times:
                node_index += 1
                g.add_node(node_index, created = t, root = False)
                g.add_edge(comment_node, node_index)
                tree_node_list.append(node_index)
        while len(tree_node_list)!=0:
            current_node = deepcopy(tree_node_list[0])
            del tree_node_list[0]
            comment_time = g.node[current_node]['created']
            further_comment_times = generate_phi_poisson_times(float(start_time-comment_time), params_phi, n_b, T2 )
            further_comment_times = [i+comment_time for i in further_comment_times]
            if len(further_comment_times)>0:
                for t2 in further_comment_times:
                    node_index += 1
                    g.add_node(node_index, created = t2, root = False)
                    g.add_edge(current_node, node_index)
                    tree_node_list.append(node_index)
                    node_index+=1
            if nx.number_of_nodes(g)>N_max:
                return g, False   
    new_root_comment_times = generate_mu_poisson_times(float(start_time), params_mu, T)
    for t in new_root_comment_times:
        node_index += 1
        g.add_node(node_index, created = t, root = False)
        g.add_edge(root, node_index)
        tree_node_list = []
        new_further_comment_times = generate_phi_poisson_times(0, params_phi, n_b, T2 )
        new_further_comment_times = [i+t for i in new_further_comment_times]
        if len(new_further_comment_times)>0:
            for t2 in new_further_comment_times:
                node_index += 1
                g.add_node(node_index, created = t2, root = False)
                g.add_edge(node_index, node_index)
                tree_node_list.append(node_index)
        while len(tree_node_list)!=0:
            current_node = tree_node_list[0]
            del tree_node_list[0]
            t_offspring = g.node[current_node]['created']
            new_further_comment_times = generate_phi_poisson_times(0, params_phi, n_b, T2 )
            new_further_comment_times = [i+t_offspring for i in new_further_comment_times]
            if len(new_further_comment_times)>0:
                for t2 in new_further_comment_times:
                    node_index += 1
                    g.add_node(node_index, created = t2, root = False)
                    g.add_edge(current_node, node_index)
                    tree_node_list.append(node_index)
            if nx.number_of_nodes(g)>N_max:
                return g, False
    return g, True


# # Technical functions

# Curve_fit of $\mu(t)$ (less precise for prediction)

# In[9]:


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


# In[10]:


def get_root(g):
    for u in g.nodes():
        if (g.node[u]['root']):
            return u, g.node[u]['created']
    return None

def get_root_comment_times(tree):
    r, root_time = get_root(tree)
    root_comment_times = []
    for u in tree.neighbors(r):
        root_comment_times.append((tree.node[u]['created'] - root_time)/60)  # in minutes
    root_comment_times.sort()
    return root_comment_times

def get_other_comment_times(tree):
    hawkes_others = []
    r, root_time = get_root(tree)
    sh_paths_dict = nx.shortest_path_length(tree, source=r)
    for u, d in sh_paths_dict.items():
        if d > 0:
            if tree.degree(u)>1:
                time_to_add = []
                for v in tree.neighbors(u):
                    time_to_add.append(tree.node[v]['created'])
                time_to_add.sort()
                del time_to_add[0]
                time_to_add = [(t-tree.node[u]['created'])/60 for t in time_to_add]
                hawkes_others = hawkes_others + time_to_add
    hawkes_others.sort()
    return hawkes_others

def get_trunc_tree(tree, trunc_value):
    g = nx.Graph()
    g = tree.copy()
    for u in g.nodes():
        if (g.node[u]['root']):
            root = u
            break
    nodes_to_delete = []
    for u in g.nodes():
        t = (g.node[u]['created']-g.node[root]['created'])/60
        if t>trunc_value:
            nodes_to_delete.append(u)
    for u in nodes_to_delete:
        g.remove_node(u)
    g_out = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')
    for u in g_out.nodes()[1:]:
        g_out.node[u]['created'] = float(g_out.node[u]['created']-g_out.node[0]['created'])/60
    g_out.node[0]['created'] = 0.0
    return g_out

def get_trunc_tree_no_relabel(tree, trunc_value):
    g = nx.Graph()
    g = tree.copy()
    r, root_creation_time = get_root(tree)
    nodes_to_delete = []
    for u in g.nodes():
        t = (g.node[u]['created']-root_creation_time)/60
        if t>trunc_value:
            nodes_to_delete.append(u)
    for u in nodes_to_delete:
        g.remove_node(u)
    return g

def get_size_tree(tree):
    return nx.number_of_nodes(tree)


# # Main function

# Introducing the parameters

# In[13]:


i = 13
sim_num_runs = 50
t_learn_list = ['4h', '6h', '8h', '12h']
trunc_values = [240, 360, 480, 720]

tree = Otrees_list[i]
len(tree)


# Here is the main code. Go through *trunc_values*, cut the *tree* into *given_tree* available at the current t_learn from *trunc_values*, infer parameters for $\mu(t)$ and $\phi(t)$, grow the tree according to the Hawkes model.

# In[14]:


result_dict = {}

root, root_creation_time = get_root(tree)

result_dict['true_size'] = get_size_tree(tree)
list_hawkes_sizes = [[] for i in range(0,len(trunc_values))]

run_success = True
for t in range(0,len(trunc_values)):
    print("     ---    T_LEARN = ", t_learn_list[t], "   ---")
    t_learn = trunc_values[t]
    given_tree = get_trunc_tree_no_relabel(tree, t_learn)
    if len(given_tree) <= 10:  # break if size of the observed tree is too small for prediction at that moment
        print("Not enough data for parameters estimation!")
        negative_result = 0
        list_hawkes_sizes[t].append(negative_result)
        print("RUN " + str(i) + ": T_learn: "+t_learn_list[t] + "Not enough data for parameters estimation!")
        run_success = False
        continue
    root_comment_times = get_root_comment_times(given_tree)
    mu_params = mu_parameters_estimation(root_comment_times)
    if mu_params == None:  # if loglikelihood estimation fails - use curve_fit
        mu_params = mu_func_fit_weibull(root_comment_times)
    print("Mu_params:", mu_params)
    other_comment_times = get_other_comment_times(given_tree)
    phi_params = phi_parameters_estimation(other_comment_times)
    print("Phi_params:", phi_params)
    n_b = nb_parameters_estimation(given_tree, root)
    print("n_b:", n_b)
    
    hawkes_times = []
    given_tree = get_trunc_tree(tree, t_learn)
    add_count = sim_num_runs/5
    for j in range(0,sim_num_runs):
        hawkes_times.clear()
        if j%add_count==0:
            print(j, " of HAWKES trees simulated for the tree i=", i)
        sim_tree, success = simulate_comment_tree(given_tree, t_learn, mu_params, phi_params, n_b)
        if success:
            list_hawkes_sizes[t].append(len(sim_tree))
        else:
            print('Generation failed! Too many nodes')
            print("RUN " + str(i) + ": T_learn: "+ t_learn_list[t] + ': Generation HAWKES failed! Too many nodes')
            list_hawkes_sizes[t] = [-1]
            break
    print("\n")
result_dict['hawkes_sizes'] = list_hawkes_sizes
result_dict["run_success"] = run_success
print('Sequence done!')


# Output the average relative size error.

# In[15]:


for i, size_list in enumerate(result_dict['hawkes_sizes']):
    print("t_learn:", t_learn_list[i], 
          "| avg size error:", np.abs(np.mean(size_list)-result_dict['true_size'])/result_dict['true_size'])


# In[ ]:




