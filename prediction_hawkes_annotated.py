#!/usr/bin/env python
# coding: utf-8

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
        #special-case to prevent infinte returns
        if math.isinf(f):
        	f = 1.0
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
            #print("accept time", t)

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
    #print("root children", root_node_list)

    #get list of existing comments to other comments
    comment_node_list = []
    for v in g.nodes()[1:]:
        if v not in root_node_list:
            comment_node_list.append(v)
    #print("comments", comment_node_list)

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

            #generate replies to this new comment (deeper level now)
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
    #def weib_func(t, a, b, alpha): # a>0, b>0, alpha>0  --- Weibull pdf
    #    return (a*alpha/b)*(t/b)**(alpha-1)*np.exp(-(t/b)**alpha)
    
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
def get_root_comment_times(tree, seconds_to_minutes = True):
    r, root_time = get_root(tree)
    root_comment_times = []
    for u in tree.neighbors(r):
        if seconds_to_minutes:
            root_comment_times.append((tree.node[u]['created'] - root_time)/60)  # in minutes
        else:
            root_comment_times.append(tree.node[u]['created'] - root_time)  # no conversion
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
    for u in g_out.nodes()[1:]:
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

#visualize graph, save to file (probably not going to look very pretty, but you can usually identify the root)
def viz_graph(g, filename):
		# same layout using matplotlib with no labels
		plt.clf()
		pos = graphviz_layout(g, prog='dot')
		nx.draw(g, pos, with_labels=False, arrows=False, node_size=15)
		plt.savefig(filename)

#given a tree and root, count the number of nodes on each level
def count_nodes_per_level(g, root):
	paths = nx.shortest_path_length(g, root)
	depth_counts = defaultdict(int)
	for node, depth in paths.items():
		depth_counts[depth] += 1

	return depth_counts

def weib_func(t, a, b, alpha): # a>0, b>0, alpha>0  --- Weibull pdf
		return (a*alpha/b)*(t/b)**(alpha-1)*np.exp(-(t/b)**alpha)

#plot input top-level comment distribution vs. fitted Weibull curve
def plot_dist_and_fit(times, params, filename):	

	plt.clf()
	fig, ax1 = plt.subplots()
	
	ax1.hist(times, bins=50, normed=False)	#plot histogram for true post distribution (observed comments only)
	ax1.set_xlabel('time (minutes)')
	ax1.set_ylabel('root comment frequency')

	#plot fitted weibull curve
	ax2 = ax1.twinx()
	x = np.arange(0, max(times), 1)
	y = [weib_func(t, params[0], params[1], params[2]) for t in x]
	ax2.plot(x, y, 'r-')
	ax2.set_ylabel('Weibull fit', color='r')

	plt.savefig("hawkes_testing/%s" % filename)

#given observed and simulated comment times, and fitted Weibull params, plot all the things
def plot_all(obs, sim, actual, params, fit_full, filename):

	plt.clf()
	fig, ax1 = plt.subplots(figsize=(8, 6))

	max_time = max(sim + actual) + 5	#last time, +5 for plot buffer

	#first: plot side-by-side histogram of entire simulated tree (including observed) against actual tree
	
	#plot histogram for simulated tree, stacking observed and simulated together
	bins = np.linspace(0, max_time, 30)
	ax1.hist([actual, obs + sim], bins, stacked=False, normed=False, color=["blue", "green"], label=["actual", "simulated"])	
	ax1.set_xlabel('time (minutes)')
	ax1.set_ylabel('root comment frequency')

	#then, plot observed times on top of the final simulated version to emulate a stacked graph - if there were observed root comments
	if len(obs) > 0:
		ax1.hist([[], obs], bins, normed=False, color=["red", "orange"], label=["", "given"])

	#plot fitted weibull curve
	ax2 = ax1.twinx()
	x = np.arange(0, max(sim), 1)
	y = [weib_func(t, params[0], params[1], params[2]) for t in x]
	ax2.plot(x, y, 'r-', label="Weibull fit")
	if fit_full:
		ax2.set_ylabel('Weibull fit (full tree)', color='r')
	else:
		ax2.set_ylabel('Weibull fit (observed tree)', color='r')

	ax1.legend()
	plt.savefig("hawkes_testing/%s" % filename)


# # Main function

print("")

# # Upload a set of trees

# In this work the preprocessed set of trees is used. Each tree is asssumed to be a networkx Graph (undirected) with key node attributes: 'root': Bool - if the node is the post; 'created': int, POSIX timestamp -- creation time of the node. 

dump_filename = './sample_trees.dump'
final_tree_list = pickle.load(open (dump_filename, "rb"))

# Create a list of all trees sorted by its size

Otrees_list = sorted(final_tree_list, key=lambda t: nx.number_of_nodes(t), reverse=False)
print("read", len(Otrees_list), "trees")

# Introducing the parameters

i = 13
#sim_num_runs = 50
sim_num_runs = 1
#t_learn_list = ['4h', '6h', '8h', '12h']
t_learn_list = [ '4h' ]
#trunc_values = [240, 360, 480, 720]
trunc_values = [240]

#new parameters - let's make this code do something it's not designed for!
TRAIN_FULL_TREE = True 			#if True, train on the entire tree, not just a portion
								#if False, train on a portion of the tree dictated by trunc_values

PREDICT_FROM_ROOT = True 		#if True, predict new tree structure from an empty root (no observed comments)
								#if False, predict additional comments from the observed range defined by trunc_values


#pull one tree for testing
tree = Otrees_list[i]
print("tree size", len(tree))
viz_graph(tree, "hawkes_testing/input_tree.png")


# Here is the main code. Go through *trunc_values*, cut the *tree* into *given_tree* available at the current t_learn from *trunc_values*, infer parameters for $\mu(t)$ and $\phi(t)$, grow the tree according to the Hawkes model.

result_dict = {}

root, root_creation_time = get_root(tree)		#fetch root of tree
#print("root created", root_creation_time)

result_dict['true_size'] = get_size_tree(tree)	#save true size of tree for comarison later
list_hawkes_sizes = [[] for i in range(0,len(trunc_values))]	#init empty list for each of the truncation sizes

run_success = True

#test all training lengths
for trunc_time in range(0,len(trunc_values)):
    print("\n     ---    T_LEARN = ", t_learn_list[trunc_time], "   ---")

    #filter tree to training window
    t_learn = trunc_values[trunc_time]		#current learning time in minutes
    given_tree = get_trunc_tree_no_relabel(tree, t_learn)		#filter tree, only observe stuff in training window
    print("filter tree to", len(given_tree), "nodes")
    viz_graph(given_tree, "hawkes_testing/filtered_tree.png")

    #break if size of the observed tree is too small for prediction at that moment
    if len(given_tree) <= 10:  
        print("Not enough data for parameters estimation!")
        negative_result = 0
        list_hawkes_sizes[t].append(negative_result)
        print("RUN " + str(i) + ": T_learn: "+t_learn_list[t] + "Not enough data for parameters estimation!")
        run_success = False
        continue

    #get root comment times (in minutes) for fitting
    if TRAIN_FULL_TREE:
    	root_comment_times = get_root_comment_times(tree)	#entire cascade tree
    	print(len(root_comment_times), "root comment times in full tree")
    else:
    	root_comment_times = get_root_comment_times(given_tree)		#filtered tree
    	print(len(root_comment_times), "root comment times in filtered tree,", len(get_root_comment_times(tree)), "in full tree\n")
    #print(str(root_comment_times), "(minutes)\n")

    #fit the weibull based on root comment times
    mu_params = mu_parameters_estimation(root_comment_times)		
    if mu_params == None:  # if loglikelihood estimation fails - use curve_fit
        mu_params = mu_func_fit_weibull(root_comment_times)
    print("Mu_params:", "\n   a", mu_params[0], "\n   b", mu_params[1], "\n   alpha", mu_params[2], "\n")	#a, b, alpha

    plot_dist_and_fit(root_comment_times, mu_params, "observed_dist_fit.png")

    #fit log-normal based on all other comment times
    if TRAIN_FULL_TREE:
    	other_comment_times = get_other_comment_times(tree)		#full tree
    else:
    	other_comment_times = get_other_comment_times(given_tree)	#filtered tree
    phi_params = phi_parameters_estimation(other_comment_times)
    print("Phi_params:", "\n   mu", phi_params[0], "\n   sigma", phi_params[1], "\n")	#mu, sigma

    #estimate branching factor (average number of replies per comment)
    if TRAIN_FULL_TREE:
    	n_b = nb_parameters_estimation(tree, root)
    else:
    	n_b = nb_parameters_estimation(given_tree, root)
    print("branching factor n_b:", n_b, "\n")
    
    root_comment_times = get_root_comment_times(given_tree)		#done training, get filtered root_comment_times
    given_tree = get_trunc_tree(tree, t_learn)	#filter tree, but this time set timestamps to minutes

    #print level breakdown of source tree
    depth_counts = count_nodes_per_level(tree, root)
    print("full original tree nodes per level:")
    for depth, count in depth_counts.items():
    	print(depth, ":", count)
    print("")

    add_count = sim_num_runs / 5
    for j in range(0, sim_num_runs):		#repeated simulations

        #intermittent prints to show progress
        #if j % add_count == 0:
        #    print(j, "of", sim_num_runs, "HAWKES trees simulated for the tree i =", i)

        if PREDICT_FROM_ROOT:
	        #simulate from empty starting tree (just the root)
	        empty_tree = nx.Graph()
	        empty_tree = given_tree.copy()
	        r, root_creation_time = get_root(given_tree)
	        nodes_to_delete = []
	        for u in empty_tree.nodes():
	            if empty_tree.node[u]['root'] == False:
	                nodes_to_delete.append(u)
	        for u in nodes_to_delete:
	            empty_tree.remove_node(u)
	        sim_tree, success = simulate_comment_tree(empty_tree, 0, mu_params, phi_params, n_b)
        else:
        	#simulate by added to observed tree
        	sim_tree, success = simulate_comment_tree(given_tree, t_learn, mu_params, phi_params, n_b)

        #viz just the first sim tree for now (slows the process waaaaay down to do them all)
        if j == 0:
        	viz_graph(sim_tree, "hawkes_testing/sim_tree_%s.png" % j)
        if success:
            list_hawkes_sizes[trunc_time].append(len(sim_tree))
        else:
            print('Generation failed! Too many nodes')
            print("RUN " + str(i) + ": T_learn: "+ t_learn_list[t] + ': Generation HAWKES failed! Too many nodes')
            list_hawkes_sizes[trunc_time] = [-1]
            break

        #print final simulated tree sizes against initial tree size (filtered tree starting point)
        depth_counts = count_nodes_per_level(sim_tree, get_root(sim_tree)[0])
        depth_counts_start = count_nodes_per_level(given_tree, get_root(given_tree)[0])
        print("sim tree nodes per level:")
        for depth, count in depth_counts.items():
            print(depth, ":", count, "(" + str(depth_counts_start[depth] if depth in depth_counts_start and PREDICT_FROM_ROOT == False else 0), "given)")
        print("")

        #get all root comment times from the simulation
        sim_root_comment_times = get_root_comment_times(sim_tree, seconds_to_minutes=False)
        plot_dist_and_fit(sim_root_comment_times, mu_params, "simulate_dist_fit.png")

        #get comment times from simulation separate from the observed/given; also get actual comment times        
       	if PREDICT_FROM_ROOT:		#if predicted from empty root, no observed comments
       		root_comment_times = []
        new_root_comment_times = sim_root_comment_times[len(root_comment_times):]
        actual_comment_times = get_root_comment_times(tree)
        plot_all(root_comment_times, new_root_comment_times, actual_comment_times, mu_params, TRAIN_FULL_TREE, "dist_fit.png")

    print("")

print('Sequence done!\n')
#save generated tree sizes
result_dict['hawkes_sizes'] = list_hawkes_sizes
print("generated tree sizes:", list_hawkes_sizes)
result_dict["run_success"] = run_success

# Output the average relative size error.

for i, size_list in enumerate(result_dict['hawkes_sizes']):
    print("t_learn:", t_learn_list[i], 
          "| avg size error:", np.abs(np.mean(size_list)-result_dict['true_size'])/result_dict['true_size'])
print("")





