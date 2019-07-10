#!/usr/bin/env python
# coding: utf-8

from functions_comparative_hawkes import *

import functions_gen_cascade_model		#use the same load/eval/conversion functions

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



functions_gen_cascade_model.define_vprint(True)


# # Upload a set of trees

# In this work the preprocessed set of trees is used. Each tree is asssumed to be a networkx Graph (undirected) with key node attributes: 'root': Bool - if the node is the post; 'created': int, POSIX timestamp -- creation time of the node. 

#list of trees, each is a networkx graph
'''
dump_filename = './sample_trees.dump'
final_tree_list = pickle.load(open (dump_filename, "rb"))
'''



#load my own tree - pickle of edges and nodes
tree = pickle.load(open('./tree.pkl', "rb"))

#and convert that to a networkx graph
final_tree_list = []
H = nx.Graph()
H.add_nodes_from(tree['nodes'])
H.add_edges_from(tree['edges'])
final_tree_list.append(H)

#print(H)
#print(H.nodes(data=True))
#print(tree.edges())


#test cascade - will our conversion stuff work?
cascade = pickle.load(open('test_cascade.pkl', "rb"))
print("cascade:", cascade)

#convert to networkx graph
graph = functions_gen_cascade_model.cascade_to_graph(cascade)
#print("graph nodes:", graph.nodes(data=True))
#print("graph edges:", graph.edges())

#can we convert a graph back to a cascade?
new_cascade = functions_gen_cascade_model.graph_to_cascade(graph)
#print(new_cascade)


final_tree_list = [graph]


# Create a list of all trees sorted by its size

Otrees_list = sorted(final_tree_list, key=lambda t: nx.number_of_nodes(t), reverse=False)
print(len(Otrees_list), "trees")



# # Main function

# Introducing the parameters


i = 0		#index of tree to sim
t_learn_list = ['4h', '6h', '8h', '12h']		#observation times
trunc_values = [240, 360, 480, 720]				#corresponding times in minutes

tree = Otrees_list[i]
print(len(tree), "nodes in sim tree\n")


# Here is the main code. Go through *trunc_values*, cut the *tree* into *given_tree* available at the current t_learn from *trunc_values*, infer parameters for $\mu(t)$ and $\phi(t)$, grow the tree according to the Hawkes model.


result_dict = {}

root, root_creation_time = get_root(tree)		#fetch root of tree

result_dict['true_size'] = get_size_tree(tree)	#save true size of tree for comarison later
list_hawkes_sizes = [[] for i in range(0,len(trunc_values))]	#init empty list for each of the truncation sizes

run_success = True

#test all training lengths
for t in range(0,len(trunc_values)):
    print("     ---    T_LEARN = ", t_learn_list[t], "   ---")
    t_learn = trunc_values[t]		#current learning time in minutes

    given_tree = get_trunc_tree_no_relabel(tree, t_learn)		#filter tree, only observe stuff in training window

    #break if size of the observed tree is too small for prediction at that moment
    if len(given_tree) <= 10:  
        print("Not enough data for parameters estimation!")
        negative_result = 0
        list_hawkes_sizes[t].append(negative_result)
        print("RUN " + str(i) + ": T_learn: "+t_learn_list[t] + "Not enough data for parameters estimation!")
        run_success = False
        continue

    #fit the weibull based on root comment times
    root_comment_times = get_root_comment_times(given_tree)
    mu_params = mu_parameters_estimation(root_comment_times)		
    if mu_params == None:  # if loglikelihood estimation fails - use curve_fit
        mu_params = mu_func_fit_weibull(root_comment_times)
    print("Mu_params:", mu_params)

    #fit log-normal based on all other comment times
    other_comment_times = get_other_comment_times(given_tree)
    phi_params = phi_parameters_estimation(other_comment_times)
    print("Phi_params:", phi_params)

    #estimate branching factor (average number of replies per comment)
    n_b = nb_parameters_estimation(given_tree, root)
    print("n_b:", n_b)
    
    hawkes_times = []
    given_tree = get_trunc_tree(tree, t_learn)	#filter tree, but this time set timestamps to minutes

    sim_tree, success = simulate_comment_tree(given_tree, t_learn, mu_params, phi_params, n_b)	#simulate!
    if success:
        list_hawkes_sizes[t].append(len(sim_tree))
        print("mean error per distance layer", functions_gen_cascade_model.mean_error_per_distance_layer(cascade, functions_gen_cascade_model.graph_to_cascade(sim_tree)))
    else:
        print('Generation failed! Too many nodes')
        print("RUN " + str(i) + ": T_learn: "+ t_learn_list[t] + ': Generation HAWKES failed! Too many nodes')
        list_hawkes_sizes[t] = [-1]

    print("\n")

result_dict['hawkes_sizes'] = list_hawkes_sizes
result_dict["run_success"] = run_success
print('Sequence done!')


# Output the average relative size error.

for i, size_list in enumerate(result_dict['hawkes_sizes']):
    print("t_learn:", t_learn_list[i], 
          "| avg size error:", np.abs(np.mean(size_list)-result_dict['true_size'])/result_dict['true_size'])