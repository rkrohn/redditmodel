from MHP import MHP
import numpy as np
import os
from CascadeTree import CascadeTree
from Node import Node

#create directory for output plots if it does not exist
if not os.path.exists("tree_plots"):
    os.makedirs("tree_plots")

#univariate process with default parameters mu=[0.1], alpha=[[0.5]], and omega=1.0
#sequence is stored as P.data, a numpy.ndarray with 2 columns: the first column with the timestamps, the second with the stream assignment (in this case there is only one stream)
'''
P = MHP()
P.generate_seq(60)
P.plot_events(filename="tree_plots/univariate.png")
P.plot_rates(filename="tree_plots/univariate_events_and_rates.png")
'''

#multivariate, custom parameters
'''
m = np.array([0.2, 0.0, 0.0])
a = np.array([[0.1, 0.0, 0.0], 
              [0.9, 0.0, 0.0],
              [0.0, 0.9, 0.0]])
w = 3.1

P = MHP(mu=m, alpha=a, omega=w)
P.generate_seq(60)
P.plot_events(filename="tree_plots/multivariate_events.png")
P.plot_rates(filename="tree_plots/multivariate_events_and_rates.png")
'''


#build a tree?
SIM_TIME = 60		#time units to simulate

root = Node('POST', None)	#root has no parent

#first, get top-level comments - ie, replies to the post
#just some random params for now that (hopefully) die off eventually
P = MHP(mu = [0.05], alpha = [[0.5]], omega = 0.75)
P.generate_seq(SIM_TIME, init_event = True)
#print(P.data[:,0])
P.plot_rates(filename="tree_plots/root_test.png", horizon = 60)
#not really dying, but let's go with that for now

#add these comments as children of the root
root.add_children(P.data[:,0])

#now, can we generate child events for each of those parent comments? following a different distribution?
for parent in root.children:
	print(parent.id)
	subP = MHP(mu = [0.1], alpha = [[0.75]], omega = 1)
	#no initial event here (can't comment at the same time as the parent goes up!)
	#only simulate for time remaining in our sim period
	subP.generate_seq(SIM_TIME - parent.id)
	#offset times by parent time
	for i in range(subP.data.shape[0]):
		subP.data[i][0] += parent.id
	print("   ", subP.data[:,0])

	#add child nodes
	parent.add_children(subP.data[:,0])


#plot the tree!
tree = CascadeTree(root)
tree.viz_graph("tree.png")