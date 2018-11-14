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
P = MHP(mu = [0.5], alpha = [[0.5]], omega = 0.75)
P.generate_seq(SIM_TIME, init_event = True)
#not really dying, but let's go with that for now

#add these comments as children of the root
root.add_children(P.data[:,0])

nodes = [[x, 1] for x in root.children]		#list of nodes to process (ie, generate children for)

#now, can we generate child events for each of those parent comments? following a different distribution?
curr_level = 0
nodes_at_level = 1
for node_pair in nodes:
	node = node_pair[0]
	level = node_pair[1]

	if level != curr_level:
		print("level", curr_level, ":", nodes_at_level, "nodes")
		curr_level = level
	else:
		nodes_at_level += 1

	if level == 15:
		print("too deep, terminating")
		break

	#init Hawkes process - taper the mu parameter off so the comment tree dies off as we get deeper
	subP = MHP(mu = [0.05 / level], alpha = [[0.75]], omega = 1)

	#generate event sequence
	#no initial event here (can't comment at the same time as the parent goes up!)
	#only simulate for time remaining in our sim period
	subP.generate_seq(SIM_TIME - node.id)
	#offset times by parent time, so they occur after
	for i in range(subP.data.shape[0]):
		subP.data[i][0] += node.id

	#add these new child nodes
	node.add_children(subP.data[:,0])

	#and add the new node to list of nodes to process
	nodes.extend([[x, level+1] for x in node.children])

#plot the tree!
tree = CascadeTree(root)
tree.viz_graph("tree.png")