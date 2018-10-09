import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import Node

class CascadeTree:
	#pass in tree as Node object of root (see Node.py class)
	def __init__(self, root):
		self.G = nx.DiGraph()

		#build graph
		self.G.add_edges_from(self.BFS_edges(root))

	def BFS_edges(self, root):
		edges = []
		nodes = [root]
		while len(nodes) != 0:
			curr = nodes.pop(0)
			if curr.parent != None:
				edges.append((curr.parent, curr.id))
			nodes.extend(curr.children)
		return edges

	def viz_graph(self, filename):
		#this labels nodes with the id (time, in our case)
		'''
		self.p = nx.drawing.nx_pydot.to_pydot(self.G)
		self.p.write_png(filename)
		'''

		# same layout using matplotlib with no labels
		plt.clf()
		pos = graphviz_layout(self.G, prog='dot')
		nx.draw(self.G, pos, with_labels=False, arrows=False, node_size=50)
		plt.savefig(filename)

'''
g=nx.DiGraph()
g.add_edges_from([(1,2), (1,3), (1,4), (2,5), (2,6), (2,7), (3,8), (3,9),
                  (4,10), (5,11), (5,12), (6,13)])
p=nx.drawing.nx_pydot.to_pydot(g)
p.write_png('example.png')
'''

