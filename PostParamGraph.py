#methods for building and performing lookups from the param graph
#one node per post
#connect posts by the same user with an edge of weight = 1
#also connect posts with words in common by a single edge of weight = (# shared) / (# in shortest title)

#a new post introduces a new node to this graph, but with undefined parameters
#use node2vec to get these missing parameters

import string
import numpy
from collections import defaultdict
import itertools
import operator

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from node2vec import Node2Vec


#params are indexed as follows:
#	a 		0
#	lbd 	1
#	k 		2
#	mu 		3
#	sigma	4
#	n_b		5

#global display setting for the class: applies to debug-type output only
DISPLAY = False

#to save an instance of this class, simply use file_utils.save_pickle
#then use file_utils.load_pickle to restore the object

class ParamGraph(object):


	def __init__(self):
		self.graph = None			#line/edge multigraph of posts
		self.post_params = None 	#dictionary of post-ids -> params
		self.post_tokens = None 	#dictionary of post-ids -> set tokens
		self.tokens = None 			#dictionary of tokens -> set of posts
		self.users = None			#user->set of post-ids
		self.model = None 			#node2vec embedding model
	#end __init__


	#given a set of posts and corresponding fitted parameters, build the post parameter graph
	def build_graph(self, posts, params):
		#make sure we have same number of posts and paramters
		if len(posts) != len(params):
			print("Post and paramter sets are not the same size - skipping graph build")
			return
		print("Building param graph for", len(posts), "posts")

		#build post->tokens, user->posts, and tokens->posts dictionaries
		#also save params
		self.__init_dictionaries(posts, params)
		print("   Found", len(self.users), "users")
		print("   Found", len(self.tokens), "tokens")

		#build the multi-graph
		#one node for each post
		#edge of weight=1 connecting posts by the same user
		#edge of weight=(# shared)/(# in shortest title) between posts with common words
		self.graph = nx.MultiGraph()

		#add all as isolated nodes to start, to make sure they're all represented
		self.graph.add_nodes_from(self.post_tokens.keys())

		#add edges with weight=1 between posts by the same user
		for user, posts in self.users.items():
			post_pairs = list(itertools.combinations(posts, 2))		#get all post pairs from this user

			for post_pair in post_pairs:
				self.graph.add_edge(post_pair[0], post_pair[1], weight=1)

		#add edges with weight=(# shared)/(# in shortest title) betwee posts sharing tokens
		#version 1: loop all post pairs - SLOW, DO NOT USE
		'''
		post_pairs = list(itertools.combinations(self.post_tokens.keys(), 2))	#get all post pairs
		for post_pair in post_pairs:
			#compute edge weight based on post token sets
				weight = self.__compute_edge_weight(self.post_tokens[post_pair[0]], self.post_tokens[post_pair[1]])
				#if valid weight, add edge
				if weight != False:
					self.graph.add_edge(post_pair[0], post_pair[1], weight=weight)		#add edge
		'''
		#version2: loop only pairs that we know have tokens in common - FASTER
		for token, posts in self.tokens.items():
			post_pairs = list(itertools.combinations(posts, 2))		#get all post pairs for this token
			for post_pair in post_pairs:
				#already an edge between them? skip
				if self.graph.has_edge(post_pair[0], post_pair[1]):
					continue

				#compute edge weight based on post token sets
				weight = self.__compute_edge_weight(self.post_tokens[post_pair[0]], self.post_tokens[post_pair[1]])
				#if valid weight, add edge
				if weight != False:
					self.graph.add_edge(post_pair[0], post_pair[1], weight=weight)		#add edge

		print("Finished graph has", self.graph.number_of_nodes(), "nodes and", self.graph.size(), "edges")
	#end build_graph


	#given a single post object (sorry, on you to loop), add relevant node to graph
	#this WILL change the class's graph, not make a temporary copy
	def add_post(self, post, params=None):
		#verify that we have a graph
		if self.graph == None:
			print("Must build graph before inferring parameters")
			return False

		#verify that post not already represented in graph
		if post['id_h'] in self.post_tokens:
			print("Post already represented in graph - no changes to graph")
			return False

		#tokenize this post, grab post user
		new_tokens = self.__extract_tokens(post)
		new_user = post['author_h']

		#add node first, to make sure is represented in graph even if isolated - could get connected later
		self.graph.add_node(post['id_h'])

		#add connecting edges for this post to the graph

		#add edges of weight=1 between posts also by this user
		for prev_post in self.users[new_user]:
			self.graph.add_edge(post['id_h'], prev_post)

		#add edges with scaled weight between posts with same tokens as this one
		for token in new_tokens:
			for other_post in self.tokens[token]:
				#already an edge between them? skip
				if self.graph.has_edge(post['id_h'], other_post):
					continue
				#compute edge weight based on post token sets
				weight = self.__compute_edge_weight(self.post_tokens[other_post], new_tokens)
				#if valid weight, add edge
				if weight != False:
					self.graph.add_edge(post['id_h'], other_post, weight=weight)		#add edge

		if DISPLAY:
			print("   Updated graph has", self.graph.number_of_nodes(), "nodes and", self.graph.size(), "edges")

		#add this post to graph tracking
		self.post_tokens[post['id_h']] = new_tokens
		self.users[new_user].add(post['id_h'])
		for token in new_tokens:
			self.tokens[token].add(post['id_h'])
		if params != None:
			self.post_params[post['id_h']] = params

		#invalidate any model
		self.model = None

		return True
	#end add_post


	#assuming a pre-built graph, run node2vec to generate model and embeddings
	def run_node2vec(self):
		#if have a model already, skip rerunning
		if self.model != None:
			print("Reusing existing model")
			return True

		#verify that we have a graph
		if self.graph == None:
			print("Must build graph and compute pagerank before inferring parameters")
			return False

		#precompute probabilities and generate walks
		print("Running node2vec...")
		node2vec = Node2Vec(self.graph, dimensions=16, walk_length=10, num_walks=200, workers=4, quiet=False)	#example uses 64 dimensions and walk_length 10, let's go smaller

		#compute embeddings - dimensions and workers automatically passed from the Node2Vec constructor
		self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

		print("Done")
		return True
	#end run_node2vec


	#given a single post object (already added to graph), and a fitted model, infer parameters
	#must pass in the list of post_nodes returned by add_post
	#mode indicates inference method, must be one of the following:
	#	average - simple average of most-similar to each of the post nodes
	#	weighted - weighted average of most-similar, where weight is similarity
	#	max - take params directly from most-similar of all post nodes (if a node has defined params, use those)
	#if mode is average or weighted, must pass in avg_count, indicating the number of most similar nodes to include in average
	#if skip_default is True, disregard any hardcoded default parameters in averages and most-similar selections
	def infer_params(self, post, mode, skip_default = True, avg_count = None):		
		#verify that post already represented in graph
		if post['id_h'] not in self.post_tokens:
			print("Post not represented in graph - cannot infer parameters")
			return False

		#verify that we have a model
		if self.model == None:
			print("No fitted Node2Vec model - cannot infer parameters")
			return False

		#verify valid mode
		if mode != 'average' and mode != 'weighted' and mode != 'max':
			print("Invalid mode, must be one of \"average\", \"weighted\" or \"max\"")
			return False

		#if mode is average or weighted, make sure an avg_count was given
		if (mode == 'average' or mode == 'weighted') and avg_count == None:
			print("Must specify number of nodes to include in average for this mode")
			return False

		#pull post name - corresponds to post's node
		post_name = post['id_h']

		#if already have params for this node, use those - as long as they don't violate the current skip_default setting
		lookup_params = self.__get_params(post_name, skip_default)
		if lookup_params != False:
			print("Direct param lookup for post", post_name)
			return lookup_params

		#infer parameters using fitted node2vec model, based on most similar node in graph with defined params
		print("Inferring parameters for post", post_name)				

		#init average for 'average' and 'weighted' modes
		avg_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		divisor = 0.0	
		contrib_count = 0		#need avg_count nodes/posts for average and weighted modes

		#look for most similar node to this post's node with defined params

		#pull params from best match with defined/fitted params				
		top_count = 0

		#loop until we get some params
		while True:
			top_count += 10		#increase, in case we had to go further (fetch batches of 10)
			#get most similar nodes
			#returns list of tuples (node, similarity), sorted by similarity
			similar = self.model.wv.most_similar(post_name, topn=top_count)
			#find best match with defined parameters (these are sorted by similarity)
			for match in similar:
				match_params = self.__get_params(match[0], skip_default)
				similarity = match[1]
				#if these params invalid, move to next
				if match_params == False:
					continue

				#if mode is max, this is the best it's going to get - return
				if mode == 'max':
					return match_params

				#otherwise, add this to running average
				contrib_count += 1
				for i in range(6):
					#standard average
					if mode == 'average':
						avg_params[i] += match_params[i]
						divisor += 1
					#weighted average
					elif mode == 'weighted':
						avg_params[i] += match_params[i] * similarity
						divisor += similarity

				#if enough contributors for average or weighted, compute final average and return
				if contrib_count == avg_count:
					avg_params = [param / divisor for param in avg_params]
					return avg_params
	#end infer_params


	#for posts not connected to graph (ie, new user and all new tokens), take the average of all graph params
	#average all nodes together, where an individual node might be an average
	def __avg_params(self, skip_default):
		all_params = []		#list of all params, across entire graph
		#loop all nodes, average
		for node in list(self.graph.nodes()):
			node_params = self.__get_params(node, skip_default)
			#append to list if valid
			if node_params != False:
				all_params.append(node_params)

		#average all of those together
		avg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		divisor = 0
		for item in all_params:
			#if this param set contains any defaults and we want to skip those, skip it
			if skip_default and True in self.__check_default(item):
				continue
			#not defaults, contribute to average
			for i in range(6):
				avg[i] += item[i]
			divisor += 1

		#final average calculation
		if divisor != 0:
			avg = [item / divisor for item in avg]
		else:
			return False

		return avg
	#end __avg_params


	#given token sets from two posts, compute weight of edge between them
	def __compute_edge_weight(self, tokens1, tokens2):
		#get count of words shared by these posts
		intersect_size = len(tokens1.intersection(tokens2))

		#if intersection is non-empty, compute edge weight and add edge
		if intersect_size > 0:
			#weight = # tokens in common / # of tokens in shorter post
			weight = intersect_size / (len(tokens1) if len(tokens1) < len(tokens2) else len(tokens2))
			return weight
		#no intersection, return False indicating no edge
		return False
	#end __compute_edge_weight


	#given a node (post id), get params for that node - wrapper method for self.post_params
	#if no params for this node exist, return False
	#if skip_default is True, disregard any hardcoded default parameters and return False 
	#(assume that *any* default params in the param set invalidate the rest)
	def __get_params(self, node, skip_default):

		#fetch params from class data - if they exist
		if node in self.post_params:
			params = self.post_params[node]
		else:
			return False	#no params, return False

		#if skip_default is True and these params contain some defaults, throw them out
		if skip_default and True in self.__check_default(params):
			return False 		#params invalidated by default value, return False

		#made it this far, params are good, return them
		return params
	#end __get_params


	#given a set of post parameters, check if they are default settings
	#params come packaged as: a, lbd, k, mu, sigma, n_b
	#return three bools, one each for weibull, lognormal, and branching, true if those params are default hardcodes
	def __check_default(self, params):
		#default param settings, taken from fit_weibull, fit_lognormal, and fit_cascade (respectively)
		#weibull params: a, lambda, k
		DEFAULT_WEIBULL_NONE = [1, 1, 0.15]     #param results if post has NO comments to fit
		DEFAULT_WEIBULL_SINGLE = [1, 2, 0.75]	#param result if post has ONE comment and other fit methods fail
												#for other fit fails, set a = number of comments (index 0)
		#lognormal: mu, sigma
		DEFAULT_LOGNORMAL = [0, 1.5]    #param results if post has no comment replies to fit
		#branching factor
		DEFAULT_BRANCHING = 0.05        #default branching factor n_b if post has no comments, or post comments have no replies

		weibull_params, lognorm_params, n_b = self.__unpack_params(params)

		#check weibull
		if weibull_params == DEFAULT_WEIBULL_NONE or weibull_params == DEFAULT_WEIBULL_SINGLE or (weibull_params[1] == DEFAULT_WEIBULL_SINGLE[1] and weibull_params[2] == DEFAULT_WEIBULL_SINGLE[2]):
			weibull = True
		else:
			weibull = False

		return weibull, lognorm_params == DEFAULT_LOGNORMAL, n_b == DEFAULT_BRANCHING
	#end __check_default


    #helper function to unpack parameters as given to separate items
	def __unpack_params(self, params):
		weibull_params = params[:3]
		lognorm_params = params[3:5]
		n_b = params[5]

		return weibull_params, lognorm_params, n_b
	#end __unpack_params


	#given set of posts and fitted params, build four dictionaries
	#	post -> fitted params
	#	post-> set of tokens
	#	user->set of posts
	#	tokens->set of posts
	def __init_dictionaries(self, posts, params):
		self.post_params = {}					#dictionary of post-ids -> params
		self.post_tokens = defaultdict(set) 	#dictionary of post-ids -> set tokens
		self.tokens = defaultdict(set) 			#dictionary of tokens -> set of posts
		self.users = defaultdict(set)			#user->set of post-ids

		for post_id, post in posts.items():
			post_words = self.__extract_tokens(post)	#get tokens for this post

			self.post_tokens[post_id] = post_words		#save tokens to dict
			self.users[post['author_h']].add(post_id)	#add this post to user set

			self.post_params[post_id] = params[post_id]	#save post's params

			for word in post_words:
				self.tokens[word].add(post_id)		#add post to each token set
	#end __init_dictionaries


	#given a post, extract words by tokenizing and normalizing (no limitization for now)
	#removes all leading/trailing punctuation
	#converts list to set to remove duplicates (if we want that?)
	def __extract_tokens(self, post):
		punctuations = list(string.punctuation)		#get list of punctuation characters
		punctuations.append('—')	#kill these too
		
		title = post['title_m']		#grab post title as new string		
		tokens = [word.lower() for word in title.split()]	#tokenize and normalize (to lower)		
		tokens = [word.strip("".join(punctuations)) for word in tokens]		#strip trailing and leading punctuation
		tokens = [word for word in tokens if word != '' and word not in punctuations and word != '']		#remove punctuation-only tokens and empty strings

		return set(tokens)		#convert to set before returning
	#end __extract_tokens


	#viz the graph, so I can see what's happening
	def viz_graph(self, filename):
		print("Visualizing graph...")
		pos = graphviz_layout(self.graph, prog='dot')
		nx.draw(self.graph, pos, with_labels=False, arrows=False, node_size=15)
		plt.savefig(filename)
		print("Graph visual saved to", filename)
	#end viz_graph

#end ParamGraph