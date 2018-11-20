#methods for building and performing lookups from the param graph
#start with a bipartite graph of users and words, with edges labelled with parameters
#perform a bipartite projection to get a graph of parameter nodes

#a new post introduces new nodes to this graph, but with undefined parameters
#use PageRank to get these missing parameters

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
		self.graph = None			#line/edge graph of users and words (generated based on bipartite grpah)
		self.post_ids = None 		#set of post_ids represented by this object's graph
		self.post_node_dict = {}	#dictionary of post-id->list of post nodes (created in add_post, used by infer_params)
		self.users = None			#user->word->params dict
		self.tokens = None			#word->users dict (no params)
		self.model = None 			#node2vec embedding model

	#end __init__


	#given a set of posts and corresponding fitted parameters, build the parameter graph
	def build_graph(self, posts, params):
		#make sure we have same number of posts and paramters
		if len(posts) != len(params):
			print("Post and paramter sets are not the same size - skipping graph build")
			return
		print("Building param graph for", len(posts), "posts")

		self.post_ids = set(posts.keys())

		#build user->word->params dictionary, and set of all tokens
		self.__get_users_and_tokens(posts, params)
		print("   Found", len(self.users), "users")
		print("   Found", len(self.tokens), "tokens")

		#build the line/edge graph
		#one node for each user-word pair (no matter how many times a user used that word)
		#identify nodes as <user>--<word>
		#edge between nodes with either a user or a word in common
		self.graph = nx.Graph()

		multi_param_nodes_count = 0		#count number of nodes with multiple parameter sets
		multi_param_words = set()		#set of words that have multiple params for any single user

		#add edges between words used by the same user
		for user, words in self.users.items():			
			word_pairs = list(itertools.combinations(words, 2))		#get all 2-word combos from this user

			if DISPLAY:
				print("\nuser:\t", user)
				print("   words:\n    ", list(words.keys()))
				print("   word pairs:\n    ", word_pairs)

			for word_pair in word_pairs:
				self.graph.add_edge(self.__node_name(user, word_pair[0]), self.__node_name(user, word_pair[1]))

			for word, params in words.items():
				if len(params) > 1:
					multi_param_nodes_count += 1
					multi_param_words.add(word)

		multi_user_words = set()	#set of words used by more than one user

		#add edges between users that used the same word
		for word, users in self.tokens.items():
			user_pairs = list(itertools.combinations(users, 2))		#get all 2-user combos for this word

			if DISPLAY:
				print("\nword:\t", word)
				print("   users:\n    ", users)
				print("   user pairs:\n    ", user_pairs)

			for user_pair in user_pairs:
				self.graph.add_edge(self.__node_name(user_pair[0], word), self.__node_name(user_pair[1], word))

			if len(users) > 1:
				multi_user_words.add(word)

		print("Finished graph has", self.graph.number_of_nodes(), "nodes and", self.graph.size(), "edges")
		if DISPLAY:
			print("  ", multi_param_nodes_count, "nodes have multi-params, labelled by", len(multi_param_words), "tokens")
			print("  ", len(multi_user_words), "tokens used by more than one user")
	#end build_graph


	#given a single post object (sorry, on you to loop), add relevant nodes to graph
	#this WILL change the class's graph, not make a temporary copy
	def add_post(self, post, params=None):
		#verify that we have a graph
		if self.graph == None:
			print("Must build graph before inferring parameters")
			return False

		#verify that post not already represented in graph
		if post['id_h'] in self.post_ids:
			print("Post already represented in graph - no changes to graph")
			return False

		#tokenize this post, grab post user
		tokens = self.__extract_tokens(post)
		user = post['author_h']

		#edge case check: if we've never seen this user, and ALL the tokens are unfamiliar... bad day
		if user not in self.users and len([word for word in tokens if word in self.tokens]) == 0:
			print("No way to connect this post to graph!")
			return ["disconnect"] + list(tokens)

		#grab list of any words used previously by this user (not including those used now)
		prev_tokens = set(self.users[user].keys()) if user in self.users else set()
		unique_prev_tokens = prev_tokens - tokens

		if DISPLAY:
			print("Incorporating post into existing graph...")
			print("   Existing graph has", self.graph.number_of_nodes(), "nodes and", self.graph.size(), "edges")
			print("   user:", user, "\n   tokens:", tokens)
			print("   previous tokens:", prev_tokens, "\n   unique prev:", unique_prev_tokens)

		#add nodes and connecting edges for this post to the graph (or rather, a copy of the graph)

		#add word-edges between words of this post, and between new/old word pairs
		word_pairs = list(itertools.combinations(tokens, 2))	#pairs of words from new post
		combo_pairs = list(itertools.product(tokens, unique_prev_tokens))		#pairs of new word + old word by same user
		for word_pair in itertools.chain(word_pairs, combo_pairs):
			self.graph.add_edge(self.__node_name(user, word_pair[0]), self.__node_name(user, word_pair[1]))

		#add edges between this post's user and other users that used the same word
		for word in tokens:
			#get list of other users for current word (exclude the current user)
			word_users = [word_user for word_user in self.tokens[word] if word_user != user]	

			for word_user in word_users:
				self.graph.add_edge(self.__node_name(word_user, word), self.__node_name(user, word))

		if DISPLAY:
			print("   Updated graph has", self.graph.number_of_nodes(), "nodes and", self.graph.size(), "edges")

		#get list of nodes representing this post (not necessarily all new, bad name)
		self.post_node_dict[post['id_h']] = [self.__node_name(user, word) for word in tokens]

		#add this post to graph tracking
		self.post_ids.add(post['id_h'])
		#tokens: add user to list
		#users: add token to dictionary, 
		for token in tokens:
			self.tokens[token].add(user)
			if params != None:
				self.users[user][token].append(params)
			elif token not in self.users[user]:
				self.users[user][token]

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
		node2vec = Node2Vec(self.graph, dimensions=16, walk_length=10, num_walks=200, workers=2, quiet=False)	#example uses 64 dimensions and walk_length 10, let's go smaller

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
	#if skip_default is True, disregard any hardcoded default parameters in averages and most-similar selections
	def infer_params(self, post, mode, skip_default = True):		
		#verify that post already represented in graph
		if post['id_h'] not in self.post_ids or post['id_h'] not in self.post_node_dict:
			#no params? node must not be connected to graph - use average of subreddit params, since that's the best we can do
			print("Post not represented in graph - cannot infer parameters - using graph average")
			return self.__avg_params(skip_default)

		#verify that we have a model
		if self.model == None:
			print("No fitted Node2Vec model - cannot infer parameters")
			return False

		#verify valid mode
		if mode != 'average' and mode != 'weighted' and mode != 'max':
			print("Invalid mode, must be one of \"average\", \"weighted\" or \"max\"")
			return False

		print("Inferring parameters for post", post['id_h'])
		
		#infer parameters using fitted node2vec model

		#pull post nodes
		post_nodes = self.post_node_dict[post['id_h']]

		#init average for 'average' and 'weighted' modes
		avg_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		divisor = 0.0	
		#other tracking for 'max' mode
		max_similarity = None
		max_match_node = None
		max_params = None

		#init params and associated
		node_params = None
		similarity = None
		match_node = None

		#look for most similar nodes to each post node
		for node in post_nodes:

			#if already have params for this node, use those - as long as they don't violate the current skip_default setting
			node_params = self.__get_params(node, skip_default)
			if node_params != False and ((skip_default == True and True not in self.__check_default(node_params)) or skip_default == False):
				if DISPLAY:
					print(node, ": param lookup", node_params)
				similarity = 1
				match_node = node

			#otherwise, infer based on most similar node in graph with defined params
			else:
				#pull params from best match with defined params				
				top_count = 0
				node_params = None 		#reset for loop check

				#loop until we get some params
				while node_params == None:
					top_count += 10		#increase, in case we had to go further (fetch batches of 10)
					#get most similar nodes
					#returns list of tuples (node, similarity), sorted by similarity
					similar = self.model.wv.most_similar(node, topn=top_count)
					#find best match with defined parameters (these are sorted by similarity)
					for match in similar:
						match_params = self.__get_params(match[0], skip_default)
						#if skipping default and these params are defaults... skip them!
						if match_params != False and skip_default == True and True in self.__check_default(match_params):
							continue
						elif match_params != False:
							node_params = match_params
							similarity = match[1]
							match_node = match[0]
							break
				if DISPLAY:
					print(node, ": param infer from", match_node, "with similarity", similarity, "\n   ", node_params)

			#combine all node params together, depending on mode
			#'max' mode
			if mode == 'max' and (max_similarity == None or similarity > max_similarity):
				max_similarity = similarity
				max_match_node = match_node
				max_params = node_params

			#'average' or 'weighted' modes
			else:
				for i in range(6):
					#standard average
					if mode == 'average':
						avg_params[i] += node_params[i]
						divisor += 1
					#weighted average
					elif mode == 'weighted':
						avg_params[i] += node_params[i] * similarity
						divisor += similarity

		#processed all node, finish and return
		if mode == 'max':
			return max_params
		else:	#weighted or average
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


	#given a node (user-word pair), get a single set of params
	#if no params for this node exist, return False
	#if skip_default is True, disregard any hardcoded default parameters in average 
	#(and assume that any default params in the set invalidate the rest)
	def __get_params(self, node, skip_default):
		#unpack the node into user and word
		user, word = self.__unpack_node(node)

		#fetch params from class data - if they exist
		if user in self.users and word in self.users[user]:
			params = self.users[user][word]
		else:
			return False

		#average all parameter sets together
		avg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		divisor = 0
		for item in params:
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
	#end __get_params


	#given user and word, get corresponding node name 
	#(tiny helper method, but makes it easy to change the naming scheme)
	def __node_name(self, user, word):
		return user + "**" + word
	#end __node_name


	#given node name, extract user and word the node represents
	#(tiny helper method, but perform reverse of __node_name)
	def __unpack_node(self, name):
		return name.split("**")		#return user, then word
	#end __unpack_node


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


	#given set of posts and fitted params, and build user_id->word->list of param sets dictionary
	#also build complete set of tokens, seen across all posts
	def __get_users_and_tokens(self, posts, params):
		self.users = defaultdict(ddl)		#user->word->params (hacky method for nested dict to allow for pickle)
		self.tokens = defaultdict(set)		#word->users

		for post_id, post in posts.items():
			post_words = self.__extract_tokens(post)		#get tokens for this post
			for word in post_words:
				self.users[post['author_h']][word].append(params[post_id])
				self.tokens[word].add(post['author_h'])
	#end __get_users_and_tokens


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


def ddl():
	return defaultdict(list)