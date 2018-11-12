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
		self.users = None			#user->word->params dict
		self.tokens = None			#word->users dict (no params)
		self.static_rank = None 	#pagerank results for static (initial) graph

	#end __init__


	#given a set of posts and corresponding fitted parameters, build the parameter graph
	def build_graph(self, posts, params):
		#make sure we have same number of posts and paramters
		if len(posts) != len(params):
			print("Post and paramter sets are not the same size - skipping tensor build")
			return
		print("\nBuilding param graph for", len(posts), "posts")

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
		print("  ", multi_param_nodes_count, "nodes have multi-params, labelled by", len(multi_param_words), "tokens")
		print("  ", len(multi_user_words), "tokens used by more than one user")
	#end build_graph


	#given a single post object (not a part of current graph or pagerank), infer parameters
	def infer_params(self, post):
		#verify that we have a graph and pre-computed pagerank
		if self.graph == None or self.static_rank == None:
			print("Must build graph and compute pagerank before inferring parameters")
			return False

		#verify that post not already represented in graph
		if post['id_h'] in self.post_ids:
			print("Post already represented in graph - no parameters to infer")
			return False

		print("\nInferring parameters for post", post['id_h'])

		#tokenize this post, grab post user
		tokens = self.__extract_tokens(post)
		user = post['author_h']

		#grab list of any words used previously by this user (not including those used now)
		prev_tokens = set(self.users[user].keys())
		unique_prev_tokens = prev_tokens - tokens

		if DISPLAY:
			print("   user:", user, "\n   tokens:", tokens)
			print("   previous tokens:", prev_tokens, "\n   unique prev:", unique_prev_tokens)

		#add nodes and connecting edges for this post to the graph (or rather, a copy of the graph)
		print("   Incorporating new post into existing graph...")
		temp_graph = self.graph.copy()

		#add word-edges between words of this post, and between new/old word pairs
		word_pairs = list(itertools.combinations(tokens, 2))	#pairs of words from new post
		combo_pairs = list(itertools.product(tokens, unique_prev_tokens))		#pairs of new word + old word
		for word_pair in itertools.chain(word_pairs, combo_pairs):
			temp_graph.add_edge(self.__node_name(user, word_pair[0]), self.__node_name(user, word_pair[1]))

		#add edges between this post's user and other users that used the same word
		for word in tokens:
			#get list of other users for current word (exclude the current user)
			word_users = [word_user for word_user in self.tokens[word] if word_user != user]	

			for word_user in word_users:
				temp_graph.add_edge(self.__node_name(word_user, word), self.__node_name(user, word))

		print("   Updated graph has", temp_graph.number_of_nodes(), "nodes and", temp_graph.size(), "edges")

		#now that we've added the new-post nodes to the graph, infer parameters using node2vec


	#end infer_params

	#given a node (user-word pair), get a single set of params
	def __get_params(self, node):
		#unpack the node into user and word
		user, word = self.__unpack_node(node)

		#fetch params from class data
		params = self.users[user][word]

		#average all parameter sets together
		avg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		for item in params:
			for i in range(6):
				avg[i] += item[i]
		avg = [item / len(params) for item in avg]

		return avg
	#end __get_params


	#given user and word, get corresponding node name 
	#(tiny helper method, but makes it easy to change the naming scheme)
	def __node_name(self, user, word):
		return user + "--" + word
	#end __node_name


	#given node name, extract user and word the node represents
	#(tiny helper method, but perform reverse of __node_name)
	def __unpack_node(self, name):
		return name.split("--")		#return user, then word
	#end __unpack_node


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

#end ParamTensor


def ddl():
	return defaultdict(list)