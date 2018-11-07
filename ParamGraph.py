#methods for building and performing lookups from the param graph
#start with a bipartite graph of users and words, with edges labelled with parameters
#perform a bipartite projection to get a graph of parameter nodes

#a new post introduces new nodes to this graph, but with undefined parameters
#use PageRank to get these missing parameters

import string
import numpy
from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


#params are indexed as follows:
#	a 		0
#	lbd 	1
#	k 		2
#	mu 		3
#	sigma	4
#	n_b		5

#global display setting for the class: applies to debug-type output only
DISPLAY = False

class ParamGraph:


	def __init__(self, filename = None):
		#if given filename of cached tensor, load from disk
		if filename != None and file_utils.verify_file(filename):
			load_cached_tensor(filename)
		#given filename, but file doesn't exist - error message, and create a new empty tensor object
		elif filename != None:
			print("No cached tensor to load - creating new object")
			filename = None

		#no filename given, create new tensor object
		if filename == None:
			self.graph = nx.Graph()		#line/edge graph of users and words (generated based on bipartite grpah)
			self.users = None			#user->word->params dict
			self.tokens = None			#word->users dict (no params)

	#end __init__


	#given a set of posts and corresponding fitted parameters, build the parameter graph
	def build_graph(self, posts, params):
		#make sure we have same number of posts and paramters
		if len(posts) != len(params):
			print("Post and paramter sets are not the same size - skipping tensor build")
			return
		print("Building param graph for", len(posts), "posts")

		#build user->word->params dictionary, and set of all tokens
		self.__get_users_and_tokens(posts, params)
		print("   Found", len(self.users), "users")
		print("   Found", len(self.tokens), "tokens")

		#build the line/edge graph
		#one node for each user-word pair (no matter how many times a user used that word)
		#identify nodes as <user>--<word>
		#edge between nodes with either a user or a word in common

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
		print("  ", multi_param_nodes_count, "nodes have multi-params, affected words are", multi_param_words)
		print("  ", len(multi_user_words), "tokens used by more than one user")

	#end build_graph


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
		self.users = defaultdict(lambda: defaultdict(list))		#user->word->params
		self.tokens = defaultdict(set)		#word->users

		for post_id, post in posts.items():
			post_words = self.__extract_tokens(post)		#get tokens for this post
			for word in post_words:
				self.users[post['author_h']][word].append(params[post_id])
				self.tokens[word].add(post['author_h'])
	#end __build_user_dict


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


	#load a tensor from saved pickle
	def load_cached_graph(self):
		print("unpickle some things")
	#end load_cached_tensor


	#dump a pickle to tensor
	def save_graph(self):
		print("pickle some things")
	#end save_tensor

#end ParamTensor