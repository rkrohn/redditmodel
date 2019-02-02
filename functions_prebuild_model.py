#helper functions for prebuild_model (hide the details)

import cascade_manip
import cascade_analysis

import string
from collections import defaultdict
import itertools
import os


#loads cascades and comments for subreddit, if not already loaded
def load_cascades(subreddit, domain, cascades, comments):
	if cascades == None or comments == None:
		cascades, comments = load_subreddit_cascades(subreddit, domain, cascades, comments)
	return cascades, comments
#end load_cascades

#given a list of itemsets, and a list of corresponding set labels, 
#build a dictionary where item->list of corresponding labels
def build_domain_dict(itemsets, labels):
	labeled_items = defaultdict(list)		#dictionary for items, item->list of occurrence domains

	for i in range(len(itemsets)):
		for item in itemsets[i]:
			labeled_items[item].append(labels[i])

	#for item, occurrences in labeled_items.items():
	#	print(item, occurrences)

	return labeled_items
#end build_domain_dict

#load all posts and comments for given subreddit, removing any incomplete cascades
def load_subreddit_cascades(subreddit, domain, cascades, comments):
	#load filtered cascades for this subreddit
	filtered_cascades, filtered_comments = cascade_manip.load_filtered_cascades(domain, subreddit)

	#don't exist, filter them now
	if filtered_cascades == False:

		#have we loaded the raw cascades/comments yet? if not, do it now
		#(waiting until now in case we have all the filtered versions and don't need these at all)
		if cascades == None or comments == None:
			#build/load cascades (auto-load as a result, either raw data or cached cascades)
			cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(domain)
			#optional: filter out cascades with any missing elements (posts or comments)
			cascades, comments = cascade_manip.remove_missing(domain, cascades, comments)

		#filter cascades by a particular subreddit
		filtered_cascades = cascade_manip.filter_cascades_by_subreddit(cascades, subreddit)
		#and filter comments to match those posts
		filtered_cascades, filtered_comments = cascade_manip.filter_comments_by_posts(filtered_cascades, comments)
		#save these filtered posts/comments for easier loading later
		cascade_manip.save_cascades(code, filtered_cascades, subreddit)
		cascade_manip.save_comments(code, filtered_comments, subreddit)

	return filtered_cascades, filtered_comments
#end load_subreddit_cascades

#given a post, extract words by tokenizing and normalizing (no limitization for now)
#removes all leading/trailing punctuation
#converts list to set to remove duplicates (if we want that?)
def extract_tokens(post):
	punctuations = list(string.punctuation)		#get list of punctuation characters
	punctuations.append('—')	#kill these too
	
	title = post['title_m']		#grab post title as new string
	if title != None:
		tokens = [word.lower() for word in title.split()]	#tokenize and normalize (to lower)		
		tokens = [word.strip("".join(punctuations)) for word in tokens]		#strip trailing and leading punctuation
		tokens = [word for word in tokens if word != '' and word not in punctuations and word != '']		#remove punctuation-only tokens and empty strings
	else:
		tokens = []

	return set(tokens)		#convert to set before returning
#end extract_tokens

#given token sets from two posts, compute weight of edge between them
def compute_edge_weight(tokens1, tokens2):
	#get count of words shared by these posts
	intersect_size = len(tokens1.intersection(tokens2))

	#if intersection is non-empty, compute edge weight and add edge
	if intersect_size > 0:
		#weight = # tokens in common / # of tokens in shorter post
		weight = intersect_size / (len(tokens1) if len(tokens1) < len(tokens2) else len(tokens2))
		return weight
	#no intersection, return 0 indicating no edge
	return 0
#end compute_edge_weight

#given a set of processed posts, "build" the post parameter graph
#but don't actually build it, just get an edgelist
#save edgelist to specified file
#no return, because graph is periodically dumped and not all stored in memory at once
def build_graph(posts, filename):

	print("Building param graph for", len(posts), "posts")

	#build the multi-graph
	#	one node for each post
	#	edge of weight=1 connecting posts by the same user
	#	edge of weight=(# shared)/(# in shortest title) between posts with common words
	#(but really only one edge, with sum of both weights)
	#store graph as edge-list dictionary, where (node, node) edge -> weight
	graph = {}
	nodes = set()
	edge_count = 0

	#loop all post-pairs and determine weight of edge, if any, between them
	for post_pair in itertools.combinations(posts, 2):		#pair contains ids of two posts
		#fetch numeric ids for these posts
		node1 = posts[post_pair[0]]['id']
		node2 = posts[post_pair[1]]['id']

		#compute edge weight based on post token sets
		weight = compute_edge_weight(posts[post_pair[0]]['tokens'], posts[post_pair[1]]['tokens'])
		if weight <= 0.1:		#minimum token weight threshold, try to keep edge explosion to a minimum
			weight = 0

		#cve only: if posts have same subreddit, add 1 to weight
		if 'sub' in posts[post_pair[0]] and posts[post_pair[0]]['sub'] == posts[post_pair[1]]['sub']:
			weight += 1

		#if posts have same author, add 1 to weight
		if posts[post_pair[0]]['user'] == posts[post_pair[1]]['user']:
			weight += 1

		#if edge weight is nonzero, add edge to graph
		if weight != 0:
			graph[(node1, node2)] = weight 		#add edge
			nodes.add(node1)					#track edges in graph so we can find isolated nodes later
			nodes.add(node2)
			edge_count += 1						#keep count of all edges

			#if added edge, and current edgelist has reached dump level, dump and clear before continuing
			if len(graph) == 25000000:
				save_graph(graph, filename)
				print("   Saved", edge_count, "edges")
				graph = {}

	#handle isolated/missing nodes - return a list of them, code into edgelist during output
	isolated_nodes = [value['id'] for key, value in posts.items() if value['id'] not in nodes]

	print("Finished graph has", len(nodes) + len(isolated_nodes), "nodes (" + str(len(isolated_nodes)), "isolated) and", edge_count, "edges")	

	#dump any remaining edges/isolated nodes to edgelist file (final save)
	save_graph(graph, filename, isolated_nodes)
	print("Saved post-graph to", filename)

#end build_graph


#save graph to txt file
def save_graph(edgelist, filename, isolated_nodes = []):
	#determine file write mode: create new if file doesn't exist, otherwise append to graph in progress
	if os.path.exists(filename):
		mode = 'a'
	else:
		mode = 'w'

	#and save graph to file
	with open(filename, mode) as f:
		for edge, weight in edgelist.items():
			f.write("%d %d %f\n" % (edge[0], edge[1], weight))
		for node in isolated_nodes:
			f.write("%d\n" % node)
#end save_graph