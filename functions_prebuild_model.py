#helper functions for prebuild_model (hide the details)

import cascade_manip

import string
from collections import defaultdict
import itertools

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
def load_subreddit_cascades(subreddit, domain):
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
#duplicate of method in ParamGraph, but we don't want to break class boundaries
def extract_tokens(post):
	punctuations = list(string.punctuation)		#get list of punctuation characters
	punctuations.append('—')	#kill these too
	
	title = post['title_m']		#grab post title as new string		
	tokens = [word.lower() for word in title.split()]	#tokenize and normalize (to lower)		
	tokens = [word.strip("".join(punctuations)) for word in tokens]		#strip trailing and leading punctuation
	tokens = [word for word in tokens if word != '' and word not in punctuations and word != '']		#remove punctuation-only tokens and empty strings

	return set(tokens)		#convert to set before returning
#end extract_tokens

#given set of posts and fitted params, build two helpful dictionaries
#	user->set of posts
#	tokens->set of posts
#use original post ids, translate to numeric in build_graph
def graph_dictionaries(posts):
	tokens = defaultdict(set) 			#dictionary of tokens -> set of posts
	users = defaultdict(set)			#user->set of post-ids

	for post_id, post in posts.items():
		post_words = post['tokens']		#fetch tokens for this post

		users[post['user']].add(post_id)		#add this post to user set

		for word in post_words:
			tokens[word].add(post_id)		#add post to each token set

	return tokens, users
#end graph_dictionaries

#given token sets from two posts, compute weight of edge between them
def compute_edge_weight(tokens1, tokens2):
	#get count of words shared by these posts
	intersect_size = len(tokens1.intersection(tokens2))

	#if intersection is non-empty, compute edge weight and add edge
	if intersect_size > 0:
		#weight = # tokens in common / # of tokens in shorter post
		weight = intersect_size / (len(tokens1) if len(tokens1) < len(tokens2) else len(tokens2))
		return weight
	#no intersection, return False indicating no edge
	return False
#end compute_edge_weight

#given a set of processed posts, "build" the post parameter graph
#but don't actually build it, just get an edgelist
#return graph as dictionary where edge (node1, node2) -> weight
def build_graph(posts):

	print("Building param graph for", len(posts), "posts")

	#build user->posts, and tokens->posts dictionaries (already have post->tokens)
	tokens, users = graph_dictionaries(posts)
	print("   Found", len(users), "users")
	print("   Found", len(tokens), "tokens")

	#build the multi-graph
	#	one node for each post
	#	edge of weight=1 connecting posts by the same user
	#	edge of weight=(# shared)/(# in shortest title) between posts with common words
	#store graph as edge-list dictionary, where (node, node) edge -> weight
	graph = {}
	nodes = set()

	#add all as isolated nodes to start, to make sure they're all represented
	#hmmm... not really sure how to handle isolated nodes in edgelists, so let's just ignore them and hope they aren't a problem!

	#add edges with weight=1 between posts by the same user
	for user, user_posts in users.items():
		post_pairs = list(itertools.combinations(user_posts, 2))		#get all post pairs from this user

		for post_pair in post_pairs:
			graph[(posts[post_pair[0]]['id'], posts[post_pair[1]]['id'])] = 1.0
			nodes.add(posts[post_pair[0]]['id'])
			nodes.add(posts[post_pair[1]]['id'])

	#add edges with weight=(# shared)/(# in shortest title) betwee posts sharing tokens
	#loop only pairs that we know have tokens in common - FASTER
	for token, token_posts in tokens.items():
		post_pairs = list(itertools.combinations(token_posts, 2))		#get all post pairs for this token
		for post_pair in post_pairs:
			#fetch numeric ids for these posts
			node1 = posts[post_pair[0]]['id']
			node2 = posts[post_pair[1]]['id']

			#compute edge weight based on post token sets
			weight = compute_edge_weight(posts[post_pair[0]]['tokens'], posts[post_pair[1]]['tokens'])

			#if valid weight, add edge
			if weight != False:
				#already a user edge between them? just increase the weight
				#want impact of both common user and common tokens (sure)
				#check both edge orientations to be sure
				if (node1, node2) in graph:
					graph[(node1, node2)] += weight
				elif (node2, node1) in graph:
					graph[(node2, node1)] += weight
				#otherwise, new edge with exactly this weight
				else:
					graph[(node1, node2)] = weight		#add edge
					nodes.add(node1)
					nodes.add(node2)

	#handle isolated/missing nodes - return a list of them, code into edgelist during output
	missing_nodes = [value['id'] for key, value in posts.items() if value['id'] not in nodes]

	print("Finished graph has", len(nodes) + len(missing_nodes), "nodes (" + str(len(missing_nodes)), "isolated) and", len(graph), "edges")	

	return graph, missing_nodes
#end build_graph