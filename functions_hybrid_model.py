import json
from collections import defaultdict
import string
import random
import itertools
import os

MAX_TOKEN_MATCH_POSTS = 5		#maximum number of token-matching posts to add to graph when inferring params for a
								#post by an unseen user with all new tokens

#filepaths of input files
subreddits_filepath = "model_files/subreddits.pkl"		#dictionary of subreddit -> domain code
posts_filepath = "model_files/posts/%s_posts.pkl"			#processed post data for each post, one file per subreddit
														#each post maps original post id to numeric id, set of tokens, and user id
params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per subreddit
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)
graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this subreddit


DISPLAY = False		#toggle debug print statements

POST_PREFIX = "t3_"
COMMENT_PREFIX = "t1_"

#creates submission json file
#domain - simulation domain, must be one of 'crypto', 'cyber', or 'CVE'
#events - list of simulation events, each a dictionary with the following fields: 
'''
	"rootID": id of root post for cascade
	"actionType": 'post' or 'comment'
	"parentID": id of parent (comment or post), null/self if this object is post 
	"nodeTime": timestamp of event
	"nodeUserID": id of user posting this event
	"nodeID": id of post/comment
	"communityID": id of subreddit			???
'''
#outfile - location of output json file
#generates file of the following form:
"""
	{"team"     : 'usc',
	 "scenario" : 2,
	 "domain"   : domain,
	 "platform" : reddit,
	 "data":[
	 JSON_datapoint,
	 JSON_datapoint,
	 :
	 :
	 :
	 JSON_datapoint,
	 JSON_datapoint]
	 }
"""
def reddit_sim_to_json(domain, events, outfile):
    print("Saving results to", outfile + "...")

	#header values
    team_name = 'usc'
    scenario = 2
    platform = 'reddit'

    if domain == 'cve':
    	domain = "CVE"

    #crypto, cyber, or CVE domain
    if domain != 'crypto' and domain != 'cyber' and domain != 'CVE':
        print("Invalid domain")
        return False        
        
    #write to json
    submission = {'team'     : team_name,
                  'scenario' : scenario,
                  'domain'   : domain,
                  'platform' : platform,
                  'data'     : events}
            
    with open(outfile,'w') as f:
        json.dump(submission, f, indent=3, sort_keys=False)

    print("Done")

    return
#end reddit_sim_to_json


#given a filename of cascade seed test data, load all seed posts
#lucky us, the input data is multiple json objects in the same file - not a list of them
#at least they're one per line!
def load_reddit_seeds(filename):
    data = []
        
    with open(filename,'r') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)

    print("Read", len(data), "events")

    return data
#end load_reddit_seeds


#given a sim_tree return (tree structure of comment times), and original seed post, convert to desired dictionary structure
#{"rootID": "t3_-dt8ErhaKuULHekBf_ke3A", "communityID": "t5_2s3qj", "actionType": "post", "parentID": "t3_-dt8ErhaKuULHekBf_ke3A", "nodeTime": "1421194091", "nodeUserID": "XHa80wD0LQJNlMcrUD32pQ", "nodeID": "t3_-dt8ErhaKuULHekBf_ke3A"}
'''
	"rootID": id of root post for cascade
	"actionType": 'post' or 'comment'
	"parentID": id of parent (comment or post), null/self if this object is post 
	"nodeTime": timestamp of event
	"nodeUserID": id of user posting this event
	"nodeID": id of post/comment
	"communityID": id of subreddit			???
'''
def build_cascade_events(root, post, user_ids):
	#get start of event structure by converting sim tree to events list
	events = tree_to_events(root, post['id_h'], post['created_utc']) 

	#set the rest of the fields, the same for all events in this cascade
	for event in events:
		event['rootID'] = POST_PREFIX + post['id_h']
		event['communityID'] = post['subreddit_id']
		#assign user to each event - random from known users for now
		event['nodeUserID'] = post['author_h'] if event['nodeID'] == POST_PREFIX + post['id_h'] else random.choice(user_ids)

	if DISPLAY:
		print(post['id_h'], post['author_h'], post['created_utc'])
		print("")
		for event in events:
			print(event)
		print("")

	return events
#end build_cascade_events


NEXT_ID = 0		#global counter of next id to assign to simulated comments

#given root of sim tree, return as list of dictionary events, with correct parent pointers
def tree_to_events(root, seedID, seedTime):
	global NEXT_ID

	visited = set()    #set of visited nodes
	stack = [(root, None)]     #node processing stack

	events = []		#list of events, where each event is a dictionary

	while len(stack) != 0:
		curr, parent = stack.pop()  #get last node added to stack

		#create event dictionary for this comment, add to events list
		new_event = {'nodeTime': str(int(curr['time'] * 60) + seedTime)}		#convert offset to seconds, add to seed time, convert entire time to string
		#set nodeid if comment
		if parent != None:
			new_event['nodeID'] = COMMENT_PREFIX + "comment" + str(NEXT_ID)		#include comment prefix
			NEXT_ID += 1
		#nodeID = seed post ID if at root
		else:
			new_event['nodeID'] = POST_PREFIX + seedID
		#set parent of this event
		new_event['parentID'] = POST_PREFIX + seedID if parent == None else parent
		#set actionType of this event
		new_event['actionType'] = 'post' if parent == None else 'comment'
		#add new event to list
		events.append(new_event)

		visited.add(curr['id'])		#add this node to visited list
		#append children in reverse time order so final output is sorted (not that it matters)
		stack.extend([(child, new_event['nodeID']) for child in curr['children']][::-1])   

	return events
#end tree_to_events


#given a post, extract words by tokenizing and normalizing (no limitization for now)
#removes all leading/trailing punctuation
#converts list to set to remove duplicates (if we want that?)
#(duplicate of method in functions_prebuild_model.py, but copied here for convenience)
def extract_tokens(post):
	punctuations = list(string.punctuation)		#get list of punctuation characters
	punctuations.append('—')	#kill these too
	
	title = post['title_m']		#grab post title as new string		
	tokens = [word.lower() for word in title.split()]	#tokenize and normalize (to lower)		
	tokens = [word.strip("".join(punctuations)) for word in tokens]		#strip trailing and leading punctuation
	tokens = [word for word in tokens if word != '' and word not in punctuations and word != '']		#remove punctuation-only tokens and empty strings

	return set(tokens)		#convert to set before returning
#end extract_tokens


#given token sets from two posts, compute weight of edge between them
#(duplicate from functions_prebuild_model.py, copied for convenience)
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


#given set of posts and fitted params, build two helpful dictionaries
#	user->set of posts
#	tokens->set of posts
#use original post ids, translate to numeric in build_graph
#(duplicate of same method in functions_prebuild_model.py, but copied here for convenience)
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


TOKENS = None		#global dictionaries for post graph, used in add_post_edges
USERS = None
GRAPH_SUBREDDIT = None 		#track subreddit used for current dictionaries (because need to clear when change)

def add_post_edges(graph, isolated_nodes, graph_posts, new_post, new_post_numeric_id, subreddit):
	global TOKENS, USERS, GRAPH_SUBREDDIT

	#if don't have token/user dictionaries yet, get them now
	if TOKENS == None or USERS == None or subreddit != GRAPH_SUBREDDIT:
		TOKENS, USERS = graph_dictionaries(graph_posts)
		GRAPH_SUBREDDIT = subreddit

	#set isolated flag to make sure this node gets connected - otherwise need to handle as isolated
	isolated = True

	#tokenize this post, grab post user
	new_tokens = extract_tokens(new_post)
	new_user = new_post['author_h']

	#add connecting edges for this post to the graph, token edges first

	#add edges with scaled weight between posts with same tokens as this one
	for token in new_tokens:
		for other_post in TOKENS[token]:

			#already an edge (and therefore a token edge) in the graph? skip
			if (new_post_numeric_id, graph_posts[other_post]['id']) in graph or (graph_posts[other_post]['id'], new_post_numeric_id) in graph:
				continue

			#compute edge weight based on post token sets
			weight = compute_edge_weight(graph_posts[other_post]['tokens'], new_tokens)

			#if valid weight, add edge
			if weight != False:
				graph[(new_post_numeric_id, graph_posts[other_post]['id'])] = weight		#add edge
				isolated = False

	#add edges of weight=1 between posts also by this user
	for prev_post in USERS[new_user]:

		#already an edge (token edge) between these users? just increase the weight
		#want impact of both common user and common tokens
		#check both edge orientations to be sure
		if (new_post_numeric_id, graph_posts[prev_post]['id']) in graph:
			graph[(new_post_numeric_id, graph_posts[prev_post]['id'])] += 1.0
		elif (graph_posts[prev_post]['id'], new_post_numeric_id) in graph:
			graph[(graph_posts[prev_post]['id'], new_post_numeric_id)] += 1.0
		#new edge, just set weight
		else:
			graph[(new_post_numeric_id, graph_posts[prev_post]['id'])] = 1.0
			isolated = False

	#cve only: edge of weight 1 between posts in same subreddit
	if 'sub' in new_post:
		for prev_post_id, prev_post in graph_posts:
			#same subreddit, add edge
			if prev_post['sub'] == new_post['sub']:
				#already an edge between these posts? just increase the weight
				#check both edge orientations to be sure
				if (new_post_numeric_id, prev_post['id']) in graph:
					graph[(new_post_numeric_id, prev_post['id'])] += 1.0
				elif (prev_post['id'], new_post_numeric_id) in graph:
					graph[(prev_post['id'], new_post_numeric_id)] += 1.0
				#new edge, just set weight
				else:
					graph[(new_post_numeric_id, prev_post['id'])] = 1.0
					isolated = False

	#if node is isolated (no edges added), include in isolated_nodes list
	if isolated:
		isolated_nodes.append(new_post_numeric_id)

	#add this post to graph tracking, so we can connect it to other seed posts
	graph_posts[new_post['id_h']] = {'tokens': new_tokens, 'id': new_post_numeric_id}
	USERS[new_user].add(new_post['id_h'])
	for token in new_tokens:
		TOKENS[token].add(new_post['id_h'])

	return graph, isolated_nodes, graph_posts
#end add_post_edges


#load fitted/inferred params from file
def load_params(filename, posts, inferred=False, quality=False):
	#read all lines of file
	with open(filename, 'r') as f:
		lines = f.readlines()

	#if reading inferred file, skip first line
	if inferred:
		lines.pop(0)

	all_params = {}
	all_quality = {}

	#process each line, extract params
	for line in lines:
		values = line.split()
		post_id = int(values[0])	#get original id for this post
		params = []
		for i in range(1, 7):
			params.append(float(values[i]))
		if inferred == False:
			all_quality[post_id] = float(values[7])
		all_params[post_id] = params

	if quality:
		return all_params, all_quality
	return all_params
#end load_params


#given sampled graph posts, build sampled params file
def get_sampled_params(posts, in_filename, out_filename):

	#build list of post ids included in graph
	included_ids = [value['id'] for key, value in posts.items()]
	print("Filtering params to", len(included_ids), "sampled posts")

	#only save the params we actually need
	post_params = {}

	#read file, one line at at a time
	with open(in_filename, "r") as ins:
		for line in ins:
			values = line.split()
			post_id = int(values[0])	#get original id for this post
			if post_id in included_ids:
				params = []
				for i in range(1, 8):
					params.append(float(values[i]))
				post_params[post_id] = params
	#dump filtered params to file
	with open(out_filename, "w") as f: 
		for post_id, params in post_params.items():
			f.write(str(post_id) + " ")		#write numeric post id
			for i in range(len(params)):
				f.write((' ' if i > 0 else '') + str(params[i]))
			f.write("\n")
	print("Saved sampled params to", out_filename)
#end get_sampled_params


#given complete set of posts/params for current subreddit, and seed posts,
#sample down to reach the target graph size (hopefully feasible for inference)
def user_sample_graph(raw_sub_posts, seeds, max_nodes, subreddit):
	#is this a cve run? if so, set flag
	cve = False
	if subreddit == "cve":
		cve = True

	#set of authors of seed
	authors = set([post['author_h'] for post in seeds])
	print("  ", len(authors), "authors in seed posts")
	#cve: get list of subreddits of seed posts
	if cve:
		subs = set([post['subreddit'] for post in seeds])

	#cve: keep posts with same subreddit or user
	if cve:
		sub_posts = {key: value for key, value in raw_sub_posts.items() if value['user'] in authors or value['sub'] in subs}
		graph_subs = set([post['sub'] for post_id, post in sub_posts.items()])
	#filter the posts: only those by users in the author pool
	else:
		sub_posts = {key: value for key, value in raw_sub_posts.items() if value['user'] in authors}
	graph_authors = set([post['user'] for post_id, post in sub_posts.items()])
	print("   Filtered to", len(sub_posts), "posts by", len(graph_authors), "authors")

	#no more than MAX_GRAPH_POSTS posts, or this will never finish
	if len(sub_posts) > max_nodes:
		print("   Sampling down...")
		#keep at least one post by each author that we have posts for (pick a random one)
		#and keep any seed posts that we have
		keep = set([seed['id_h'] for seed in seeds if seed['id_h'] in sub_posts])
		for author in graph_authors:
			keep.add(random.choice([key for key, value in sub_posts.items() if value['user'] == author]))
		#if cve, keep at least one per sub
		if cve:
			for sub in graph_subs:
				keep.add(random.choice([key for key, value in sub_posts.items() if value['sub'] == sub]))
		#sample down (as far as we can, while maintaining one post per author (and sub if cve))
		if max_nodes - len(keep) > 0:
			keep.update(random.sample([key for key in sub_posts.keys() if key not in keep], max_nodes-len(keep)))
		sub_posts = {key: sub_posts[key] for key in keep}
	#too few? draw more
	if len(sub_posts) < max_nodes:
		print("   Drawing more posts...")
		#add more
		draw_keys = [key for key in raw_sub_posts.keys() if key not in sub_posts.keys()]
		num_draw = max_nodes - len(sub_posts)
		more = random.sample(draw_keys, num_draw)
		sub_posts.update({key: raw_sub_posts[key] for key in more})

	graph_authors = set([post['user'] for post_id, post in sub_posts.items()])	#update graph authors list
	print("   Sampled to", len(sub_posts), "posts by", len(set(graph_authors)), "authors for inference (" +  str(len([author for author in authors if author in graph_authors])), "seed authors)")
	if cve:
		graph_subs = set([post['sub'] for post_id, post in sub_posts.items()])
		print(len(graph_subs), "subreddits in graph (" + str(len([sub for sub in subs if sub in graph_subs])), "seed subs)")

	#can we connect all the seed posts to this sampled version of the graph? check authors and tokens

	#build list of tokens represented in graph (extra work, yes, repeated in graph build)
	graph_tokens = set()
	for post_id, post in sub_posts.items():
		graph_tokens.update(post['tokens'])
	print("  ", len(graph_tokens), "tokens in graph")

	#check to see if all seed posts can be connected to this graph
	for post in seeds:
		#post author in graph, no need for token match
		if post['author_h'] in graph_authors:
			continue
		#if cve and post sub in graph, no need for token match
		if cve and post['subreddit'] in graph_subs:
			continue

		post_tokens = extract_tokens(post)		#get tokens from this post
		#if no token connection, try to draw some more posts to allow for connection
		if len(graph_tokens.intersection(post_tokens)) == 0:
			#can we find a post in our library with some of these tokens? to get the node connected
			matching_posts = {key:value for key, value in raw_sub_posts.items() if len(post_tokens.intersection(value['tokens'])) != 0}
			#if too many matching token posts, sample down
			if len(matching_posts) > MAX_TOKEN_MATCH_POSTS:
				keep = random.sample(matching_posts.keys(), MAX_TOKEN_MATCH_POSTS)
				matching_posts = {key: matching_posts[key] for key in keep}

			#if no matching token/user posts - params will basically be a guess
			if len(matching_posts) == 0:
				print("   Cannot connect seed post to graph - parameter inference compromised")
			#add the matching posts to the graph
			else:
				print("   Adding", len(matching_posts), "token-matching posts to graph")
				sub_posts.update(matching_posts)

	#return sampled posts and params
	return sub_posts
#end user_sample_graph


#given a set of processed posts, "build" the post parameter graph
#but don't actually build it, just get an edgelist
#save edgelist to specified file
#no return, because graph is periodically dumped and not all stored in memory at once
#(near-duplicate of functions_prebuild_model.py for convenience)
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
	#and save graph to file
	with open(filename, "w") as f:
		for edge, weight in edgelist.items():
			f.write("%d %d %f\n" % (edge[0], edge[1], weight))
		for node in isolated_nodes:
			f.write("%d\n" % node)
#end save_graph