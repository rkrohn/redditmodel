import json
from collections import defaultdict
import string
import random



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

def add_post_edges(graph, isolated_nodes, graph_posts, new_post, new_post_numeric_id):
	global TOKENS, USERS

	#if don't have token/user dictionaries yet, get them now
	if TOKENS == None or USERS == None:
		TOKENS, USERS = graph_dictionaries(graph_posts)

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
def load_params(filename, posts, inferred=False):
	#read all lines of file
	with open(filename, 'r') as f:
		lines = f.readlines()

	#if reading inferred file, skip first line
	if inferred:
		lines.pop(0)

	all_params = {}

	#process each line, extract params
	for line in lines:
		values = line.split()
		post_id = int(values[0])	#get original id for this post
		params = []
		for i in range(1, 7):
			params.append(float(values[i]))
		all_params[post_id] = params

	return all_params
#end load_params