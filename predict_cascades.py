#given a set of seed posts, predict them all and create submission output file
#requires three command line args: domain, input filename of seed posts, and output filename for simulation json results

import file_utils
from newParamGraph import ParamGraph
import cascade_manip
import sim_tree

import sys
import json
from collections import defaultdict
import random
import string


DISPLAY = False		#toggle debug print statements
MAX_GRAPH_POSTS = 50	#maximum number of library posts to include in graph for paramter inference
MAX_TOKEN_MATCH_POSTS = 5		#maximum number of token-matchin posts to add to graph when inferring params for a
								#post by an unseen user with all new tokens

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
        json.dump(submission, f)

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
		event['rootID'] = post['id_h']
		event['communityID'] = post['subreddit_id']
		#assign user to each event - random from known users for now
		event['nodeUserID'] = post['author_h'] if event['nodeID'] == post['id_h'] else random.choice(user_ids)

	if DISPLAY:
		print(post['id_h'], post['author_h'], post['created_utc'])
		print("")
		for event in events:
			print(event)
		print("")

	return events
#end build_cascade_events


NEXT_ID = 0		#global counter of next id to assign to comments

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
			new_event['nodeID'] = "comment" + str(NEXT_ID)
			NEXT_ID += 1
		#nodeID = seed post ID if at root
		else:
			new_event['nodeID'] = seedID
		#set parent of this event
		new_event['parentID'] = seedID if parent == None else parent
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


#main execution begins here
print("")

#verify command line args
if len(sys.argv) != 4:
	print("Incorrect command line arguments\nUsage: python3 predict_cascades.py <domain> <seed filename> <output filename>")
	exit(0)

#extract arguments
domain = sys.argv[1]
infile = sys.argv[2]
outfile = sys.argv[3]

#load post seeds
raw_post_seeds = load_reddit_seeds(infile)

#convert to dictionary of subreddit->list of post objects
post_seeds = defaultdict(list)
for post in raw_post_seeds:
	post_seeds[post['subreddit']].append(post)
print({key : len(post_seeds[key]) for key in post_seeds})

all_events = []		#list for all sim events, across all seed posts
post_counter = 1	#counter of posts to simulate, across all subreddits

#process each subreddit
for subreddit, seeds in post_seeds.items():
	'''
	#TESTING ONLY!!!!
	if subreddit != "Bitcoin":
		continue
	'''

	print("\nProcessing", subreddit, "with", len(seeds), "posts to simulate")

	#load subreddit posts (don't need the comments!)
	raw_sub_posts = cascade_manip.load_filtered_posts(domain, subreddit)
	#load subreddit parameters
	raw_sub_params = cascade_manip.load_cascade_params(domain, subreddit)

	'''
	#verifying the graph build
	#build graph with all loaded posts
	print("\nBuilding full graph")
	full_graph = ParamGraph()
	full_graph.build_graph(raw_sub_posts, raw_sub_params)
	print("")
	'''

	#filter posts - TESTING ONLY!!!!!!!! - if you didn't load all the params
	'''
	sub_posts = {post_id : post for post_id, post in sub_posts.items() if post_id in sub_params}
	print("Filtered to", len(sub_posts), "posts with fitted parameters")
	'''

	#verify loads
	if raw_sub_posts == False or raw_sub_params == False:
		print("Load failed - exiting")
		exit(0)

	#remove seed posts from fitted list - no cheating
	for post in seeds:
		if post['id_h'] in raw_sub_posts:
			raw_sub_posts.pop(post['id_h'])
			raw_sub_params.pop(post['id_h'])
	print(len(raw_sub_posts), "remaining after removing seed posts")

	'''
	#verifying the graph build
	#build graph again, this time with filtered, then adding in the posts
	print("\nBuilding additive graph")
	add_graph = ParamGraph()
	add_graph.build_graph(raw_sub_posts, raw_sub_params)
	for post in seeds:
		add_graph.add_post(post)
	print("Finished graph has", add_graph.graph.number_of_nodes(), "nodes and", add_graph.graph.size(), "edges\n")
	'''

	#get list of posting users to pull comment user ids from
	#list, not set, so more frequent posters are more likely to come up
	user_ids = [post['author_h'] for post_id, post in raw_sub_posts.items() if post['author_h'] != "[Deleted]" ]

	#print dates of posts
	#find earliest date of seed posts
	earliest = -1
	earliest_post = None
	latest = -1
	latest_post = None
	for post in seeds:
		if post['created_utc'] < earliest or earliest == -1:
			earliest = post['created_utc']
			earliest_post = post
		if post['created_utc'] > latest or latest == -1:
			latest = post['created_utc']
			latest_post = post


	print("earliest post", earliest, earliest_post['created_date'])
	print("latest post", latest, latest_post['created_date'])	

	#list of authors of seed posts
	authors = [post['author_h'] for post in seeds]
	print(len(authors), "authors in seed posts")

	#filter the posts: only those ocurring between the sim post timestamps, and created by users in the author pool
	sub_posts = {key: value for key, value in raw_sub_posts.items() if value['created_utc'] >= earliest and value['created_utc'] <= latest and value['author_h'] in authors}
	sub_params = {key: value for key, value in raw_sub_params.items() if key in sub_posts}
	print("Filtered to", len(sub_posts), "posts")

	#no more than MAX_GRAPH_POSTS posts, or this will never finish
	if len(sub_posts) > MAX_GRAPH_POSTS:
		#sample down
		keep = random.sample(sub_posts.keys(), MAX_GRAPH_POSTS)
		sub_posts = {key: sub_posts[key] for key in keep}
		sub_params = {key: sub_params[key] for key in keep}
	#too few? draw more
	if len(sub_posts) < MAX_GRAPH_POSTS:
		#add more
		draw_keys = [key for key in raw_sub_posts.keys() if key not in sub_posts.keys()]
		num_draw = MAX_GRAPH_POSTS - len(sub_posts)
		more = random.sample(draw_keys, num_draw)
		sub_posts.update({key: raw_sub_posts[key] for key in more})
		sub_params.update({key: raw_sub_params[key] for key in more})
	print("Using", len(sub_posts), "posts for inference (" + str(len(sub_params)), "params)")

	#build graph of these posts/params
	sub_graph = ParamGraph()
	sub_graph.build_graph(sub_posts, sub_params)

	graph_infer = False

	#add all seed posts from this subreddit to graph
	for post in seeds:

		#check if we already have params for this post - if not, flag for inference and add post to graph
		if post['id_h'] not in sub_params:
			graph_infer = True
			res = sub_graph.add_post(post)
			if res != True:
				print(res)

			#check for isolated node
			if type(res) == list and res[0] == "disconnect":
				tokens = set(res[1:])	#tokens from the disconnected post

				#can we find a post in our library with some of these tokens, or the same user? to get the node connected
				matching_posts = {key:value for key, value in raw_sub_posts.items() if len(tokens.intersection(extract_tokens(value))) != 0 or value['author_h'] == post['author_h']}
				#if too many matching token posts, sample down
				if len(matching_posts) > MAX_TOKEN_MATCH_POSTS:
					keep = random.sample(matching_posts.keys(), MAX_TOKEN_MATCH_POSTS)
					matching_posts = {key: matching_posts[key] for key in keep}
				#if no matching token/user posts - make a guess and move on, will use average of the subreddit for params
				elif len(matching_posts) == 0:
					print("   No matching posts! Using subreddit average")
					print(post['author_h'], tokens)
				#add the matching posts to the graph
				print("   Adding", len(matching_posts), "token-matching posts to graph")
				for post_id, matching_post in matching_posts.items():
					sub_graph.add_post(matching_post, raw_sub_params[post_id])
				#and re-add the seed post
				sub_graph.add_post(post)
	print("Updated graph has", sub_graph.graph.number_of_nodes(), "nodes and", sub_graph.graph.size(), "edges")

	#run node2vec to get embeddings - if we have to infer parameters
	if graph_infer:
		sub_graph.run_node2vec()

	#for each post, infer parameters and simulate
	for post in seeds:

		#if have fitted params, use those
		if post['id_h'] in sub_params:
			post_params = sub_params[post['id_h']]
		#otherwise, infer
		else:
			post_params = sub_graph.infer_params(post, 'weighted')

		#simulate a comment tree! just the event times first
		sim_root, all_times = sim_tree.simulate_comment_tree(post_params)

		#convert that to desired output format
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
		post_events = build_cascade_events(sim_root, post, user_ids)

		#add these events to running list
		all_events.extend(post_events)

		print("Finished post", post_counter, "/", len(raw_post_seeds))
		post_counter += 1

#finished all posts across all subreddit, time to dump
print("Finished simulation, have", len(all_events), "events to save")

reddit_sim_to_json(domain, all_events, outfile)
