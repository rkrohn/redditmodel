#given a set of seed posts, predict them all and create submission output file
#requires two command line args: input filename of seed posts, and output filename for simulation json results
#offload node2vec to c++, because speed

import file_utils
import sim_tree
from functions_hybrid_model import *

from shutil import copyfile
import sys
import subprocess


#filepaths of input files
posts_filepath = "model_files/posts/%s_posts.pkl"		#processed post data for each post, one file per subreddit
														#each post maps original post id to numeric id, set of tokens, and user id
params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per subreddit
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)
graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this subreddit
users_filepath = "model_files/users/%s_users.txt"	#list of users seen in posts/comments, one file per subreddit

#filepaths of output/temporary files
temp_graph_filepath = "sim_files/%s_graph.txt"			#updated graph for this sim run
output_params_filepath = "sim_files/%s_params.txt"		#output params from node2vec


DISPLAY = False

#verify command line args
if len(sys.argv) != 4:
	print("Incorrect command line arguments\nUsage: python3 hybrid_model.py <seed filename> <output filename> <domain>")
	exit(0)

#extract arguments
infile = sys.argv[1]
outfile = sys.argv[2]
domain = sys.argv[3]

#print some log-ish stuff in case output being piped and saved
print("Input", infile)
print("Output", outfile)
print("Domain", domain)
print("")

file_utils.verify_dir("sim_files")		#ensure working directory exists

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
	if subreddit != "Lisk":
		continue
	'''

	print("\nProcessing", subreddit, "with", len(seeds), "posts to simulate")

	#load preprocessed posts for this subreddit
	if file_utils.verify_file(posts_filepath % subreddit):
		posts = file_utils.load_pickle(posts_filepath % subreddit)
		print("Loaded", len(posts), "processed posts from", posts_filepath % subreddit)
	else:
		print("Cannot simulate for subreddit", subreddit, "without processed posts file", posts_filepath % subreddit)
		exit(0)

	#find highest assigned post id for this data, so we know where to assign new ids if we need to
	next_id = max([value['id'] for key, value in posts.items()]) + 1

	#do we need to build a graph at all? loop to find out
	build_graph = False

	#also fetch/assign numeric ids to seed posts
	seed_numeric_ids = {}

	for seed_post in seeds:
		#if post id contains the t3_ prefix, strip it off so we don't have to change everything
		#(will manually add it back to output later)
		if seed_post['id_h'].startswith(POST_PREFIX):
			seed_post['id_h'] = seed_post['id_h'][3:]
		#does this post need to be added to the graph? if yes, compute new edges and assign new id
		if seed_post['id_h'] not in posts:
			build_graph = True				#flag for graph build
			seed_numeric_ids[seed_post['id_h']] = next_id			#assign id to this unseen post
			next_id += 1
		#seen this post, have params fitted, just fetch id
		else:
			seed_numeric_ids[seed_post['id_h']] = posts[seed_post['id_h']]['id']


	#graph stuff - sample graph if necessary, add new nodes, etc
	if build_graph:
	
		graph = {}
		isolated_nodes = []
		added_count = 0

		#add all seed posts from this subreddit to graph, if not already there
		#also convert seed post ids to prefix-less format, if required
		#and assign each seed post a numeric id for node2vec fun (use existing id if seen post before)
		print("Adding seed posts to graph")
		for seed_post in seeds:
			#does this post need to be added to the graph? if yes, compute new edges and assign new id
			if seed_post['id_h'] not in posts:		
				graph, isolated_nodes, posts = add_post_edges(graph, isolated_nodes, posts, seed_post, seed_numeric_ids[seed_post['id_h']])
				added_count += 1

		print("   Added", added_count, "nodes (" + str(len(isolated_nodes)), "isolated) and", len(graph), "edges")

		#copy subreddit graph file, append these new edges to it
		copyfile(graph_filepath % subreddit, temp_graph_filepath % subreddit)
		with open(temp_graph_filepath % subreddit, "a") as f:
			for edge, weight in graph.items():
				f.write("%d %d %f\n" % (edge[0], edge[1], weight))
			for node in isolated_nodes:
				f.write("%d\n" % node)
		print("Saved updated post-graph to", temp_graph_filepath % subreddit)


		#run node2vec to get embeddings - if we have to infer parameters
		#offload to C++, because I feel the need... the need for speed!:
		if added_count != 0:
			subprocess.check_call(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % subreddit), "-ie:"+(params_filepath % subreddit), "-o:"+(output_params_filepath % subreddit), "-d:6", "-l:3", "-w", "-s"])

		#load the inferred params
		inferred_params = load_params(output_params_filepath % subreddit, posts, inferred=True)

	#end if build_graph
	else:
		print("No infer needed, skipping graph build.")

	#load in fitted simulation params - will use either fitted or inferred, whichever is better
	fitted_params = load_params(params_filepath % subreddit, posts)	

	#load active users list to draw from when assigning users to comments
	user_ids = file_utils.load_pickle(users_filepath % subreddit)

	#node2vec finished, on to the simulation!
	#for each post, infer parameters and simulate
	print("Simulating comment trees...")
	for seed_post in seeds:

		#if we can, use fitted params
		if posts[seed_post['id_h']]['id'] in fitted_params:
			post_params = fitted_params[posts[seed_post['id_h']]['id']]
		#otherwise, use inferred params
		elif build_graph:
			post_params = inferred_params[posts[seed_post['id_h']]['id']]
		else:
			print("Something's gone wrong - no params for this post! Skipping.")
			continue

		#simulate a comment tree!
		sim_root, all_times = sim_tree.simulate_comment_tree(post_params)

		#convert that to desired output format
		post_events = build_cascade_events(sim_root, post, user_ids)

		#add these events to running list
		all_events.extend(post_events)

		if post_counter % 50 == 0:
			print("Finished post", post_counter, "/", len(raw_post_seeds))
		post_counter += 1

#finished all posts across all subreddit, time to dump
print("Finished all simulations, have", len(all_events), "events to save")

reddit_sim_to_json(domain, all_events, outfile)
