#given a set of seed posts, predict them all and create submission output file
#requires two command line args: input filename of seed posts, and output filename for simulation json results
#offload node2vec to c++, because speed

import file_utils
import sim_tree
from functions_hybrid_model import *

from shutil import copyfile
import sys
import subprocess
import os


#filepaths of input files
posts_filepath = "model_files/posts/%s_posts.pkl"		#processed post data for each post, one file per subreddit
														#each post maps original post id to numeric id, set of tokens, and user id
params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per subreddit
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)
graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this subreddit
users_filepath = "model_files/users/%s_users.txt"	#list of users seen in posts/comments, one file per subreddit
limit_filepath = "model_files/subreddit_size_limits.txt"	#per-subreddit graph size limits

#filepaths of output/temporary files
temp_graph_filepath = "sim_files/%s_graph.txt"			#updated graph for this sim run
temp_params_filepath = "sim_files/%s_in_params.txt"		#temporary, filtered params for sim run (if sampled graph)
output_params_filepath = "sim_files/%s_params.txt"		#output params from node2vec


DISPLAY = False

#verify command line args
if len(sys.argv) < 5:
	print("Incorrect command line arguments\nUsage: python3 hybrid_model.py <seed filename> <output filename> <domain> <max nodes for infer graph> <min_node_quality (set to -1 for no filter)> esp(optional, for estimating initial params)")
	exit(0)

#extract arguments
infile = sys.argv[1]
outfile = sys.argv[2]
domain = sys.argv[3]
default_max_nodes = int(sys.argv[4])
if len(sys.argv) > 5:
	min_node_quality = float(sys.argv[5])
else:
	min_node_quality = -1
if len(sys.argv) > 6 and sys.argv[6] == "esp":
	estimate_initial_params = True
else:
	estimate_initial_params = False

#read subreddit-specific size limits from file
with open(limit_filepath, 'r') as f:
	lines = f.readlines()
sub_limits = {}
#process each line, extract subreddit and limit
for line in lines:
	values = line.split()
	sub_limits[values[0]] = int(values[1])

#print some log-ish stuff in case output being piped and saved
print("Input", infile)
print("Output", outfile)
print("Domain", domain)
if min_node_quality != -1:
	print("Minimum node quality", min_node_quality)
if estimate_initial_params:
	print("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
print("")

file_utils.verify_dir("sim_files")		#ensure working directory exists

#load post seeds
raw_post_seeds = load_reddit_seeds(infile)

#convert to dictionary of subreddit->list of post objects
post_seeds = defaultdict(list)
for post in raw_post_seeds:
	#cve, group all together with subreddit set to cve
	if domain == "cve":
		post_seeds["cve"].append(post)
	else:
		post_seeds[post['subreddit']].append(post)
print({key : len(post_seeds[key]) for key in post_seeds})

all_events = []		#list for all sim events, across all seed posts
post_counter = 1	#counter of posts to simulate, across all subreddits

#process each subreddit
for subreddit, seeds in post_seeds.items():
	'''
	#TESTING ONLY!!!!
	if subreddit != "pivx":
		continue
	'''

	print("\nProcessing", subreddit, "with", len(seeds), "posts to simulate")

	#what is the graph limit for this subreddit?
	if subreddit in sub_limits:
		max_nodes = sub_limits[subreddit]
		print("Max graph size for this subreddit:", max_nodes)
	else:
		max_nodes = default_max_nodes
		print("Using default max graph size:", max_nodes)

	#load preprocessed posts for this subreddit
	if file_utils.verify_file(posts_filepath % subreddit):
		posts = file_utils.load_pickle(posts_filepath % subreddit)
		print("Loaded", len(posts), "processed posts from", posts_filepath % subreddit)
	else:
		print("Cannot simulate for subreddit", subreddit, "without processed posts file", posts_filepath % subreddit)
		exit(0)

	#find highest assigned post id for this data, so we know where to assign new ids if we need to
	next_id = max([value['id'] for key, value in posts.items()]) + 1

	#do we need to build a graph and infer at all? loop to find out
	infer = False
	infer_count = 0

	#also fetch/assign numeric ids to seed posts
	seed_numeric_ids = {}

	for seed_post in seeds:
		#if post id contains the t3_ prefix, strip it off so we don't have to change everything
		#(will manually add it back to output later)
		if seed_post['id_h'].startswith(POST_PREFIX):
			seed_post['id_h'] = seed_post['id_h'][3:]
		#does this post need to be added to the graph? if yes, compute new edges and assign new id
		if seed_post['id_h'] not in posts:
			infer = True			#flag for graph build
			infer_count += 1
			seed_numeric_ids[seed_post['id_h']] = next_id			#assign id to this unseen post
			next_id += 1
			#print("New id", next_id-1, "assigned to seed post", seed_post['id_h'])
		#seen this post, have params fitted, just fetch id
		else:
			seed_numeric_ids[seed_post['id_h']] = posts[seed_post['id_h']]['id']
	print(infer_count, "new posts")

	#load in fitted simulation params - will use either fitted or inferred, whichever is better
	#but definitely need these for graph build
	fitted_params, fitted_quality = load_params(params_filepath % subreddit, posts, False, True)	

	#graph stuff - sample graph if necessary, add new nodes, etc
	if infer:
	
		graph = {}
		isolated_nodes = []
		added_count = 0
		sample_graph = False

		#do we need to sample the graph? sample if whole graph too big, or we don't have a precomputed graph file
		if len(posts) + len(seeds) > max_nodes or file_utils.verify_file(graph_filepath % subreddit) == False or min_node_quality != -1:
			print("Sampling graph to", max_nodes, "nodes")
			graph_posts = user_sample_graph(posts, seeds, max_nodes-infer_count, subreddit, min_node_quality, fitted_quality)
			build_graph(graph_posts, temp_graph_filepath % subreddit)
			get_sampled_params(graph_posts, params_filepath % subreddit, temp_params_filepath % subreddit)
			sample_graph = True

		#for initial param estimate, store dict of numeric id -> param initialization estimates
		param_estimate = {}

		#add all seed posts from this subreddit to graph, if not already there
		#also convert seed post ids to prefix-less format, if required
		#and assign each seed post a numeric id for node2vec fun (use existing id if seen post before)x
		print("Adding seed posts to graph")
		for seed_post in seeds:
			#does this post need to be added to the graph? if yes, compute new edges and assign new id
			if seed_post['id_h'] not in posts:
				if sample_graph:
					graph, isolated_nodes, posts, new_initial_params = add_post_edges(graph, isolated_nodes, graph_posts, seed_post, seed_numeric_ids[seed_post['id_h']], subreddit, posts, estimate_initial_params, fitted_params, fitted_quality)
				else:		
					graph, isolated_nodes, posts, new_initial_params = add_post_edges(graph, isolated_nodes, posts, seed_post, seed_numeric_ids[seed_post['id_h']], subreddit, posts, estimate_initial_params, fitted_params, fitted_quality)
				if estimate_initial_params and new_initial_params != None:
					param_estimate[seed_numeric_ids[seed_post['id_h']]] = new_initial_params
					#print("estimated initial params", new_initial_params)
				added_count += 1

		print("   Added", added_count, "nodes (" + str(len(isolated_nodes)), "isolated) and", len(graph), "edges")

		#copy subreddit graph file if using the whole thing
		if sample_graph == False:
			print("Copying full graph file")
			copyfile(graph_filepath % subreddit, temp_graph_filepath % subreddit)
		#append these new edges to subreddit graph, sampled or complete
		print("Adding edges for seed posts")
		with open(temp_graph_filepath % subreddit, "a") as f:
			for edge, weight in graph.items():
				f.write("%d %d %f\n" % (edge[0], edge[1], weight))
			for node in isolated_nodes:
				f.write("%d\n" % node)
		print("Saved updated post-graph to", temp_graph_filepath % subreddit)

		#initial param estimate: use temp param file, filtered or not
		if estimate_initial_params:
			#no temp params file, copy full graph params first
			if file_utils.verify_file(temp_params_filepath % subreddit) == False:
				print("Copying full param file")
				copyfile(params_filepath % subreddit, temp_params_filepath % subreddit)
			#append new param estimates (initializations) to file
			with open(temp_params_filepath % subreddit, "a") as f:
				for post_id, init_params in param_estimate.items():
					f.write("%d %f %f %f %f %f %f\n" % (post_id, init_params[0], init_params[1], init_params[2], init_params[3], init_params[4], init_params[5]))
			print("Added", len(param_estimate), "seed param initializations to param file.")

		#run node2vec to get embeddings - if we have to infer parameters
		#offload to C++, because I feel the need... the need for speed!:

		if file_utils.verify_file(output_params_filepath % subreddit):
			os.remove(output_params_filepath % subreddit)		#clear output to prevent append

		#get correct params filepath for node2vec call: use full file only if graph not sampled and not estimating initial params
		run_params_path = (temp_params_filepath % subreddit) if sample_graph or estimate_initial_params else (output_params_filepath % subreddit)

		#sampled graph and params
		subprocess.check_call(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % subreddit), "-ie:"+run_params_path, "-o:"+(output_params_filepath % subreddit), "-d:6", "-l:3", "-w", "-s"])

		#load the inferred params
		inferred_params = load_params(output_params_filepath % subreddit, posts, inferred=True)

	#end if infer
	else:
		print("No infer needed, skipping graph build.")

	#load active users list to draw from when assigning users to comments
	user_ids = file_utils.load_pickle(users_filepath % subreddit)

	#node2vec finished, on to the simulation!
	#for each post, infer parameters and simulate
	print("\nSimulating comment trees...")
	infer_count = 0
	fitted_count = 0
	added_events = 0
	for seed_post in seeds:

		#if we can, use fitted params
		if posts[seed_post['id_h']]['id'] in fitted_params:
			post_params = fitted_params[posts[seed_post['id_h']]['id']]
			fitted_count += 1
		#otherwise, use inferred params
		elif infer:
			post_params = inferred_params[posts[seed_post['id_h']]['id']]
			infer_count += 1
			#print("Inferred post params:", post_params)
		else:
			print("Something's gone wrong - no params for this post! Skipping.")
			continue

		#simulate a comment tree!
		sim_root, all_times = sim_tree.simulate_comment_tree(post_params)

		#convert that to desired output format
		post_events = build_cascade_events(sim_root, post, user_ids)

		#add these events to running list
		all_events.extend(post_events)
		added_events += len(post_events)

		if post_counter % 50 == 0:
			print("Finished post", post_counter, "/", len(raw_post_seeds))
		post_counter += 1

	print("Used fitted params for", fitted_count, "posts and inferred params for", infer_count, "posts")
	print("Generated", added_events, "for subreddit", subreddit)

#finished all posts across all subreddit, time to dump
print("\nFinished all simulations, have", len(all_events), "events to save")

reddit_sim_to_json(domain, all_events, outfile)
