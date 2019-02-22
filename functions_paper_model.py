#functions for paper_model.py - offloading and modularizing all the things

import file_utils
import functions_hybrid_model
import sim_tree

from shutil import copyfile
import subprocess
import os
import sys
import random


#filepaths of pre-computed model files
posts_filepath = "model_files/posts/%s_posts.pkl"		#processed post data for each post, one file per group
														#each post maps original post id to numeric id, set of tokens, and user id

params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per group
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)

graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this group

users_filepath = "model_files/users/%s_users.txt"		#list of users seen in posts/comments, one file per group

domain_mapping_filepath = "model_files/domain_mapping.pkl"		#maps group -> domain for file loads

limit_filepath = "model_files/group_size_limits.txt"	#per-group graph size limits

#filepaths of output/temporary files - used to pass graph to C++ node2vec for processing
temp_graph_filepath = "sim_files/%s_graph.txt"			#updated graph for this sim run
temp_params_filepath = "sim_files/%s_in_params.txt"		#temporary, filtered params for sim run (if sampled graph)
output_params_filepath = "sim_files/%s_params.txt"		#output params from node2vec

#filenames of filtered cascades and comments
cascades_filepath = "data_cache/filtered_cascades/%s_%s_cascades.pkl"	#domain and group cascades
comments_filepath = "data_cache/filtered_cascades/%s_%s_comments.pkl"	#domain and group comments


#for a given post, infer parameters using post graph
def graph_infer(sim_post, sim_post_id, group, max_nodes, min_node_quality, estimate_initial_params):
	print("Inferring post parameters from post graph")

	#load preprocessed posts for this group
	if file_utils.verify_file(posts_filepath % group):
		posts = file_utils.load_pickle(posts_filepath % group)
		print("Loaded", len(posts), "processed posts from", posts_filepath % group)
	else:
		print("Cannot simulate for group", group, "without processed posts file", posts_filepath % group)
		exit(0)

	#if seed post not in posts file - we're gonna have a bad time
	if sim_post['id_h'] not in posts:
		print("Simulation post not in dataset - exiting\n")
		exit(0)

	#grab numeric/graph id of sim post
	numeric_sim_post_id = posts[sim_post_id]['id']

	#load in fitted simulation params - need these for graph build
	fitted_params, fitted_quality = functions_hybrid_model.load_params(params_filepath % group, posts, False, True)	

	#remove sim post from graph params - no cheating! (pop based on numeric id)
	res = fitted_params.pop(numeric_sim_post_id)
	res = fitted_quality.pop(numeric_sim_post_id)

	#graph stuff - sample graph if necessary, add new nodes, etc
	graph = {}
	isolated_nodes = []
	added_count = 0
	sample_graph = False

	#do we need to sample the graph? sample if whole graph too big, imposing a min node quality, need to estimate initial params, we don't have a precomputed graph file
	if len(posts) > max_nodes or file_utils.verify_file(graph_filepath % group) == False or min_node_quality != -1 or estimate_initial_params:
		print("\nSampling graph to", max_nodes, "nodes")
		#sample down posts
		graph_posts = functions_hybrid_model.user_sample_graph(posts, [sim_post], max_nodes, group, min_node_quality, fitted_quality)
		#build graph, getting initial param estimate if required
		if estimate_initial_params:
			estimated_params = functions_hybrid_model.build_graph_estimate_node_params(graph_posts, fitted_params, fitted_quality, numeric_sim_post_id, temp_graph_filepath % group)
		else:
			functions_hybrid_model.build_graph(graph_posts, temp_graph_filepath % group)
		
		sample_graph = True
	#no graph sample, use the full set and copy graph file to temp location
	else:
		graph_posts = posts
		copyfile(graph_filepath % group, temp_graph_filepath % group)
		print("Copied complete post-graph to", temp_graph_filepath % group)

	#ALWAYS sample down params to match whatever graph we have - because we can't use the previously fitted params!
	if estimate_initial_params:
		functions_hybrid_model.get_graph_params(graph_posts, numeric_sim_post_id, fitted_params, fitted_quality, temp_params_filepath % group, estimated_params)
	else:
		functions_hybrid_model.get_graph_params(graph_posts, numeric_sim_post_id, fitted_params, fitted_quality, temp_params_filepath % group)

	#graph is built and ready - graph file and input params file

	#run node2vec to get embeddings - if we have to infer parameters
	#offload to C++, because I feel the need... the need for speed!:

	if file_utils.verify_file(output_params_filepath % group):
		os.remove(output_params_filepath % group)		#clear output to prevent append

	#run node2vec on graph and params
	subprocess.check_call(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % group), "-ie:"+(temp_params_filepath % group), "-o:"+(output_params_filepath % group), "-d:6", "-l:3", "-w", "-s"])
	print("")

	#load the inferred params (dictionary of numeric id -> params)
	all_inferred_params = functions_hybrid_model.load_params(output_params_filepath % group, posts, inferred=True)
	inferred_params = all_inferred_params[numeric_sim_post_id]

	return inferred_params
#end graph_infer


#given params, simulate a comment tree
def simulate_comment_tree(sim_post, sim_post_id, sim_params, group, sim_comments, time_observed):
	print("\nSimulating comment tree")

	#load active users list to draw from when assigning users to comments
	user_ids = file_utils.load_pickle(users_filepath % group)

	#simulate tree structure + comment times!	
	print("Post created at", sim_post['created_utc'] / 60.0)
	#simulate from partially observed tree
	if time_observed != 0:
		#build new list/structure of observed comments to pass to sim function - offset times by post time
		observed_tree = {}		#build as dictionary of id->object, then just pass in root
		#add post first
		observed_tree[sim_post_id] = {'id': sim_post_id, 'time': 0, 'children': list()}		#post at time 0
		#add all comments that we observed, loop by comment time
		for comment in sorted(list(sim_comments.values()), key=lambda k: k['created_utc']):
			#if comment within observed window, add to our object
			if comment['created_utc'] - sim_post['created_utc'] <= time_observed * 3600:
				#new object in dictionary for this comment
				observed_tree[comment['id_h']] = {'id': comment['id_h'], 'time': (comment['created_utc'] - sim_post['created_utc']) / 60.0, 'children': list()}		#time is offset from post in minutes
				#add this comment to parent's children list
				parent = comment['parent_id_h'][3:] if (comment['parent_id_h'].startswith("t1_") or comment['parent_id_h'].startswith("t3_")) else comment['parent_id_h']
				observed_tree[parent]['children'].append(observed_tree[comment['id_h']])
		#simulate from this observed tree
		sim_root, all_times = sim_tree.simulate_comment_tree(sim_params, time_observed*60, observed_tree[sim_post_id])
	#simulate entirely new tree from root only
	else:
		sim_root, all_times = sim_tree.simulate_comment_tree(sim_params)

	#convert that to desired output format
	sim_events = functions_hybrid_model.build_cascade_events(sim_root, sim_post, user_ids, group)
	#sort list of events by time
	sim_events = sorted(sim_events, key=lambda k: k['nodeTime']) 

	print("Generated", len(sim_events)-1, "comments for post", sim_post_id)
	print("   ", len(sim_comments), "actual\n")

	return sim_events
#end simulate_comment_tree


#load cascade/comment data for specified group
def load_group_data(group):
	#read group -> domain mapping for later file loads
	domain_mapping = file_utils.load_pickle(domain_mapping_filepath)
	if group not in domain_mapping:
		print(group, "not in domain mapping - exiting.\n")
		exit(0)
	domain = domain_mapping[group]

	#load cascades and comments for this group
	print("")
	raw_posts = file_utils.load_pickle(cascades_filepath % (domain, group))
	raw_comments = file_utils.load_pickle(comments_filepath % (domain, group))
	print("Loaded", len(raw_posts), "posts and", len(raw_comments), "comments\n")

	return raw_posts, raw_comments
#end load_group_data


#parse out all command line arguments and return results
def parse_command_args():
	#verify command line args
	if len(sys.argv) < 7:
		print("Incorrect command line arguments\nUsage: python3 paper_model.py <source group> <seed post id or \"random\"> <time post observed (hours)> <output filename> <max nodes for infer graph (if none given in group_size_limits.txt> <min_node_quality (set to -1 for no filter)> esp(optional, for estimating initial params)")
		exit(0)

	#extract arguments
	group = sys.argv[1]
	sim_post_id = sys.argv[2]
	time_observed = float(sys.argv[3])
	outfile = sys.argv[4]
	max_nodes = int(sys.argv[5])
	min_node_quality = float(sys.argv[6])
	if len(sys.argv) > 7 and sys.argv[7] == "esp":
		estimate_initial_params = True
	else:
		estimate_initial_params = False

	#read group-specific size limits from file
	with open(limit_filepath, 'r') as f:
		lines = f.readlines()
	sub_limits = {}
	#process each line, extract group and limit
	for line in lines:
		values = line.split()
		sub_limits[values[0]] = int(values[1])
	#use limit from file, if it exists
	limit_from_file = False
	if group in sub_limits:
		max_nodes = sub_limits[group]
		limit_from_file = True

	#print some log-ish stuff in case output being piped and saved
	print("Post ID:", sim_post_id)
	print("Time Observed:", time_observed)
	print("Output:", outfile)
	print("Source group:", group)
	if min_node_quality != -1:
		print("Minimum node quality:", min_node_quality)
	else:
		print("No minimum node quality")
	print("Max graph size:", max_nodes, "from file" if limit_from_file else "from argument")
	if estimate_initial_params:
		print("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
	print("")

	#return all arguments
	return group, sim_post_id, time_observed, outfile, max_nodes, min_node_quality, estimate_initial_params 
#end parse_command_args


#given a post id and list of dataset ids, ensure post is in this set
def verify_post_id(sim_post_id, all_post_ids):
	#if random post id, pick an id from loaded posts
	if sim_post_id == "random":
		random_post = True
		sim_post_id = random.choice(all_post_ids)
		print("Choosing random simulation post:", sim_post_id)
	#if not random, make sure given post id is in the dataset
	else:
		random_post = False
		#if not in set, exit
		if sim_post_id not in all_post_ids:
			print("Given post id not in group set - exiting.\n")
			exit(0)
	return sim_post_id, random_post
#end verify_post_id


#convert ground-truth cascade to output format and save for later evaluation
def save_groundtruth(post, comments, outfile):
	print("Saving groundtruth as", outfile+"_groundtruth.csv")

	#convert ground_truth from given format to eval format
	truth_events = []
	#include post
	truth_events.append({'rootID': "t3_"+post['id_h'], 'nodeID': "t3_"+post['id_h'], 'parentID': "t3_"+post['id_h']})
	#and all comments, sorted by time
	for comment in sorted(comments.values(), key=lambda k: k['created_utc']): 
		truth_events.append({'rootID': comment['link_id_h'], 'nodeID': "t1_"+comment['id_h'], 'parentID': comment['parent_id_h']})

	#save ground-truth of this cascade	
	file_utils.save_csv(truth_events, outfile+"_groundtruth.csv", fields=['rootID', 'nodeID', 'parentID'])
#end save_groundtruth


#save simulated cascade to json
def save_sim_json(group, sim_post_id, random_post, time_observed, min_node_quality, max_nodes, estimate_initial_params, sim_events, outfile):
	#save sim results to output file - json with events and run settings
	print("Saving results to", outfile + ".json...")    
	#write to json, include some run info
	output = {'group'    				: group,
	          'post_id'  				: sim_post_id,
	          'post_randomly_selected'	: random_post,
	          'time_observed'   		: time_observed,
	          'min_node_quality' 		: min_node_quality,
	          'max_graph_size' 			: max_nodes,
	          'estimate_initial_params' : estimate_initial_params,
	          'data'     				: sim_events}
	file_utils.save_json(output, outfile+".json")
#end save_sim_json


#given a list of events, check event times to make sure they are in sorted order
def verify_sorted(events):
	prev_time = -1
	for event in events:
		curr_time = event['nodeTime']
		if prev_time != -1 and curr_time < prev_time:
			print("out of order!", prev_time, curr_time)
			return False
		prev_time = curr_time
	print("Events are sorted")
	return True
#end verify_sorted