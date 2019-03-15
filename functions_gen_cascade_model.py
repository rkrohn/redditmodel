#functions for paper_model.py - offloading and modularizing all the things

import file_utils
import functions_hybrid_model
import sim_tree
import tree_edit_distance

from shutil import copyfile
import subprocess
import os
import random
from argparse import *


#filepaths of pre-computed model files
posts_filepath = "model_files/posts/%s_posts.pkl"		#processed post data for each post, one file per group
														#each post maps original post id to numeric id, set of tokens, and user id

params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per group
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)

graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this group

users_filepath = "model_files/users/%s_users.txt"		#list of users seen in posts/comments, one file per group

domain_mapping_filepath = "model_files/domain_mapping.pkl"		#maps group -> domain for file loads

#filepaths of output/temporary files - used to pass graph to C++ node2vec for processing
temp_graph_filepath = "sim_files/%s_graph.txt"			#updated graph for this sim run
temp_params_filepath = "sim_files/%s_in_params.txt"		#temporary, filtered params for sim run (if sampled graph)
output_params_filepath = "sim_files/%s_params.txt"		#output params from node2vec

#filenames of filtered cascades and comments
cascades_filepath = "data_cache/filtered_cascades/%s_%s_cascades.pkl"	#domain and group cascades
comments_filepath = "data_cache/filtered_cascades/%s_%s_comments.pkl"	#domain and group comments


#parse out all command line arguments and return results
def parse_command_args():
	#arg parser
	parser = ArgumentParser(description="Simulate reddit cascades from partially-observed posts.")

	#required arguments (still with -flags, because clearer that way, and don't want to impose an order)
	parser.add_argument("-s", "--sub", dest="subreddit", required=True, help="subreddit to process")
	parser.add_argument("-o", "--out", dest="outfile", required=True, help="output filename")
	#must pick one of three processing options: a single id, random, or all
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("-id", dest="sim_post_id", default=None,  help="post id for single-processing")
	group.add_argument("-r", "--rand", dest="sim_post_id", action="store_const", const="random", help="choose a random post from the subreddit to simulate")
	group.add_argument("-a", "--all", dest="sim_post_id", action="store_const", const="all", help="simulate all posts in the subreddit")
	#must provide year and month for start of testing data set
	parser.add_argument("-y", "--year", dest="testing_start_year", required=True, help="year to use for test set")
	parser.add_argument("-m", "--month", dest="testing_start_month", required=True, help="month to use for test set")

	#optional args	
	parser.add_argument("-t", dest="time_observed", default=0, help="time of post observation, in hours")
	parser.add_argument("-g", "--graph", dest="max_nodes", default=None, help="max nodes in post graph for parameter infer")
	parser.add_argument("-q", "--qual", dest="min_node_quality", default=None, help="minimum node quality for post graph")
	parser.add_argument("-e", "--esp", dest="estimate_initial_params", action='store_true', help="estimate initial params as inverse quality weighted average of neighbor nodes")
	parser.set_defaults(estimate_initial_params=False)
	parser.add_argument("-l", "--testlen", dest="testing_len", default=1, help="number of months to use for testing")
	parser.add_argument("-p", "--periodtrain", dest="training_len", default=1, help="number of months to use for training (preceding first test month")

	args = parser.parse_args()		#parse the args (magic!)

	#extract arguments (since want to return individual variables)
	subreddit = args.subreddit
	sim_post_id = args.sim_post_id
	time_observed = float(args.time_observed)
	outfile = args.outfile
	max_nodes = args.max_nodes if args.max_nodes == None else int(args.max_nodes)
	min_node_quality = args.min_node_quality
	estimate_initial_params = args.estimate_initial_params
	testing_start_month = int(args.testing_start_month)
	testing_start_year = int(args.testing_start_year)
	testing_len = int(args.testing_len)
	training_len = int(args.training_len)
	#extra flag for batch processing
	if sim_post_id == "all":
		batch = True
	else:
		batch = False

	#print some log-ish stuff in case output being piped and saved
	print("Post ID:", sim_post_id)
	print("Time Observed:", time_observed)
	print("Output:", outfile)
	print("Source subreddit:", subreddit)
	print("Minimum node quality:", min_node_quality)
	print("Max graph size:", max_nodes)
	if estimate_initial_params:
		print("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
	print("Testing Period: %d-%d" % (testing_start_month, testing_start_year), "through %d-%d (%d months)" % (monthdelta(testing_start_month, testing_start_year, testing_len, inclusive=True)+(testing_len,)) if testing_len > 1 else "(%d month)" % testing_len)
	print("Training Period: %d-%d" % monthdelta(testing_start_month, testing_start_year, -training_len), "through %d-%d (%d months)" % (monthdelta(testing_start_month, testing_start_year, -1)+(training_len,)) if training_len > 1 else "(%d month)" % training_len)
	print("")

	#return all arguments
	return subreddit, sim_post_id, time_observed, outfile, max_nodes, min_node_quality, estimate_initial_params, batch, testing_start_month, testing_start_year, testing_len, training_len
#end parse_command_args


#given a month and year, shift by delta months (pos or neg) and return result
#if inclusive is True, reduct magnitude of delta by 1 to only include months in the range
def monthdelta(month, year, delta, inclusive=False):
	if inclusive:
		delta = delta + 1 if delta < 0 else delta - 1
	m = (month + delta) % 12
	if not m: 
		m = 12
	y = year + (month + delta - 1) // 12
	return m, y
#end monthdelta


#load cascade/comment data for specified subreddit
def load_subreddit_data(subreddit):
	print("")
	raw_posts = file_utils.load_pickle(cascades_filepath % (domain, group))
	raw_comments = file_utils.load_pickle(comments_filepath % (domain, group))
	print("Loaded", len(raw_posts), "posts and", len(raw_comments), "comments\n")

	return raw_posts, raw_comments
#end load_subreddit_data


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

	#do we need to sample/process the graph? sample if whole graph too big, imposing a min node quality, need to estimate initial params, or we don't have a precomputed graph file
	if (max_nodes != None and len(posts) > max_nodes) or file_utils.verify_file(graph_filepath % group) == False or min_node_quality != None or estimate_initial_params:

		#only sample down if we actually have to
		if max_nodes != None:
			print("\nSampling graph to", max_nodes, "nodes")
			#sample down posts
			graph_posts = user_sample_graph(posts, [sim_post], max_nodes, group, min_node_quality, fitted_quality)
		#otherwise, use them all
		else:
			graph_posts = posts

		#build graph, getting initial param estimate if required
		if estimate_initial_params:
			estimated_params = functions_hybrid_model.build_graph_estimate_node_params(graph_posts, fitted_params, fitted_quality, numeric_sim_post_id, temp_graph_filepath % group)
		else:
			functions_hybrid_model.build_graph(graph_posts, temp_graph_filepath % group)		
		
	#no graph sampling/processing, use the full set and copy graph file to temp location
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
	subprocess.check_call(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % group), "-ie:"+(temp_params_filepath % group), "-o:"+(output_params_filepath % group), "-d:6", "-l:3", "-w", "-s", "-otf"])
	print("")

	#load the inferred params (dictionary of numeric id -> params)
	all_inferred_params = functions_hybrid_model.load_params(output_params_filepath % group, posts, inferred=True)
	inferred_params = all_inferred_params[numeric_sim_post_id]

	return inferred_params
#end graph_infer


#given a ground-truth cascade, and an optional observed time, convert from list of comments to nested dictionary tree
#output is the form expected by sim_tree.simulate_comment_tree and tree_edit_distance.build_tree
#time_observed should be given in seconds
def convert_comment_tree(post, comments, time_observed=False):
	#build new list/structure of post comments - offset times by post time
	observed_tree = {}		#build as dictionary of id->object, then just pass in root

	#add post first - will serve as root of tree
	observed_tree[post['id_h']] = {'id': post['id_h'], 'time': 0, 'children': list()}		#post at time 0

	#add all comments, loop by comment time - filter by observed time, if given
	for comment in sorted(list(comments.values()), key=lambda k: k['created_utc']):
		#if filtering and comment within observed window, add to our object
		#if not filtering, add all comments
		if (time_observed == False) or (time_observed != False and comment['created_utc'] - post['created_utc'] <= time_observed * 3600):
			#new object in dictionary for this comment
			observed_tree[comment['id_h']] = {'id': comment['id_h'], 'time': (comment['created_utc'] - post['created_utc']) / 60.0, 'children': list()}		#time is offset from post in minutes
			#add this comment to parent's children list
			parent = comment['parent_id_h'][3:] if (comment['parent_id_h'].startswith("t1_") or comment['parent_id_h'].startswith("t3_")) else comment['parent_id_h']
			observed_tree[parent]['children'].append(observed_tree[comment['id_h']])

	#return post/root
	return observed_tree[post['id_h']]
#end convert_comment_tree


#given params, simulate a comment tree
def simulate_comment_tree(sim_post, sim_params, group, sim_comments, time_observed):
	print("\nSimulating comment tree")

	#load active users list to draw from when assigning users to comments
	user_ids = file_utils.load_pickle(users_filepath % group)

	#simulate tree structure + comment times!	
	print("Post created at", sim_post['created_utc'] / 60.0)
	#simulate from partially observed tree
	if time_observed != 0:
		#get alternate structure of observed tree
		observed_tree = convert_comment_tree(sim_post, sim_comments, time_observed)
		#simulate from this observed tree
		sim_root, all_times = sim_tree.simulate_comment_tree(sim_params, time_observed*60, observed_tree)
	#simulate entirely new tree from root only
	else:
		sim_root, all_times = sim_tree.simulate_comment_tree(sim_params)

	#convert that to desired output format
	sim_events = functions_hybrid_model.build_cascade_events(sim_root, sim_post, user_ids, group)
	#sort list of events by time
	sim_events = sorted(sim_events, key=lambda k: k['nodeTime']) 

	print("Generated", len(sim_events)-1, "total comments for post", sim_post['id_h'], "(including observed)")
	print("   ", len(sim_comments), "actual\n")

	return sim_events, sim_root		#return events list, and dictionary format of simulated tree
#end simulate_comment_tree


#given a post id and list of dataset ids, ensure post is in this set
def verify_post_id(input_sim_post_id, process_all, all_post_ids):
	#if processing all posts, return list of ids
	if process_all:
		random_post = False
		sim_post_id_list = all_post_ids
		print("Processing all posts")

	#if random post id, pick an id from loaded posts
	elif input_sim_post_id == "random":
		random_post = True
		sim_post_id_list = [random.choice(all_post_ids)]
		print("Choosing random simulation post:", sim_post_id_list)

	#if not random, make sure given post id is in the dataset
	else:
		random_post = False
		#if not in set, exit
		if input_sim_post_id not in all_post_ids:
			print("Given post id not in group set - exiting.\n")
			exit(0)
		sim_post_id_list = [input_sim_post_id]
	return sim_post_id_list, random_post
#end verify_post_id


#given simulated and ground-truth cascades, compute the tree edit distance between them
#both trees given as dictionary-nested structure (returned from simulate_comment_tree and convert_comment_tree)
def eval_trees(sim_dict_tree, sim_post, sim_comments):
	#get ground-truth cascade in same tree format
	truth_dict_tree = convert_comment_tree(sim_post, sim_comments)

	#return distance
	return tree_edit_distance.compare_trees(sim_dict_tree, truth_dict_tree)
#end eval_trees


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


#given complete set of posts/params for current subreddit, and seed posts, filter out posts not 
#meeting node quality threshhold, and sample down to reach the target graph size if necessary
def user_sample_graph(raw_sub_posts, seeds, max_nodes, subreddit, min_node_quality, fitted_quality):
	#graph seed posts to make sure they are preserved
	seed_posts = {seed['id_h']: raw_sub_posts[seed['id_h']] for seed in seeds}
	#and remove them from sampling pool
	for seed, seed_info in seed_posts.items():
		raw_sub_posts.pop(seed, None)

	#if have minimum node quality threshold, throw out any posts with too low a quality
	#this is the most important criteria, overrides all others (may lose user info, or end up with a smaller graph)
	if min_node_quality != None:
		raw_sub_posts = {key: value for key, value in raw_sub_posts.items() if value['id'] in fitted_quality and fitted_quality[value['id']] > min_node_quality}
		print("   Filtered to", len(raw_sub_posts), "based on minimum node quality of", min_node_quality)

	#no more than max_nodes posts, or this will never finish
	if max_nodes != None and len(raw_sub_posts)+len(seed_posts) > max_nodes:
		print("   Sampling down...")
		sub_posts = dict(random.sample(raw_sub_posts.items(), max_nodes-len(seed_posts)))
	#no limit, or not too many, take all that match the quality threshold
	else:
		sub_posts = raw_sub_posts

	#add seed posts back in, so they are built into the graph
	sub_posts.update(seed_posts)

	#return sampled posts
	return sub_posts
#end user_sample_graph