#functions for paper_model.py - offloading and modularizing all the things

import file_utils
import functions_hybrid_model
import sim_tree
import tree_edit_distance
import fit_cascade_gen_model as fit_cascade

from shutil import copyfile
import subprocess
import os
import random
from argparse import *
import pandas as pd
import string
import glob
import re



#filepaths of data and pre-processed files - keeping everything in the same spot, for sanity/simplicity

#raw posts for (sub, sub, year, month)
raw_posts_filepath = "reddit_data/%s/%s_submissions_%d_%d.tsv"	
#raw comments for (sub, sub, post year, comment year, comment month)
raw_comments_filepath = "reddit_data/%s/%s_%sdiscussions_comments_%s_%s.tsv"  
#processed posts for (sub, sub, year, month) - dictionary of post id -> post containing title tokens, author, created utc
processed_posts_filepath = "reddit_data/%s/%s_processed_posts_%d_%d.pkl"
#fitted params for posts for (sub, sub, year, month) - dictionary of post id -> params tuple
fitted_params_filepath = "reddit_data/%s/%s_post_params_%d_%d.pkl"
#reconstructed cascades for (sub, sub, year, month) - dictionary of post id -> cascade dict, with "time", "num_comments", and "replies", where "replies" is nested list of reply objects
cascades_filepath = "reddit_data/%s/%s_cascades_%d_%d.pkl"

#filepaths of output/temporary files - used to pass graph to C++ node2vec for processing
temp_graph_filepath = "sim_files/%s_graph.txt"			#updated graph for this sim run
temp_params_filepath = "sim_files/%s_in_params.txt"		#temporary, filtered params for sim run (if sampled graph)
output_params_filepath = "sim_files/%s_params.txt"		#output params from node2vec


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
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose output")
	parser.set_defaults(verbose=False)

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
	verbose = args.verbose
	#extra flag for batch processing
	if sim_post_id == "all":
		batch = True
	else:
		batch = False

	#compute start of training period for easy use later
	training_start_month, training_start_year = monthdelta(testing_start_month, testing_start_year, -training_len)

	#hackery: declare a special print function for verbose output
	#make it global here for all the other functions to use
	global vprint
	if verbose:
		def vprint(*args):
			# Print each argument separately so caller doesn't need to
			# stuff everything to be printed into a single string
			for arg in args:
				print(arg, end='')
			print("")
	else:   
		vprint = lambda *a: None      # do-nothing function

	#print some log-ish stuff in case output being piped and saved
	vprint("Post ID: ", sim_post_id)
	vprint("Time Observed: ", time_observed)
	vprint("Output: ", outfile)
	vprint("Source subreddit: ", subreddit)
	vprint("Minimum node quality: ", min_node_quality)
	vprint("Max graph size: ", max_nodes)
	if estimate_initial_params:
		print("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
	vprint("Testing Period: %d-%d" % (testing_start_month, testing_start_year), " through %d-%d (%d months)" % (monthdelta(testing_start_month, testing_start_year, testing_len, inclusive=True)+(testing_len,)) if testing_len > 1 else " (%d month)" % testing_len)
	vprint("Training Period: %d-%d" % (training_start_month, training_start_year), " through %d-%d (%d months)" % (monthdelta(training_start_month, training_start_year, training_len, inclusive=True)+(training_len,)) if training_len > 1 else " (%d month)" % training_len)
	vprint("")

	#return all arguments
	return subreddit, sim_post_id, time_observed, outfile, max_nodes, min_node_quality, estimate_initial_params, batch, testing_start_month, testing_start_year, testing_len, training_start_month, training_start_year, training_len, verbose
#end parse_command_args


#given a month and year, shift by delta months (pos or neg) and return result
#if inclusive is True, reduce magnitude of delta by 1 to only include months in the range
def monthdelta(month, year, delta, inclusive=False):
	if inclusive:
		delta = delta + 1 if delta < 0 else delta - 1
	m = (month + delta) % 12
	if not m: 
		m = 12
	y = year + (month + delta - 1) // 12
	return m, y
#end monthdelta


#given a subreddit, starting month-year, and number of months to load, load processed posts
#if params = True, also load fitted params for these posts
#if files don't exist, call methods to perform preprocessing
def load_processed_posts(subreddit, start_month, start_year, num_months, load_params=False):
	posts = {}
	params = {}

	#loop months to load
	for m in range(0, num_months):
		month, year = monthdelta(start_month, start_year, m)	#calc month to load
		vprint("Loading %d-%d" % (month, year))

		#load posts if processed file exists
		if file_utils.verify_file(processed_posts_filepath % (subreddit, subreddit, year, month)):
			month_posts = file_utils.load_pickle(processed_posts_filepath % (subreddit, subreddit, year, month))
		#if posts file doesn't exist, create it - loading in the process
		else:
			vprint("   Processed posts file doesn't exist, creating now")
			month_posts = process_posts(subreddit, month, year)

		#params, if desired
		if load_params:
			#load if params file exists
			if file_utils.verify_file(fitted_params_filepath % (subreddit, subreddit, year, month)):
				params.update(file_utils.load_pickle(fitted_params_filepath) % (subreddit, subreddit, year, month))
			#if params file doesn't exist, create it - loading in the process
			else:
				vprint("   Fitted params file doesn't exist, creating now")
				params.update(fit_posts(subreddit, month, year, month_posts))

		#add this month to overall
		posts.update(month_posts)

	vprint("Loaded %d posts and %d params" % (len(posts), len(params)))

	#throw out posts that we don't have params for - incomplete or some other issue
	if len(posts) != len(params):
		del_keys = set([post_id for post_id in posts.keys() if post_id not in params])
		for key in del_keys:
			posts.pop(key, None)
		vprint("Deleted %d posts without params" % len(del_keys))

	#return results
	if load_params: return posts, params
	return posts
#end load_processed_posts


#for a given subreddit, month, and year, preprocess those posts - tokenize and save as pickle
def process_posts(subreddit, month, year):
	#make sure raw posts exist
	if file_utils.verify_file(raw_posts_filepath % (subreddit, subreddit, year, month)) == False:
		print("No raw posts to process - exiting")
		exit(0)
	#load raw posts into pandas dataframe
	posts_df = pd.read_csv(raw_posts_filepath % (subreddit, subreddit, year, month), sep='\t')
	vprint("Loaded %s raw posts" % len(posts_df.index))

	#convert to our nested dictionary structure
	posts = {}
	for index, row in posts_df.iterrows():
		#build new post dict
		post = {}
		post['tokens'] = extract_tokens(row['title'])
		post['time'] = int(row['created_utc'])
		post['author'] = row['author']

		#add to overall post dict
		post_id = row['name']		#post id with t3_ prefix
		posts[post_id] = post

	#save to pickle 
	file_utils.save_pickle(posts, processed_posts_filepath % (subreddit, subreddit, year, month))

	return posts
#end process_posts


#given a text string, extract words by tokenizing and normalizing (no limitization for now)
#removes all leading/trailing punctuation
#returns list, preserving duplicates
def extract_tokens(text):
	punctuations = list(string.punctuation)		#get list of punctuation characters
	punctuations.append('—')	#kill these too
	
	if text != None:
		tokens = [word.lower() for word in text.split()]	#tokenize and normalize (to lower)		
		tokens = [word.strip("".join(punctuations)) for word in tokens]		#strip trailing and leading punctuation
		tokens = [word for word in tokens if word != '' and word not in punctuations and word != '']		#remove punctuation-only tokens and empty strings
	else:
		tokens = []

	return tokens	#return list, preserves duplicates
#end extract_tokens


#given a subreddit, month, and year, and loaded posts (but not comments),
#fit parameters for these posts and save results as pickle
#load cascades if they already exist, otherwise build them first
#cascades format is post_id -> nested dict of replies, time, comment_count_total and comment_count_direct
#each reply has id, time, and their own replies field
def fit_posts(subreddit, month, year, posts):
	#if reconstructed cascades already exist, load those
	if file_utils.verify_file(cascades_filepath % (subreddit, subreddit, year, month)):
		cascades = file_utils.load_pickle(cascades_filepath % (subreddit, subreddit, year, month))
		vprint("Loaded %d cascades" % len(cascades))
	#otherwise, reconstruct the cascades (get them back as return value)
	else:
		#load comments associated with this month of posts
		comments = load_comments(subreddit, month, year, posts)
		#reconstruct the cascades
		cascades = build_cascades(subreddit, month, year, posts, comments)

	#fit parameters to each cascade
	vprint("Fitting %d cascades for %s %d-%d" % (len(cascades), subreddit, month, year))

	cascade_params = {}		#build dict of post id -> 6 fitted params + quality
	post_count = 0

	#loop and fit cascades
	for post_id, post in cascades.items():		
		param_res = fit_cascade.fit_params(post)	#fit the current cascade 

		#if negative comment times, skip this cascade and move to next
		if param_res == False: continue

		cascade_params[post_id] = param_res		#store params

		post_count += 1
		if post_count % 2500 == 0:
			vprint("Fitted %d cascades" % post_count)	

	#dump params to file
	vprint("Fitted a total of %d cascades" % len(cascade_params))

	file_utils.save_pickle(cascade_params, fitted_params_filepath % (subreddit, subreddit, year, month))
	
	return cascade_params 		#return all params
#end fit_posts


#given subreddit, month, year, and loaded posts, load comments associated with those posts
#(filtering out other comments we may encounter along the way)
#returns nested dictionary of comment_id -> link_id, parent_id, and time (all ids with prefixes)
def load_comments(subreddit, post_month, post_year, posts):
	#build set of post ids we care about
	post_ids = set(posts.keys())

	#get list of comment files for this subreddit and post year
	comment_files = glob.glob(raw_comments_filepath % (subreddit, subreddit, post_year, "*", "*"))

	#loop comment files, only load the ones that occur after the post
	comments = {}			#all relevant comments
	scanned_count = 0		#count of comments scanned for relevancy
	for file in sorted(comment_files):
		#extract years and month from filename
		file_post_year, file_year, file_month = [int(s) for s in re.findall(r'\d+', file)]

		#if file is for comments before post, skip
		if file_year < post_year or (file_year == post_year and file_month < post_month):
			continue

		#load this file's comments
		month_comments_df = pd.read_csv(file, sep='\t')

		#convert to our nested dictionary structure, filtering out irrelevant comments along the way
		month_comments = {}			#comment id-> dict with link_id, parent_id, and time
		for index, row in month_comments_df.iterrows():
			#skip comment if not for post in set
			if row['link_id'] not in post_ids:
				continue

			#build new comment dict
			comment = {}
			comment['time'] = int(row['created_utc'])
			comment['link_id'] = row['link_id']
			comment['parent_id'] = row['parent_id']
			comment['text'] = row['body']
			comment['author'] = row['author']

			#add to overall comment dict
			comment_id = row['name']		#post id with t1_ prefix
			month_comments[comment_id] = comment

		#add month comments to overall list
		comments.update(month_comments)
		scanned_count += len(month_comments_df)
		vprint("Found %d (of %d) relevant comments in %s" % (len(month_comments), len(month_comments_df), file))

	vprint("Total of %d comments for %d-%d posts (of %d scanned)" % (len(comments), post_month, post_year, scanned_count))
	return comments
#end load_comments


#given a subreddit, post month-year, dict of posts and dict of relevant comments, 
#reconstruct the post/comment (cascade) structure
#store cascades in the following way using a dictionary
#	post id -> post object
# 	post/comment replies field -> list of direct replies
#	post/comment time field -> create time of object as utc timestamp
#posts also have comment_count_total and comment_count_direct 
def build_cascades(subreddit, month, year, posts, comments):

	vprint("Extracting post/comment structure for %d %s %d-%d posts and %d comments" % (len(posts), subreddit, month, year, len(comments)))

	#create dictionary of post id -> new post object to store cascades
	cascades = {key:{'time':value['time'], 'replies':list(), 'comment_count_direct':0, 'comment_count_total':0} for key, value in posts.items()}

	#and corresponding comments dictionary
	cascade_comments = {key:{'time':value['time'], 'replies':list(), 'id':key} for key, value in comments.items()}

	#now that we can find posts and comments at will, let's build the cascades dictionary!
	
	#loop all comments, assign to immediate parent and increment comment_count of post parent
	fail_cascades = set()		#keep set of post ids with missing comments, throw out at the end
	for comment_id, comment in comments.items():

		#get immediate parent (post or comment)
		direct_parent = comment['parent_id']
		direct_parent_type = "post" if direct_parent[:2] == "t3" else "comment"
		#get post parent
		post_parent = comment['link_id']

		#add this comment to replies list of immediate parent, and update counters on post_parent
		try:
			#update post parent
			#if post parent missing, FAIL
			if post_parent not in cascades:
				fail_cascades.add(post_parent)
				continue
			#update overall post comment count for this new comment
			cascades[post_parent]['comment_count_total'] += 1

			#now handle direct parent, post or comment
			#parent is post
			if direct_parent_type == "post":
				#missing post, FAIL
				if direct_parent not in cascades:
					fail_cascades.add(post_parent)
					continue
				#add this comment to replies field of post (no total comment increment, done above)
				cascades[direct_parent]['replies'].append(cascade_comments[comment_id])
				#add 1 to direct comment count field
				cascades[direct_parent]['comment_count_direct'] += 1

			#parent is comment
			else:	
				#missing comment, FAIL
				if direct_parent not in cascade_comments:
					fail_cascades.add(post_parent)
					continue
				#add current comment to replies field of parent comment
				cascade_comments[direct_parent]['replies'].append(cascade_comments[comment_id])
		except:
			print("Something went very wrong, exiting")
			exit(0)

	#delete failed cascades with missing comments and/or post
	for key in fail_cascades:
		cascades.pop(key, None)
	#and delete all keys that point to these
	fail_comments = set([comment_id for comment_id, comment in comments.items() if comment['link_id'] in fail_cascades or comment['parent_id'] in fail_cascades])
	for key in fail_comments:
		cascade_comments.pop(key, None)

	vprint("Built %d cascades with %d comments" % (len(cascades), len(cascade_comments)))
	vprint("   Removed %d incomplete cascades (%d associated comments)" % (len(fail_cascades), len(fail_comments)))	

	#save cascades for later loading
	file_utils.save_pickle(cascades, cascades_filepath % (subreddit, subreddit, year, month))

	return cascades
#end build_cascades


#BOOKMARK - haven't done anything below this


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