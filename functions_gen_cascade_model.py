#functions for gen_cascade_model.py - offloading and modularizing all the things

import file_utils
import sim_tree
import tree_edit_distance
import fit_cascade_gen_model

from shutil import copyfile
import subprocess
import os
import random
from argparse import *
import pandas as pd
import string
import glob
import re
from collections import defaultdict
import itertools
import bisect
from copy import deepcopy
import statistics
import networkx as nx
from nltk.corpus import stopwords
import math
import sys

#filepaths of data and pre-processed files - keeping everything in the same spot, for sanity/simplicity

#raw posts for (sub, sub, year, month)
raw_posts_filepath = "/data/datasets/reddit_discussions/%s/%s_submissions_%d_%d.tsv"	
#raw comments for (sub, sub, post year, comment year, comment month)
raw_comments_filepath = "/data/datasets/reddit_discussions/%s/%s_%sdiscussions_comments_%s_%s.tsv"  
#processed posts for (sub, sub, year, month) - dictionary of post id -> post containing title tokens, author, created utc
processed_posts_filepath = "reddit_data/%s/%s_processed_posts_%d_%d.pkl"
#fitted params for posts for (sub, sub, year, month) - dictionary of post id -> params tuple
fitted_params_filepath = "reddit_data/%s/%s_post_params_%d_%d.pkl"
#reconstructed cascades for (sub, sub, year, month) - dictionary of post id -> cascade dict, with "time", "num_comments", and "replies", where "replies" is nested list of reply objects
cascades_filepath = "reddit_data/%s/%s_cascades_%d_%d.pkl"

#filepath for random test samples, determined by subreddit, subreddit, number of posts, testing start (year-month), testing length, and tree size requirements
#(save these to files so you can have repeated runs of the same random set)
random_sample_list_filepath = "reddit_data/%s/%s_%d_test_keys_list_start%d-%d_%dmonths_filter%s-%s.pkl"

#filepath for cached base graph builds
#determined by: subreddit, training_start_year, training_start_month, training_len, 
#	include_default_posts, max_nodes, min_node_quality, weight_method, min_weight, top_n,
#   graph_downsample_ratio, large_cascade_demarcation, and remove_stopwords
#(yes, it's a mess)
base_graph_filepath = "reddit_data/%s/base_graph_%d-%dtest_start_%dtrainposts_default_posts_%s_%snodes_%.1fminquality_%s_%.1fminedgeweight_%dtopn_%ssample%s%s.pkl"

#filepaths of output/temporary files - used to pass graph to C++ node2vec for processing
temp_graph_filepath = "sim_files/graph_%s.txt"			#updated graph for this sim run
temp_params_filepath = "sim_files/in_params_%s.txt"		#temporary, filtered params for sim run (if sampled graph)
output_params_filepath = "sim_files/out_params_%s.txt"		#output params from node2vec

#output filepaths
stats_filepath = "sim_results/post_set_stats_%s_test%d-%d_%d_%s_posts.csv"		#post set stats (subreddit, year, month, num_posts, type)

#hardcoded params for failed fit cascades
#only used when fit/estimation fails and these posts are still included in graph

DEFAULT_WEIBULL_NONE = [1, 1, 0.15]     #weibull param results if post has NO comments to fit
                                        #force a distribution heavily weighted towards the left, then decreasing

DEFAULT_WEIBULL_SINGLE = [1, 2, 0.75]   #weibull param result if post has ONE comment and other fit methods fail
                                        #force a distribution heavily weighted towards the left, then decreasing
                   #use this same hardcode for other fit failures, but set a (index 0) equal to the number of replies

DEFAULT_LOGNORMAL = [0.15, 1.5]    	#lognormal param results if post has no comment replies to fit
                                	#mu = 0, sigma = 1.5 should allow for occasional comment replies, but not many

DEFAULT_QUALITY = 0.45     #default param quality if hardcode params are used


#parse out all command line arguments and return results
def parse_command_args():
	#arg parser
	parser = ArgumentParser(description="Simulate reddit cascades from partially-observed posts.")

	#required arguments (still with -flags, because clearer that way, and don't want to impose an order)
	parser.add_argument("-s", "--sub", dest="subreddit", required=True, help="subreddit to process")
	parser.add_argument("-o", "--out", dest="outfile", required=True, help="base output filename")
	#must pick one of four processing options: a single id, random, all, or sample of size n
	proc_group = parser.add_mutually_exclusive_group(required=True)
	proc_group.add_argument("-id", dest="sim_post", default=None,  help="post id for single-processing")
	proc_group.add_argument("-r", "--rand", dest="sim_post", action="store_const", const="random", help="choose a random post from the subreddit to simulate")
	proc_group.add_argument("-a", "--all", dest="sim_post", action="store_const", const="all", help="simulate all posts in the subreddit")
	proc_group.add_argument("-n", "--n_sample", dest="sim_post", default=None, help="number of posts to test, taken as first n posts in the testing period")
	#must pick an edge weight computation method: cosine (based on tf-idf) or jaccard
	weight_group = parser.add_mutually_exclusive_group(required=True)
	weight_group.add_argument("-j", "--jaccard", dest="weight_method", action='store_const', const="jaccard", help="compute edge weight between pairs using jaccard index")
	weight_group.add_argument("-c", "--cosine", dest="weight_method", action='store_const', const="cosine", help="compute edge weight between pairs using tf-idf and cosine similarity")
	weight_group.add_argument("-wmd", "--word_mover", dest="weight_method", action='store_const', const="word_mover", help="compute edge weight between pairs using GloVe embeddings and word-mover distance")
	#if running jaccard, can elect to remove stopwords
	parser.add_argument("-stopwords", dest="remove_stopwords", action="store_true", help="remove stopwords before computing edge weight")
	parser.set_defaults(remove_stopwords=False)
	#must pick an edge limit method: top n edges per node, or weight threshold, or both
	parser.add_argument("-topn", dest="top_n", default=False, metavar=('<max edges per node>'), help="limit post graph to n edges per node")
	parser.add_argument("-threshold", dest="weight_threshold", default=False, metavar=('<minimum edge weight>'), help="limit post graph to edges with weight above threshold")
	#must select an observation type (time or comments) and provide at least one time/count
	observed_group = parser.add_mutually_exclusive_group(required=True)
	observed_group.add_argument("-t", dest="time_observed", default=False, help="time of post observation, in hours", nargs='+')
	observed_group.add_argument("-nco", "--num_comments_observed", dest="num_comments_observed", default=False, help="number of comments observed", nargs='+')

	#must provide year and month for start of testing data set - unless running off crypto, cve, or cyber
	parser.add_argument("-y", "--year", dest="testing_start_year", default=False, help="year to use for test set")
	parser.add_argument("-m", "--month", dest="testing_start_month", default=False, help="month to use for test set")

	#optional args
	parser.add_argument("-g", "--graph", dest="max_nodes", default=False, help="max nodes in post graph for parameter infer")
	parser.add_argument("-q", "--qual", dest="min_node_quality", default=False, help="minimum node quality for post graph")
	parser.add_argument("-e", "--esp", dest="estimate_initial_params", action='store_true', help="estimate initial params as inverse quality weighted average of neighbor nodes")
	parser.set_defaults(estimate_initial_params=False)
	parser.add_argument("-n_train", dest="training_num", default=10000, help="number of posts to use for training (immediately preceding test month")
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose output")
	parser.set_defaults(verbose=False)
	parser.add_argument("-d", "--default_params", dest="include_default_posts", action='store_true', help="include posts with hardcoded default parameters in infer graph")
	parser.set_defaults(include_default_posts=False)
	parser.add_argument("--err", dest="time_error_margin", default=False, help="allowable time error for evaluation, in minutes")
	parser.add_argument("--err_abs", dest="time_error_absolute", action="store_true", help="use absolute time error, instead of increasing by level")
	parser.set_defaults(time_error_absolute=False)
	parser.add_argument("--topo_err", dest="topological_error", action="store_true", help="compute topological error only, ignoring comment timestamps")
	parser.set_defaults(topological_error=False)
	parser.add_argument("-np", "--norm_param", dest="normalize_parameters", default=None, help="normalize paramters for graph inference, either mm or ln")
	parser.add_argument("--sanity", dest="sanity_check", action="store_true", help="sanity check: simulate from fitted params instead of inferring")
	parser.set_defaults(sanity_check=False)
	#can also layer in a size filter: only simulate cascades within a size range (or meeting some min/max size)
	#(filter applied before sample/rand)
	parser.add_argument("-min", "--min_size", dest="min_size", default=None, help="minimum cascade size for simulation test set")
	parser.add_argument("-max", "--max_size", dest="max_size", default=None, help="maximum cascade size for simulation test set (exclusive)")
	parser.add_argument("--train_stats", dest="training_stats", action="store_true", help="output statistics for training set")
	parser.set_defaults(training_stats=False)
	parser.add_argument("--test_stats", dest="testing_stats", action="store_true", help="output statistics for testing set")
	parser.set_defaults(testing_stats=False)
	#optional graph downsampling - can specify a ratio of large:small posts for base graph,
	#and the number of comments required to be considered a "large" post
	parser.add_argument("-down_ratio", dest="graph_downsample_ratio", default=None, help="ratio of large:small posts to use for base graph build")
	parser.add_argument("-large_req", dest="large_cascade_demarcation", default=None, help="minimum number of comments to be considered \"large\" for graph downsample")

	#just build the base graph, don't bother with any actual testing
	parser.add_argument('-preprocess', dest='preprocess', action="store_true", help="preprocess posts, construct and save base graph, no actual testing")
	parser.set_defaults(preprocess=False)

	args = parser.parse_args()		#parse the args (magic!)

	#must provide year and month for start of testing data set - unless running off crypto, cve, or cyber
	if not (args.testing_start_month and args.testing_start_year) and args.subreddit != "cve" and args.subreddit != "crypto" and args.subreddit != "cyber":
		parser.error('Must specify year and month for start of testing period, -m and -y')

	#make sure at least one edge-limit option was chosen
	if not (args.top_n or args.weight_threshold):
		parser.error('No edge limit selected, add -topn, -threshold, or both')

	#make sure error settings don't conflict
	if args.topological_error and (args.time_error_absolute or args.time_error_margin != False):
		parser.error('Cannot use topological error method with absolute time error or error margin setting')

	#if downsampling base graph, must specify both ratio and large size
	if (args.graph_downsample_ratio is not None and args.large_cascade_demarcation is None) or (args.graph_downsample_ratio is None and args.large_cascade_demarcation is not None):
		parser.error('Must specify both downsample ratio and large cascade size for graph build downsampling.')

	#only remove stopwords if running in jaccard mode
	if args.remove_stopwords and args.weight_method != "jaccard":
		parser.error("Can only remove stopwords when using jaccard for edge weight.")

	#must choose ln or minmax for normalization
	if args.normalize_parameters is not None and args.normalize_parameters != "ln" and args.normalize_parameters != "mm":
		parser.error("Must choose from ln (natural log) or mm (min-max) normalization options")

	#extract arguments (since want to return individual variables)
	subreddit = args.subreddit
	sim_post = args.sim_post
	if args.time_observed != False:
		observed_list = [float(time) for time in args.time_observed]
		observing_time = True
	elif args.num_comments_observed != False:
		observed_list = [int(count) for count in args.num_comments_observed]
		observing_time = False
	outfile = args.outfile
	max_nodes = args.max_nodes if args.max_nodes == False else int(args.max_nodes)
	min_node_quality = args.min_node_quality if args.min_node_quality == False else float(args.min_node_quality)
	estimate_initial_params = args.estimate_initial_params
	testing_start_month = int(args.testing_start_month)
	testing_start_year = int(args.testing_start_year)
	training_num = int(args.training_num)
	weight_method = args.weight_method
	remove_stopwords = args.remove_stopwords
	include_default_posts = args.include_default_posts
	verbose = args.verbose
	top_n = args.top_n
	time_error_margin = float(args.time_error_margin) if args.time_error_margin != False else 30.0
	time_error_absolute = args.time_error_absolute
	topological_error = args.topological_error
	normalize_parameters = args.normalize_parameters
	sanity_check = args.sanity_check
	min_size = int(args.min_size) if args.min_size is not None else None
	max_size = int(args.max_size) if args.max_size is not None else None
	training_stats = args.training_stats
	testing_stats = args.testing_stats
	preprocess = args.preprocess
	graph_downsample_ratio = float(args.graph_downsample_ratio) if args.graph_downsample_ratio is not None else None
	large_cascade_demarcation = int(args.large_cascade_demarcation) if args.large_cascade_demarcation is not None else None
	#convert where required
	if top_n != False:
		top_n = int(top_n)
	weight_threshold = args.weight_threshold
	if weight_threshold != False:
		weight_threshold = float(weight_threshold)
	#extra flags/variables for different processing modes
	sample_num = False
	if sim_post == "all":
		batch = True
		testing_num = False
	elif sim_post == "random":
		batch = False
		testing_num = False
	else:
		#is this a number (n_sample), or an id (single post)?
		try:
			#number of posts to sample
			int(sim_post)
			batch = True
			testing_num = int(sim_post)
			sim_post = "subset"
		except ValueError:
			#single specified post
			batch = False
			testing_num = False
	if len(observed_list) > 1:
		batch = True
	#and for eval mode
	if topological_error:
		error_method = "topo"
	elif time_error_absolute:
		error_method = "abs"
	else:	#both false, use by-level
		error_method = "level"
	#no training stats if doing sanity check
	if training_stats and sanity_check:
		print("No training data loaded for sanity check mode - cannot output training data stats.")
		training_stats = False

	#if running socsim data, set that flag
	if subreddit == "cve" or subreddit == "cyber" or subreddit == "crypto":
		socsim_data = True
		#and set counts to something for when it shows up in filenames
		training_num = 0
		testing_num = 0
	else:
		socsim_data = False

	#hackery: declare a special print function for verbose output
	define_vprint(verbose)

	#print all arguments, so we know what this run was
	vprint("\n", args, "\n")	

	#print some log-ish stuff in case output being piped and saved
	vprint("Sim Post: ", sim_post, " %d" % sample_num if sample_num != False else "")
	if observing_time:
		vprint("Time Observed: ", observed_list)
	else:
		vprint("Comments Observed: ", observed_list)
	vprint("Output: ", outfile)
	vprint("Source subreddit: ", subreddit)
	vprint("Minimum node quality: ", min_node_quality)
	vprint("Max graph size: ", max_nodes)
	vprint("Max edges per node: ", "None" if top_n==False else top_n)
	vprint("Minimum edge weight: ", "None" if weight_threshold==False else weight_threshold)
	if estimate_initial_params:
		vprint("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
	if normalize_parameters is not None:
		vprint("Normalizing params for graph infer using ", ("min-max" if normalize_parameters == "mm" else "natural log"))
	else:
		vprint("No param normalization for infer step")
	vprint("Test Set: first %d posts starting at %d-%d" % (testing_num, testing_start_month, testing_start_year))
	vprint("Training Set: %d posts immediately preceding %d-%d" % (training_num, testing_start_month, testing_start_year))
	if weight_method == "jaccard":
		vprint("Using Jaccard index to compute graph edge weights")
		if remove_stopwords:
			vprint("   Removing stopwords for edge weight calculation")
	elif weight_method == "cosine":
		vprint("Using tf-idf and cosine similarity to compute graph edge weights")
	else:  #word_mover
		vprint("Using GloVe embeddings and word-mover distance to compute graph edge weights")
	if include_default_posts:
		vprint("Including posts with hardcoded default parameters")
	else:
		vprint("Ignoring posts with hardcoded default parameters")
	if error_method == "abs":
		vprint("Using absolute time error margin for all levels of tree")
		vprint("   Allowable eval time error: ", time_error_margin)
	elif error_method == "topo":
		vprint("Using topological error for tree evaluation (ignoring comment times)")
	else:
		vprint("Using error margin increasing by level")
		vprint("   Allowable eval time error: ", time_error_margin)
	if min_size is not None and max_size is not None:
		vprint("Only simulating cascades with true size between %d and %d comments (inclusive)" % (min_size, max_size))
	elif min_size is not None:
		vprint("Only simulating cascades with true size greater than or equal to %d" % min_size)
	elif max_size is not None:
		vprint("Only simulating cascades with true size less than or equal to %d" % max_size)
	if sanity_check:
		vprint("Simulating from fitted params, skipping graph/infer/refine steps")
	if graph_downsample_ratio is not None:
		vprint("Downsampling base graph to %.1f:1 large:small posts, where large posts contain at least %d comments" % (graph_downsample_ratio, large_cascade_demarcation))
	vprint("")

	#return all arguments
	return subreddit, sim_post, observing_time, observed_list, outfile, max_nodes, min_node_quality, estimate_initial_params, normalize_parameters, batch, testing_num, testing_start_month, testing_start_year, training_num, weight_method, remove_stopwords, top_n, weight_threshold, include_default_posts, time_error_margin, error_method, sanity_check, min_size, max_size, training_stats, testing_stats, socsim_data, graph_downsample_ratio, large_cascade_demarcation, verbose, preprocess
#end parse_command_args


#helper function: calling this will define vprint based on the current verbosity setting
def define_vprint(verbose):
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
#end define_vprint


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


#given a subreddit, starting month-year, and number of posts to load, load processed posts
#	if load_forward=False, do not load the starting month, start loading on the previous
#if params = True, also load fitted params for these posts
#if cascades = True, also load reconstructed cascades for these posts
#if files don't exist, call methods to perform preprocessing
#load as many months as required to reach the num_posts given
#	if load_forward=True, load months following the given if need more posts
#	if load_forward=False, load months backwards from the given if need more posts
def load_processed_posts(subreddit, start_month, start_year, num_posts, load_params=False, load_cascades=False, load_forward=True):
	posts = {}
	params = {}
	failed_fit_posts = []
	cascades = {}

	#do we want to load the start month? or start with the preceding?
	if load_forward == False:
		start_month, start_year = monthdelta(start_month, start_year, -1)

	#loop load until reach required number of posts (each loop will load a single month)
	#if running in random, id, or all mode, load just one month
	m = 0
	while (len(posts) < num_posts and num_posts != False) or len(posts) == 0:
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
				params_data = file_utils.load_pickle(fitted_params_filepath % (subreddit, subreddit, year, month))
			#if params file doesn't exist, create it - loading in the process
			else:
				vprint("   Fitted params file doesn't exist, creating now")
				params_data = fit_posts(subreddit, month, year, month_posts)
			#extract successfully fitted params and list of failed posts
			params.update(params_data['params_dict'])
			failed_fit_posts.extend(params_data['failed_fit_list'])

		#cascades, if desired
		if load_cascades:
			#get cascades for this month
			cascades.update(get_cascades(subreddit, month, year, month_posts))

		#add this month's posts to overall
		posts.update(month_posts)
		#increment/decrement month counter to load correct "next" month based on direction
		if load_forward: m += 1
		else: m += -1

	#throw out posts that we don't have cascades for - incomplete or some other issue
	if load_cascades and len(posts) != len(cascades):
		posts, deleted_count = filter_dict_by_list(posts, list(cascades.keys()), num_deleted=True)
		vprint("Deleted %d posts without cascades" % deleted_count)

	#throw out posts that we don't have params (or fail note) for - neg comment times or some other issue
	if load_params and len(posts) != len(params)+len(failed_fit_posts):
		posts, deleted_count = filter_dict_by_list(posts, list(params.keys())+failed_fit_posts, num_deleted=True)
		vprint("Deleted %d posts without params" % deleted_count)

	vprint("Loaded %d posts " % len(posts))
	if load_cascades: vprint("   %d cascades" % len(cascades))
	if load_params: vprint("   %d fitted params, %d failed fit" % (len(params), len(failed_fit_posts)))

	#return results
	if load_params and load_cascades: return posts, cascades, params, failed_fit_posts
	if load_params: return posts, params, failed_fit_posts
	if load_cascades: return posts, cascades
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

	#convert to our nested dictionary structure
	posts = {}
	for index, row in posts_df.iterrows():
		#if author or title has been deleted, skip
		if row['author'] == "[deleted]" or row['title'] == "[deleted]":
			continue

		#check for good row, fail and error if something is amiss (probably a non-quoted body)
		if pd.isnull(row['title']) or pd.isnull(row['subreddit']) or pd.isnull(row['created_utc']) or pd.isnull(row['author']):
			print("Invalid post, exiting\n", row)
			exit(0)

		#build new post dict
		post = {}
		post['tokens'] = extract_tokens(row['title'])
		if post['tokens'] == False:
			print(row)
			exit(0)
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
	punctuations.extend(['—', '“', '”'])	#kill these too
	
	if text != None:
		try:
			tokens = [word.lower() for word in text.split()]	#tokenize and normalize (to lower)		
			tokens = [word.strip("".join(punctuations)) for word in tokens]		#strip trailing and leading punctuation
			#remove punctuation-only tokens and empty strings
			tokens = [word for word in tokens if word != '' and word not in punctuations and word != '']		
		except Exception as e:
			print("Token extraction fail")
			print(e)
			print(text)
			return False
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
	#get cascades for this month
	cascades = get_cascades(subreddit, month, year, posts)

	#fit parameters to each cascade
	vprint("Fitting %d cascades for %s %d-%d" % (len(cascades), subreddit, month, year))

	params_out = fit_posts_from_cascades(cascades)

	file_utils.save_pickle(params_out, fitted_params_filepath % (subreddit, subreddit, year, month))
	
	return params_out		#return params + fail list in dict
#end fit_posts


#given a set of cascades, fit parameters for these posts and return result
def fit_posts_from_cascades(cascades):
	cascade_params = {}		#build dict of post id -> 6 fitted params + quality
	failed_fit_posts = []	#list of posts that failed to fit
	post_count = 0
	fail_count = 0		#count of posts with failed param fit
	fail_size = 0
	succeed_count = 0
	succeed_size = 0
	partial_count = 0
	partial_size = 0
	partial_weibull_fail = 0
	neg_comment_times_count = 0

	#loop and fit cascades
	for post_id, post in cascades.items():		
		param_res = fit_cascade_gen_model.fit_params(post)	#fit the current cascade 

		#if negative comment times, skip this cascade and move to next
		if param_res == False: 
			neg_comment_times_count += 1
			continue

		#if both weibull and lognorm fit failed, increment fail count and add post_id to fail list
		if not any(param_res):		#entire param array False
			fail_count += 1
			fail_size += post['comment_count_total']
			failed_fit_posts.append(post_id)
		#fit succeeded to some degree, add params to success dictionary
		else:	#some True		
			#both fits succeeded
			if all(param_res):
				succeed_count += 1
				succeed_size += post['comment_count_total']
			#partial success
			else:
				partial_count += 1
				partial_size += post['comment_count_total']
				if param_res[0] == False:
					partial_weibull_fail += 1
			#always store params (will handle the False values later)
			cascade_params[post_id] = param_res		

		post_count += 1
		if post_count % 2500 == 0:
			vprint("Fitted %d cascades (%d failed)" % (post_count, fail_count))

	#dump params to file
	vprint("Fitted params for a total of %d cascades" % len(cascade_params))	
	vprint("   skipped %d cascades with negative comment times" % neg_comment_times_count)
	vprint("   %d cascades failed fit process" % fail_count)	
	if fail_count != 0:
		vprint("   fail average cascade size: %d" % (fail_size/fail_count))
	vprint("   %d cascades succeeded fit process" % succeed_count)
	if succeed_count != 0:
		vprint("   succeed average cascade size: %d" % (succeed_size/(succeed_count)))
	vprint("   %d cascades partially-succeeded fit process (some missing params)" % partial_count)
	vprint("      %d cascades failed weibull fit" % partial_weibull_fail)
	if partial_count != 0:
		vprint("   partial-succeed average cascade size: %d" % (partial_size/(partial_count)))

	#wrap both fitted params and list of failed fits in a dictionary
	params_out = {"params_dict": cascade_params, "failed_fit_list": failed_fit_posts}

	return params_out
#end fit_posts_from_cascades	


#given subreddit, month, year and loaded posts, load cascades associated with those posts
#if cascades don't exist, build them
#returns cascades dictionary
def get_cascades(subreddit, month, year, posts):
	#if reconstructed cascades already exist, load those
	if file_utils.verify_file(cascades_filepath % (subreddit, subreddit, year, month)):
		cascades = file_utils.load_pickle(cascades_filepath % (subreddit, subreddit, year, month))		
	#otherwise, reconstruct the cascades (get them back as return value)
	else:
		vprint("Reconstructing cascades for %s %d-%d" % (subreddit, month, year))
		#load comments associated with this month of posts
		comments = load_comments(subreddit, month, year, posts)
		#reconstruct the cascades
		cascades = build_and_save_cascades(subreddit, month, year, posts, comments)

	return cascades
#end get_cascades


#filter one dictionary based on list of keys
#returns new, filtered dictionary
#if num_deleted=True, also return number of items removed
def filter_dict_by_list(dict_to_filter, keep_list, num_deleted=False):
	#new dictionary containing only the keys we care about
	updated_dict = { key: dict_to_filter[key] for key in keep_list if key in dict_to_filter }

	if num_deleted: return updated_dict, len(dict_to_filter) - len(updated_dict)
	return updated_dict
#end filter_dict_by_list


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
		vprint("Loading comments from %s" % file)
		month_comments_df = pd.read_csv(file, sep='\t', engine='python')

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
		vprint("Found %d (of %d) relevant comments" % (len(month_comments), len(month_comments_df)))

	vprint("Total of %d comments for %d-%d posts (of %d scanned)" % (len(comments), post_month, post_year, scanned_count))
	return comments
#end load_comments


#given a subreddit, post month-year, dict of posts and dict of relevant comments, 
#reconstruct the post/comment (cascade) structure
#heavy-lifting done in build_cascades, this just handles the load/save
def build_and_save_cascades(subreddit, month, year, posts, comments):
	vprint("Extracting post/comment structure for %d %s %d-%d posts and %d comments" % (len(posts), subreddit, month, year, len(comments)))

	cascades = build_cascades(posts, comments)

	#save cascades for later loading
	file_utils.save_pickle(cascades, cascades_filepath % (subreddit, subreddit, year, month))

	return cascades
#end build_and_save_cascades


#given a dict of posts and dict of relevant comments, 
#reconstruct the post/comment (cascade) structure
#store cascades in the following way using a dictionary
#	post id -> post object
# 	post/comment replies field -> list of direct replies
#	post/comment time field -> create time of object as utc timestamp
#posts also have comment_count_total and comment_count_direct 
def build_cascades(posts, comments):

	#create dictionary of post id -> new post object to store cascades
	cascades = {key:{'id':key, 'time':value['time'], 'replies':list(), 'comment_count_direct':0, 'comment_count_total':0} for key, value in posts.items()}

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
			#if comment time is before post time, FAIL
			if comment['time'] < cascades[post_parent]['time']:
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
				#if comment time is before post time, FAIL
				if comment['time'] < cascades[direct_parent]['time']:
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
				#if comment time is before parent comment time, FAIL
				if comment['time'] < cascade_comments[direct_parent]['time']:
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
	vprint("   Removed %d incomplete/erroneous cascades (%d associated comments)" % (len(fail_cascades), len(fail_comments)))	

	return cascades
#end build_cascades


#given a dictionary of posts, a desired sample size, and a flag indicating direction,
#return a chronological sample of n posts
#if forward=True, take the first n posts sorted by time
#if forward=False, take the last n posts sorted by time
def sample_chronologically(posts, sample_num, forward=True):
	#get list of (post id, time) tuples sorted by time
	sorted_posts = sorted([(key, value['time']) for key, value in posts.items()], key=lambda x: x[1])

	#if forward is true, take just the first n of them
	if forward: sorted_posts = sorted_posts[:sample_num]
	#otherwise, take the last n of them
	else: sorted_posts = sorted_posts[-1 * sample_num:]

	#return as list of keys
	return [i[0] for i in sorted_posts]
#end sample_chronologically


#given a post id, boolean mode flags, and dictionary of posts, ensure post is in this set
#if running in sample mode, pick the sample as the first n posts chronologically
#returns modified post dictionary that contains only the posts to be tested
def get_test_post_set(input_sim_post, min_size, max_size, sample_num, posts, cascades, subreddit, testing_start_month, testing_start_year):

	#if processing all posts in test set, return list of ids
	if input_sim_post == "all":		
		vprint("Processing all %d posts in test set" % len(posts))

	#if random post id, pick an id from loaded posts
	elif input_sim_post == "random":
		rand_sim_post_id = random.choice(list(posts.keys()))
		posts = {rand_sim_post_id: posts[rand_sim_post_id]}
		vprint("Choosing random simulation post: %s" % rand_sim_post_id)

	#if sampling, first n posts chronologically
	elif sample_num != False:
		vprint("Sampling %d first posts (from %d) for simulation set" % (sample_num, len(posts.keys())))
		#get list of first n keys
		keys = sample_chronologically(posts, sample_num)
		#filter posts to match corresponding keys list
		posts = filter_dict_by_list(posts, keys)

	#if single id, make sure given post id is in the dataset
	else:
		#if given not in set, exit
		if input_sim_post not in posts:
			print("Given post id not in group set - exiting.\n")
			exit(0)
		posts = {input_sim_post: posts[input_sim_post]}
		vprint("Using input post id: %s" % input_sim_post)

	#apply size filters last - assuming they're being used to break up a run
	if min_size != None:
		keys = [post_id for post_id in cascades if cascades[post_id]['comment_count_total'] >= min_size]
		posts = filter_dict_by_list(posts, keys)
		vprint("Filtered to %d posts with >= %d comments" % (len(posts), min_size))
	if max_size != None:
		keys = [post_id for post_id in cascades if cascades[post_id]['comment_count_total'] < max_size]
		posts = filter_dict_by_list(posts, keys)
		vprint("Filtered to %d posts with <= %d comments" % (len(posts), max_size))	

	return posts
#end get_test_post_set


#for a given set of processed posts and reconstructed cascades, 
#compute and output some stats on the post set
def output_post_set_stats(cascades, subreddit, year, month, set_type, num_posts):
	curr_filepath = stats_filepath % (subreddit, year, month, len(cascades), set_type)

	#do we already have a stats file for this subreddit set? if so, skip
	if file_utils.verify_file(curr_filepath):
		vprint("Training data stats already exist.")
		return

	#cascade size distribution
	cascade_sizes = defaultdict(int)
	for post_id, cascade in cascades.items():
		cascade_sizes[cascade['comment_count_total']] += 1

	#build dict of post id -> list of sorted relative comment times
	post_to_comment_times = {}
	#and dict of lifetime distribution (binned by 15 minutes)
	lifetime_dist = defaultdict(int)
	#loop cascades
	for post_id, cascade in cascades.items():
		comment_times = get_list_of_comment_times(cascade)
		post_to_comment_times[post_id] = comment_times
		if len(comment_times) == 0:
			lifetime_dist[0] += 1
		else:
			lifetime_dist[int(comment_times[-1] // 15) * 15] += 1

	#% of lifetime vs % of comments observed
	#collect mean, and median

	lifetime_percents = [x * 2.5 for x in range(0, 41)]		#do every 2.5% for now
	observed_percents = defaultdict(list)
	#loop cascades, build list of percent of comments seen at each lifetime percentage
	#only considers cascades with at least 10 comments, since the small ones muck up the plot
	for post_id, comment_times in post_to_comment_times.items():
		if len(comment_times) < 10:
			continue
		lifetime_comment_counts = get_lifetime_percent_comment_counts(comment_times, lifetime_percents)
		cascade_comment_count = len(comment_times)
		for percent, count in lifetime_comment_counts.items():
			observed_percents[percent].append(count / cascade_comment_count)
	#get mean, median
	mean_observed = {}
	median_observed = {}
	for percent in observed_percents.keys():
		mean_observed[percent] = sum(observed_percents[percent]) / len(observed_percents[percent])
		median_observed[percent] = statistics.median(observed_percents[percent])

	#write all to output
	file_utils.multi_dict_to_csv(stats_filepath % (subreddit, year, month, len(cascades), set_type), ["number_of_comments", "number_of_cascades", "lifetime(minutes)", "number_of_cascades", "percent_lifetime", "mean_comments_observed", "percent_lifetime", "median_comments_observed"], [cascade_sizes, lifetime_dist, mean_observed, median_observed])
#end output_post_set_stats


#given a single cascade, with comment times already shifted and converted to minutes,
#return a single list of ALL comment times
def get_list_of_comment_times(cascade):
	comment_times = []		#list of all comment times

	#init queue to root, will process children as nodes are removed from queue
	nodes_to_visit = [cascade]
	while len(nodes_to_visit) != 0:
		parent = nodes_to_visit.pop(0)    #grab current comment/node
		#add all reply times to set of cascade comments, and add child nodes to queue
		for comment in parent['replies']:
			comment_times.append(comment['time'])	#add time (in minutes) to list
			nodes_to_visit.append(comment)    #add reply to processing queue

	#return sorted list of times
	return sorted(comment_times)
#end get_list_of_comment_times


#given a list of sorted comment times for a single cascade and a list of lifetime percentages,
#count how many comments have been observed at each percentage through the lifetime
def get_lifetime_percent_comment_counts(comment_times, lifetime_percents):
	#if post has no comment, return 0 for all percents
	if len(comment_times) == 0:
		return {percent: 0 for percent in lifetime_percents}

	lifetime = comment_times[-1]		#lifetime is time of last comment

	#convert lifetime percentages from whole numbers to times in minutes
	time_percents = [percent/100.0 * lifetime for percent in lifetime_percents]

	#build dict of lifetime percent -> # of comments observed up to and including that time
	percent_counts = {}
	for percent, percent_time in zip(lifetime_percents, time_percents):
		percent_counts[percent] = len([time for time in comment_times if time <= percent_time])

	return percent_counts
#end get_lifetime_percent_comment_counts


#tiny defaultdict declaration helper function, so we can pickle the dict of lists
def ddlist():
    return defaultdict(list)
#end ddlist


#given a set of processed posts, and graph build settings, "build" the post parameter graph
#but don't actually build it, just create and return an adjacency list
#(adjacency list may contain duplicate edges at this point, but we'll deal with that later)
#weight_method is string, one of 'jaccard', 'cosine', or 'word_mover'
#   if jaccard, use jaccard index to compute edge weight
#   if cosine, use tf-idf and cosine similarity
#   if word_mover, use GloVe embeddings and word_mover distance
#if top_n != False, take top_n highest weight edges for each node
#if min_weight != False, only include edges with weight > min_weight in final graph
#min_weight overrides top_n, such that some nodes may have fewer than top_n edges 
#		if weight requirement leaves fewer than top_n candidate edges
#if include_default_posts = True, include posts with hardcoded default params in graph (otherwise, leave out)
#if min_node_quality != False, only include nodes with param fit quality >= threshold in graph build
#include_default_posts overrides min_node_quality - all default posts thrown out regardless of default quality setting
def build_base_graph(cascades, posts, params, default_params_list, subreddit, testing_start_year, testing_start_month, training_num, include_default_posts, max_nodes, min_node_quality, weight_method, remove_stopwords, min_weight, top_n, graph_downsample_ratio, large_cascade_demarcation):
	#first, check if we've cached this graph before
	curr_filepath = base_graph_filepath % (subreddit, testing_start_year, testing_start_month, training_num, include_default_posts, (max_nodes if max_nodes != False else "all_"), min_node_quality, weight_method, min_weight, top_n, (("%.1f" % graph_downsample_ratio) if graph_downsample_ratio is not None else "no_"), (("_>=%d" % large_cascade_demarcation) if large_cascade_demarcation is not None else ""), ("_stopwords" if remove_stopwords else ""))
	#have graph, load and return
	if file_utils.verify_file(curr_filepath):
		vprint("Loading base graph from file")
		loaded_graph = file_utils.load_pickle(curr_filepath)		
		if 'unique_edges' in loaded_graph:
			vprint("Graph contains %d nodes and %d unique edges (%d edge entries)" % (len(loaded_graph['graph']), loaded_graph['unique_edges'], loaded_graph['edge_entries']))
		else:
			vprint("Graph contains %d nodes" % len(loaded_graph['graph']))
		#return edgelist (may contain duplicates), and list of post ids considered for graph (may not all actually be in graph)
		return loaded_graph['graph'], loaded_graph['graph_post_ids']
	#no graph, build as usual, save at the end

	vprint("\nBuilding param graph for %d posts" % len(posts))
	
	#define post set to use for graph build based on options
	graph_post_ids = filter_post_set(params, default_params_list, min_node_quality, include_default_posts)	
	vprint("Using %d posts for graph" % len(graph_post_ids))

	
	#downsample post set for graph build further
	#take all large posts, and some number of small posts
	if graph_downsample_ratio is not None:
		#partition to large and small based on given comment requirement
		big_posts = [post_id for post_id in graph_post_ids if cascades[post_id]['comment_count_direct'] >= large_cascade_demarcation]
		small_posts = [post_id for post_id in graph_post_ids if cascades[post_id]['comment_count_direct'] < large_cascade_demarcation]
		#downsample if necessary, based on provided ratio
		if len(small_posts) > len(big_posts):
			small_posts = random.sample(small_posts, int(len(big_posts)/graph_downsample_ratio))
		graph_post_ids = big_posts + small_posts
		vprint("Downsampled to %d posts based on ratio and size" % len(graph_post_ids))
		vprint("   Using %d large posts with >= %d comments, and %d small posts" % (len(big_posts), large_cascade_demarcation, len(small_posts)))

	#do we need to sample the graph? if so, do it now
	#the graph may well end up smaller than the max_nodes limit (because probably not all connected)
	#but that's for the user to deal with!
	if max_nodes != False and len(graph_post_ids) >= max_nodes:
		vprint("\nSampling graph to %d nodes" % max_nodes)
		#sample down posts, true random sample
		graph_post_ids = set(random.sample(graph_post_ids, max_nodes-1))	#-1, leave room for sim_post

	#define stopwords list if we need them, and filter all the token sets
	if remove_stopwords:
		stop_words = set(stopwords.words('english'))
		for post_id in graph_post_ids:
			posts[post_id]['tokens'] = [w for w in posts[post_id]['tokens'] if not w in stop_words] 

	#chose edge computation method, store relevant function in variable for easy no-if calling later
	compute_edge_weight = get_edge_weight_method(weight_method)

	#build the multi-graph
	#	one node for each post
	#	edge of weight=1 connecting posts by the same user
	#	edge of weight=<computed val> between posts with common words
	#(but really only one edge, with sum of both weights)
	#store graph as adjacency list dictionary, where node -> dict['weights'],['neighbors'] -> sorted lists
	#   maintain two parallel sorted lists, one of edge weights and one of connected nodes
	#	keep weight list sorted to easily maintain the topn requirement, if required
	graph = defaultdict(ddlist)
	pair_count = 0

	#loop all post-pairs and determine weight of edge, if any, between them
	for post_pair in itertools.combinations(graph_post_ids, 2):		#pair contains ids of two posts
		#grab posts for easy access
		id_post_a = post_pair[0]
		id_post_b = post_pair[1]
		post_a = posts[id_post_a]
		post_b = posts[id_post_b]

		#compute edge weight based on post token sets (chose method earlier)
		weight = compute_edge_weight(post_a['tokens'], post_b['tokens'])
		if weight < min_weight:		#minimum token weight threshold, try to keep edge explosion to a minimum
			weight = 0

		#if posts have same author, add 1 to weight
		if post_a['author'] == post_b['author']:
			weight += 1

		#if edge weight is nonzero, add edge to graph
		if weight != 0:
			#add edge to adjacency list for both nodes
			add_edge(graph, id_post_a, id_post_b, weight)
			add_edge(graph, id_post_b, id_post_a, weight)

			#if top_n in effect, do we need to remove any too-small edges from these nodes?
			if top_n != False and len(graph[id_post_a]['weights']) > top_n:
				remove_low_edge(graph, id_post_a)
			if top_n != False and len(graph[id_post_b]['weights']) > top_n:
				remove_low_edge(graph, id_post_b)

		#progress prints
		pair_count += 1
		if pair_count % 1000000 == 0:
			vprint("   %d pairs" % pair_count)

	#verify top-n condition worked, get some stats
	edge_total = 0		#total number of edge entries (not unique edges)
	max_degree = 0
	min_degree = -1
	graph_edges = set()	#build a set of edges, each as tuple, order doesn't matter
	node_degrees = defaultdict(int)		#node id -> degree (from self top-n edges and others)
	for id_post_a, edges in graph.items():
		#add any unseen edges to edge set
		for id_post_b in edges['neighbors']:
			if (id_post_a, id_post_b) not in graph_edges and (id_post_b, id_post_a) not in graph_edges:
				graph_edges.add((id_post_a, id_post_b))		#add edge to set
				#update degree counts
				node_degrees[id_post_a] += 1
				node_degrees[id_post_b] += 1
		edge_total += len(edges['weights'])		#count edge entries
	#min/max degree stats
	for node, degree in node_degrees.items():
		if degree > max_degree:
			max_degree = degree
		if degree < min_degree or min_degree == -1:
			min_degree = degree

	vprint("Graph contains %d nodes and %d unique edges (%d edge entries)" % (len(graph), len(graph_edges), edge_total))
	vprint("  max degree: %d" % max_degree)
	vprint("  min degree: %d" % min_degree)

	#save graph before we return, bundled up in a dictionary
	vprint("Saving graph to %s"  % curr_filepath)
	save_data = {'graph': graph, 'graph_post_ids': graph_post_ids, 'unique_edges': len(graph_edges), 'edge_entries': edge_total}
	file_utils.save_pickle(save_data, curr_filepath)

	return graph, graph_post_ids 		#return edgelist (may contain duplicates), and list of post ids considered for graph (may not all actually be in graph)
#end build_graph


#graph build helper: filter list of posts by min_node_quality and include_default_posts
def filter_post_set(params, default_params_list, min_node_quality, include_default_posts):
	#define post set to use for graph build based on options
	graph_post_ids = list(params.keys())	#start with list of nodes we have fitted params for
	#filter these by min node quality, if given
	if min_node_quality != False:
		graph_post_ids = [post_id for post_id in graph_post_ids if params[post_id][6] >= min_node_quality]
		vprint("   %d/%d fitted posts meet quality threshold" % (len(graph_post_ids), len(params)))
	#include failed fit posts if specified and their default quality meets the threshold
	if (include_default_posts and min_node_quality != False and min_node_quality < DEFAULT_QUALITY) or (include_default_posts and min_node_quality == False):
		vprint("   Including posts with default parameters")
		graph_post_ids.extend(default_params_list)	
	return graph_post_ids
#end filter_post_set


#return correct edge weight computation function based on mode
def get_edge_weight_method(weight_method, display=True):
	#chose edge computation method, store relevant function in variable for easy no-if calling later
	if weight_method == "jaccard":
		if display:
			vprint("Using jaccard index for edge weight")
		return jaccard_edge_weight
	elif weight_method == "cosine":
		if display:
			vprint("Using cosine similarity for edge weight")
		return cosine_edge_weight
	else:  #word_mover
		if display:
			vprint("Using word-mover distance for edge weight")
		return word_mover_edge_weight
#end get_edge_weight_method


#given two lists of tokens from two posts, 
#compute the edge weight between the posts using jaccard index
def jaccard_edge_weight(tokens_a, tokens_b):
	intersection = len(set(tokens_a).intersection(tokens_b))
	union = len(set(tokens_a).union(tokens_b))
	if union == 0:
		return 0.0
	return float(intersection / union)
#end jaccard_edge_weight


#given two lists of tokens from two posts,
#compute the edge weight between the posts using tf-idf cosine similarity
def cosine_edge_weight(tokens_a, tokens_b):
	print("TODO, no cosine yet, sorry, quitting")
	exit(0)
#end cosine_edge_weight


#given two lists of tokens from two posts,
#compute the edge weight between the posts using GloVe embeddings and word-mover distance
def word_mover_edge_weight(tokens_a, tokens_b):
	print("TODO, no word-mover yet, sorry, quitting")
	exit(0)
#end word_mover_edge_weight


#small graph-build helper method, adds edge specified from a to b with weight
#adjacency list stored as two parallel lists, sorted by weight
def add_edge(graph, a, b, weight):
	index = bisect.bisect_left(graph[a]['weights'], weight)
	graph[a]['weights'].insert(index, weight)
	graph[a]['neighbors'].insert(index, b)
#end add_edge


#small graph-build helper method, removes smallest weight edge from node
#assumes edges are sorted (which they are)
def remove_low_edge(graph, node):
	graph[node]['weights'].pop(0)
	graph[node]['neighbors'].pop(0)
#end remove_low_edge


#for a given post, infer parameters using post graph
#parameters:
#	sim_post 					post to be simulated
#	sim_post_id 				id of simulation post
#	weight_method 				method to use for edge weight calculation: jaccard, cosine, or wmd
#	min_weight 					minimum weight of title-based edge to include in grpah
#	base_graph 					base graph, excluding sim post, as constructed by build_base_graph - DO NOT
#	eligible_post_ids			list of post ids eligible to be included in graph, based on selected options
#	posts 						set of training posts for graph build
#	cascades 					training cascades (for comment counts)
#	params						dictionary of post id -> fitted params + quality
#	fit_fail_list				list of posts for which param fit failed
#	max_nodes 					if not False, max nodes to include in graph
#	top_n 						if not False, max number of edges per node
#	estimate_initial_params		if True, estimate initial params for sim post based on neighbors
def graph_infer(sim_post, sim_post_id, weight_method, min_weight, base_graph, eligible_post_ids, posts, cascades, params, fit_fail_list, top_n, estimate_initial_params, normalize_parameters, filename_id, display=False):

	if display:
		vprint("Inferring post parameters from post graph")

	#define edge weight computation method
	compute_edge_weight = get_edge_weight_method(weight_method, display)

	#compute new edges between sim post and other posts already in graph - enforcing top_n if in effect
	new_edges = {}		#dictionary of neighbor -> edge weight (for edge between neighbor and sim post)
	#loop posts
	if display:
		vprint("Computing edges with %d posts" % len(eligible_post_ids))
	for post_id in eligible_post_ids:
		#post for this id
		post = posts[post_id]

		#compute edge weight based on post token sets (chose method earlier)
		weight = compute_edge_weight(post['tokens'], sim_post['tokens'])
		if weight < min_weight:		#minimum token weight threshold, try to keep edge explosion to a minimum
			weight = 0

		#if posts have same author, add 1 to weight
		if post['author'] == sim_post['author']:
			weight += 1

		#if edge weight is nonzero, add edge to list
		if weight != 0:
			new_edges[post_id] = weight

			#too many edges? remove the smallest
			if top_n != False and len(new_edges) > top_n:
				remove = min(new_edges, key=new_edges.get)
				del new_edges[remove]

	if display:
		vprint("Found %d edges connecting to sim post" % len(new_edges))

	#how many nodes are actually in the graph, if we dump it now? build set of nodes in graph
	graph_post_ids = set(base_graph.keys())		#all nodes in base graph
	graph_post_ids.update(new_edges.keys())		#plus nodes connected to sim_post	
	graph_post_ids.add(sim_post_id)				#and include sim_post in graph!

	#estimate initial params for sim_post, if required
	if estimate_initial_params:
		#weighted average of neighboring post params, weight is (1-quality)
		estimated_params = [0, 0, 0, 0, 0, 0]
		neighbor_count = 0
		#loop edges connecting sim_post to other posts
		for neighbor_id, weight in new_edges.items():
			#if neighbor included in (possibly sampled) graph, add their params to running total
			if neighbor_id in graph_post_ids:
				#grab params/quality for average
				if neighbor_id in params:
					fitted_params = params[neighbor_id]
					#fill in holes if params not complete
					if not all(fitted_params):
						fitted_params = get_complete_params(cascades[neighbor_id], fitted_params)
				else:
					fitted_params = get_default_params(cascades[neighbor_id])
				#weighted sum params, update count
				for i in range(6):
					estimated_params[i] += (1 - fitted_params[6]) * fitted_params[i]
				neighbor_count += (1 - fitted_params[6])

		#for initial param estimate, finish average and return result
		if neighbor_count != 0:
			for i in range(6):
				estimated_params[i] /= neighbor_count
		if display:
			vprint("Estimated params: ", estimated_params)

	#build edgelist of all graph edges, both base and new, removing duplicate edges as we go
	#also enforce the top_n criteria, if in effect
	#also assign numeric node ids, forcing sim_post = 0
	edges = {}			#edge (node1, node2) -> weight
	numeric_ids = {}	#post_id -> numeric node_id
	numeric_ids[sim_post_id] = 0
	next_id = 1
	disconnected = False
	#loop all posts to be included in graph
	for post_id in graph_post_ids:
		#edges connected to sim_post
		if post_id in new_edges:
			#does this post need an id? assign it
			if post_id not in numeric_ids:
				numeric_ids[post_id] = next_id
				next_id += 1
			edges[(numeric_ids[post_id], 0)] = new_edges[post_id]	#(post, sim_post) -> weight

		#base graph nodes
		if post_id in base_graph:
			#does this post need an id? assign it
			if post_id not in numeric_ids:
				numeric_ids[post_id] = next_id
				next_id += 1
			#loop this post's edgelist
			for i in range(len(base_graph[post_id]['weights'])):
				other_post = base_graph[post_id]['neighbors'][i]
				weight = base_graph[post_id]['weights'][i]
				#does other post need an id?
				if other_post not in numeric_ids:
					numeric_ids[other_post] = next_id
					next_id += 1
				#add edge if not in list already
				if (numeric_ids[other_post], numeric_ids[post_id]) not in edges and (numeric_ids[post_id], numeric_ids[other_post]) not in edges:
					edges[(numeric_ids[other_post], numeric_ids[post_id])] = weight
	if display:
		vprint("Final graph contains %d nodes and %d edges" % (len(graph_post_ids), len(edges)))

	#if no edges connecting sim post to graph (BAD), add node as isolated so that we at least get something back
	#and print a big fat warning
	if len(new_edges) == 0:
		if display: vprint("WARNING: No edges connecting sim post to infer graph. Results may be poor.")
		isolated_nodes = [0]
		disconnected = True
	else:
		isolated_nodes = []

	#save graph and params to files for node2vec
	save_graph(edges, temp_graph_filepath % filename_id, isolated_nodes, display)
	param_save_res = save_params(numeric_ids, posts, cascades, params, temp_params_filepath % filename_id, param_estimate=(estimated_params if estimate_initial_params else False), normalize=normalize_parameters, display=display)

	#clear any previous output params
	if file_utils.verify_file(output_params_filepath):
		os.remove(output_params_filepath)		#clear output to prevent append

	#graph is built and ready - graph file and input params file

	#run node2vec to get embeddings - if we have to infer parameters
	#offload to C++, because I feel the need... the need for speed!:

	#run node2vec on graph and params - with on-the-fly transition probs option
	if normalize_parameters == "ln":
		out = subprocess.check_output(["./c_node2vec_ln/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % filename_id), "-ie:"+(temp_params_filepath % filename_id), "-o:"+(output_params_filepath % filename_id), "-d:6", "-l:3", "-w", "-s", "-otf"])
	else:
		out = subprocess.check_output(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % filename_id), "-ie:"+(temp_params_filepath % filename_id), "-o:"+(output_params_filepath % filename_id), "-d:6", "-l:3", "-w", "-s", "-otf"])
	#subprocess.check_call(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % filename_id), "-ie:"+(temp_params_filepath % filename_id), "-o:"+(output_params_filepath % filename_id), "-d:6", "-l:3", "-w", "-s", "-otf"])
	if display:
		vprint("")

	#load the inferred params (dictionary of numeric id -> params) and extract sim_post inferred params
	all_inferred_params = load_inferred_params(output_params_filepath % filename_id, normalize=normalize_parameters, min_max_params=param_save_res, display=display)
	inferred_params = all_inferred_params[numeric_ids[sim_post_id]]

	#cap branching factor at 1 to prevent infinite sim
	if inferred_params[5] > 1:
		inferred_params[5] = 1

	return inferred_params, disconnected, len(new_edges)
#end graph_infer


#given a post (in the form of a cascade object), get default params for it
def get_default_params(post):
	total_comments = post['comment_count_total']
	direct_comments = post['comment_count_direct']

	#weibull: based on number of comments
	if total_comments == 0:
		params = DEFAULT_WEIBULL_NONE.copy()
	else:
		params = DEFAULT_WEIBULL_SINGLE.copy()
		params[0] = direct_comments
	#lognorm: same for all
	params.extend(DEFAULT_LOGNORMAL)
	#branching factor: estimate as in fit
	params.append(fit_cascade_gen_model.estimate_branching_factor(direct_comments, total_comments-direct_comments))
	#and append quality
	params.append(DEFAULT_QUALITY)

	return params
#end get_default_params


#given a post (in the form of a cascade object), and incomplete fitted params, 
#get a complete param set by filling in the holes
def get_complete_params(post, params):
	total_comments = post['comment_count_total']
	direct_comments = post['comment_count_direct']

	new_params = params.copy()		#don't modify original params

	#weibull: default based on number of comments
	if params[0] == False:
		if total_comments == 0:
			new_params[:3] = DEFAULT_WEIBULL_NONE.copy()
		else:
			new_params[:3] = DEFAULT_WEIBULL_SINGLE.copy()
			new_params[0] = direct_comments

	#lognorm: default same for all
	if params[3] == False:
		new_params[3:5] = DEFAULT_LOGNORMAL.copy()

	#branching factor will always be a value, so leave it alone
	#and quality will already be set based on using defaults for the missing

	return new_params
#end get_complete_params


#save graph to txt file for node2vec processing, assigning numeric node ids along the way
#force sim_post to have id=0 for easy lookup later
def save_graph(edgelist, filename, isolated_nodes = [], display=False):
	#and save graph to file
	with open(filename, "w") as f:
		for edge, weight in edgelist.items():
			f.write("%d %d %f\n" % (edge[0], edge[1], weight))
		for node in isolated_nodes:
			f.write("%d\n" % node)
	if display:
		vprint("Saved graph to %s" % filename)
#end save_graph


#save params to txt file for node2vec processing
def save_params(numeric_ids, posts, cascades, params, filename, param_estimate=False, normalize=None, display=False):

	#first, build a complete list of params to save - including any default or incomplete ones
	save_params = {}		#post id -> params
	for post_id, numeric_id in numeric_ids.items():
		#skip sim post for now
		if numeric_id == 0:
			continue

		#fetch params for this post - either from dict or as default
		if post_id in params:
			post_params = params[post_id]				
			#fill in holes if params not complete
			if not all(post_params):
				post_params = get_complete_params(cascades[post_id], post_params)
		else:
			post_params = get_default_params(cascades[post_id])

		#save to dump dictionary
		save_params[post_id] = post_params

	#if min-max normalizing, find min and max values in each position (not sticky/quality)
	if normalize == "mm":
		#start min/max tracking at the sim post estimated params (if we have them)
		if param_estimate != False:
			min_params = param_estimate.copy()
			max_params = param_estimate.copy()
		#otherwise, start them at a param set from the graph (doesn't matter which one)
		else:
			min_params = save_params[list(save_params.keys())[0]].copy()
			max_params = save_params[list(save_params.keys())[0]].copy()

		#loop to find min and max values at each position
		for post_id, params in save_params.items():
			for i in range(6):
				if params[i] < min_params[i]:
					min_params[i] = params[i]
				if params[i] > max_params[i] :
					max_params[i] = params[i]
		#compute normalization denominator
		denom = [0] * 6
		for i in range(6):
			denom[i] = max_params[i] - min_params[i]

	#if natural log normalizing, just apply the natural log to all values as we write them

	#write all params out to file
	with open(filename, "w") as f: 
		for post_id, numeric_id in numeric_ids.items():
			#skip sim post for now
			if numeric_id == 0:
				continue

			f.write(str(numeric_id) + " ")		#write numeric post id			
			post_params = save_params[post_id]	#fetch params
			#write params, normalizing if we have to
			for i in range(6):
				try:
					if normalize == "mm": f.write(str((post_params[i] - min_params[i]) / denom[i]) + ' ')
					elif normalize == "ln": 
						if post_params[i] > 0:
							f.write(str(math.log(post_params[i])) + ' ')
						else:
							f.write(str(math.log(sys.float_info.min)) + ' ')
					else: f.write(str(post_params[i]) + ' ')
				except Exception as e:
					print(e)
					print(i, post_params[i])
					exit(0)
			f.write(str(post_params[6]) + "\n")	#write quality (last value in params)

		#write estimated params for sim_post, if we have them	(no quality)
		if param_estimate != False:
			f.write(str(0) + " ")		#write numeric post id - in this case 0
			for i in range(0):
				if normalize == "mm": f.write(str((param_estimate[i] - min_params[i]) / denom[i]) + ' ')
				if normalize == "ln": f.write(str(math.log(param_estimate[i])) + ' ')
				else: f.write(str(param_estimate[i]) + ' ')
			f.write("\n")

	if display:
		vprint("Saved graph params to %s" % filename)

	#if min-max normalized, return min/max values so we can reverse it later
	#otherwise, return None (no return data)
	if normalize == 'mm':
		return min_params, max_params
	else:
		return None
#end save_params


#load inferred params from file
def load_inferred_params(filename, normalize=False, min_max_params=False, display=False):
	#if reversing min-max normalization, compute denominator
	if normalize == 'mm':
		denom = [0] * 6
		min_params = min_max_params[0]
		max_params = min_max_params[1]
		for i in range(6):
			denom[i] = max_params[i] - min_params[i]

	#read all lines of file
	with open(filename, 'r') as f:
		lines = f.readlines()

	#reading inferred file, so skip first line
	lines.pop(0)

	all_params = {}
	#process each line, extract params
	for line in lines:
		values = line.split()
		post_id = int(values[0])	#get numeric id for this post
		params = []
		#read params
		for i in range(1, 7):
			params.append(float(values[i]))
		#reverse normalization, if applicable
		if normalize == 'mm':
			for i in range(6): params[i] = params[i] * denom[i] + min_params[i]
		elif normalize == 'ln':
			for i in range(6): params[i] = math.exp(params[i])
		all_params[post_id] = params

	if display:
		vprint("Loaded %d fitted params from %s" %(len(all_params), filename))

	return all_params
#end load_inferred_params


#given params, simulate a comment tree
#arguments: simulation parameters, observed cascade (shifted comment times in minutes), 
#time observed (in minutes)
#returned simulated cascade has relative comment times in minutes
#returns a new tree, does not modify the version passed in
def simulate_comment_tree(sim_params, observed_tree, time_observed, display=False):
	if display:
		vprint("\nSimulating comment tree")

	#simulate tree structure + comment times!	
	
	#simulate from the observed tree
	sim_root, all_times = sim_tree.simulate_comment_tree(sim_params, time_observed, deepcopy(observed_tree))

	#catching the infinite sim
	if sim_root == False:
		return False, observed_count, time_observed, -1

	if display:
		vprint("Generated %d total comments for post (including %d observed)" % (len(all_times), observed_count))
		vprint("   %d actual\n" % sim_cascade['comment_count_total'])

	#return simulated tree, observed comment count, time_observed, and simulated comment count (counts for output/eval)
	return sim_root, len(all_times)
#end simulate_comment_tree


#given a ground-truth cascade stored as nested dictionary structure, 
#offset comment times by root time and convert relative comment times to minutes
def shift_comment_tree(cascade):
	#build new list/structure of post comments - offset times by post time
	shifted_tree = deepcopy(cascade)	#start with given, modify from there

	root_time = cascade['time']		#grab post time to use as offset
	shifted_tree['time'] = 0		#post at time 0

	#traverse the tree, removing unobserved comments and offsetting times
	comments_to_visit = [] + [(shifted_tree, reply) for reply in shifted_tree['replies']]	#init queue to root replies
	comment_count = 0
	while len(comments_to_visit) != 0:
		parent, curr = comments_to_visit.pop()		#get current comment
		#observed comment time, shift/convert if required and add replies to queue
		curr['time'] = (curr['time'] - root_time) / 60.0
		comments_to_visit.extend([(curr, reply) for reply in curr['replies']])
		comment_count += 1
	
	return shifted_tree, comment_count		#return root post of converted tree
#end shift_comment_tree


#given a ground-truth cascade stored as nested dictionary structure, and an observed time, 
#filter tree to only the comments we have observed
#cascade comments already shifted relative to root and converted to minutes
#time_observed given in minutes
#modifies the given cascade by deleting comments - does NOT create a copy
def filter_comment_tree(cascade, time_observed):
	#traverse the tree, removing unobserved comments
	comments_to_visit = [] + [(cascade, reply) for reply in cascade['replies']]	#init queue to root replies
	observed_count = 0
	while len(comments_to_visit) != 0:
		parent, curr = comments_to_visit.pop()		#get current comment
		#check time, delete if not within observed window
		if curr['time'] > time_observed:
			parent['replies'].remove(curr)
			continue
		#observed comment, add replies to queue
		observed_count += 1
		comments_to_visit.extend([(curr, reply) for reply in curr['replies']])

	#return post/root
	return cascade, observed_count
#end filter_comment_tree


#given a ground-truth cascade stored as nested dictionary structure, and an observed number of comments,
#filter tree to only the comments we have observed
#also return the time observed in hours
#cascade comments already shifted relative to root and converted to minutes
#time_observed given in minutes
#modifies the given cascade by deleting comments - does NOT create a copy
def filter_comment_tree_by_num_comments(cascade, num_observed):
	#get sorted list of ALL comment times, in minutes
	all_comment_times = get_list_of_comment_times(cascade)
	#pull just the observed comments
	observed_comments = all_comment_times[:num_observed]

	try:
		#what is the observation time corresponding to this set of observed comments?
		#no observed comments at all, time_observed is 0
		if len(observed_comments) == 0:		
			time_observed = 0
		#if no unobserved comments (but at least one observed), just use time of last observed comment + 1 (again, prevent float sadness)
		elif len(observed_comments) == len(all_comment_times) and len(observed_comments) != 0:
			time_observed = observed_comments[-1] + 1
		#observed some, but not all comments
		#use time of the first comment that we didn't observe (if such a thing exists),
		#averaged with the last time we did observe (midpoint between two)
		#pick this to (hopefully) prevent floating-point sadness
		elif num_observed < len(all_comment_times):
			time_observed = (all_comment_times[num_observed] + observed_comments[-1]) / 2.0
	except Exception as e:
		print(e)
		print(num_observed)
		print(all_comment_times)
		print(observed_comments)
		exit(0)	

	#what is the expected observed count based on this time? 
	#could be a little more than num_observed, if multiple comments at the same time
	expected_observed_count = len([time for time in all_comment_times if time <= time_observed])

	#filter tree based on this observation time (determined by # of comments)
	observed_tree, observed_count = filter_comment_tree(cascade, time_observed)

	#verify that the tree size matches
	if observed_count != expected_observed_count:
		print("Error filtering comment tree by number of observed comments. Exiting.")
		print("all", len(all_comment_times), all_comment_times)
		print("observed comments list", len(observed_comments), observed_comments)
		print("max observed from argument", num_observed)
		print("time_observed", time_observed)
		print("expected observed from list filter", expected_observed_count)
		print("observed count from filter function", observed_count)
		print("observed tree", observed_tree)
		exit(0)

	return observed_tree, observed_count, time_observed / 60.0
#end filter_comment_tree_by_num_comments


#given simulated and ground-truth cascades, compute the accuracy and precision of the simulation
#both trees given as dictionary-nested structure (returned from simulate_comment_tree and convert_comment_tree)
#return eval results in a metric-coded dictionary
def eval_trees(post_id, sim_cascade, true_cascade, simulated_comment_count, observed_comment_count, true_comment_count, true_virality, time_observed, observing_time, time_error_margin, error_method, disconnected, new_edges, max_observed_comment_count=None):
	#get edit distance stats for sim vs truth
	eval_res = tree_edit_distance.compare_trees(sim_cascade, true_cascade, error_method, time_error_margin)

	#compute structural virality of sim cascade
	sim_virality = get_structural_virality(sim_cascade)

	#add more data fields to the results dictionary
	eval_res['post_id'] = post_id
	eval_res['observed_comment_count'] = observed_comment_count
	eval_res['true_comment_count'] = true_comment_count
	eval_res['simulated_comment_count'] = simulated_comment_count
	eval_res['disconnected'] = "True" if disconnected else "False"
	eval_res['connecting_edges'] = new_edges
	eval_res['time_observed'] = time_observed
	eval_res['observing_by'] = "time" if observing_time else "comments"
	if max_observed_comment_count is not None:
		eval_res['max_observed_comments'] = max_observed_comment_count

	#breakdown of comment counts - root level comments for both true and sim cascades
	eval_res['true_root_comments'] = true_cascade['comment_count_direct']
	eval_res['sim_root_comments'] = len(sim_cascade['replies'])
	#can get other = total - root in post-processing

	#normalize the tree edit distance in a couple different ways - even though it's not perfect
	#divide by true comment count
	eval_res['norm_dist'] = eval_res['dist'] / eval_res['true_comment_count']	
	#divide by number of unobserved comments
	eval_res['norm_dist_exclude_observed'] = eval_res['dist'] / (eval_res['true_comment_count'] - eval_res['observed_comment_count'])

	#structural virality
	eval_res['true_structural_virality'] = true_virality
	eval_res['sim_structural_virality'] = sim_virality

	#mean error per distance layer (both min and max)
	eval_res['MEPDL_min'], eval_res['MEPDL_max'] = mean_error_per_distance_layer(true_cascade, sim_cascade)
	

	return eval_res
#end eval_trees


#given a cascade (nested dict format), count the number of nodes on each level
#return as a dictionary of depth -> node count
#also find (and return) the min and max depths of the tree
def count_nodes_per_level(cascade):
    depth_counts = defaultdict(int)     #dictionary for depth counts
    depth_counts[1] = 1			#root on level 1, all by itself

    #min/max depths
    min_leaf_level = 1		#default for tree with only root is 1
    max_leaf_level = -1

	#init processing queue to direct post replies (on level 2)
    nodes_to_visit = [(reply, 2) for reply in cascade['replies']]	#(node, depth)  
    while len(nodes_to_visit) != 0:     #BFS
        curr, depth = nodes_to_visit.pop(0)    #grab current comment and depth
        depth_counts[depth] += 1		#update count for this depth
        #add this comment's replies to processing queue
        nodes_to_visit.extend([(reply, depth+1) for reply in curr['replies']])

        #track lowest and highest leaves
        if len(curr['replies']) == 0:	#if this node is a leaf
        	if depth < min_leaf_level or min_leaf_level == 1:
        		min_leaf_level = depth
        	if depth > max_leaf_level:
        		max_leaf_level = depth

    return depth_counts, min_leaf_level, max_leaf_level
#end count_nodes_per_level


#given two cascades (true and sim), compute the mean error per distance layer between them
#compute both min and max, based on min and max depths of trees
def mean_error_per_distance_layer(true_cascade, sim_cascade):	
	#first, compute by-level counts for both trees
	true_by_level, true_min_depth, true_max_depth = count_nodes_per_level(true_cascade)
	sim_by_level, sim_min_depth, sim_max_depth = count_nodes_per_level(sim_cascade)

	#use min/max across both trees for computation
	min_depth = min(true_min_depth, sim_min_depth)
	max_depth = max(true_max_depth, sim_max_depth)

	#compute differences for all levels of trees
	level_diffs = {}
	#sum for both min and max depths
	err_max = 0
	err_min = 0
	#loop all levels, but only sum some of them for err_min
	for level in range(1, max_depth+1):
		level_diffs[level] = abs(true_by_level[level] - sim_by_level[level])	#compute diff at this level
		err_max += level_diffs[level]		#always add to sum for err_max
		if level <= min_depth:
			err_min += level_diffs[level]		#only add to sum for err-min if level <= min_depth

	#divide error summations by corresponding depth
	err_max /= max_depth
	err_min /= min_depth

	return err_min, err_max	
#end mean_error_per_distance_layer


#given a cascade tree (true or simulated), compute the structural virality
def get_structural_virality(cascade):
	#get cascade as an undirected networkx graph
	graph = cascade_to_graph(cascade)

	#compute structural virality
	n = nx.number_of_nodes(graph)
	#at least one comment, calculate
	if n > 1:
		struct_virality = nx.wiener_index(graph) * 2 / (n * (n - 1))
	#no comments, set to 0 (wiener index would be 0, but then divide by 0 so hardcode)
	else:
		struct_virality = 0.0

	return struct_virality
#end get_structural_virality


#given a cascade (nested dict structure), convert it to a networkx graph
def cascade_to_graph(cascade):
	#build list of edges in graph
	edges = []    #list of edges (list of tuples)

	#init queue to root, will process nodes as they are removed from the queue
	nodes_to_visit = [cascade]
	while len(nodes_to_visit) != 0:
		node = nodes_to_visit.pop(0)    #grab current comment
		#add edges between this node and all children to edge list
		#also add children to processing queue
		for comment in node['replies']:
			edges.append((node['id'], comment['id']))	#edge as tuple
			nodes_to_visit.append(comment)    #add reply comment to processing queue

	#use edgelist to build a graph
	G=nx.Graph()		#new graph
	G.add_edges_from(edges)

	#if no edges (ie, no comments), add a root
	if len(edges) == 0:
		G.add_node(0)

	return G 		#return the graph
#end cascade_to_graph


#load set of finished posts from pickle, if the file exists
#this acts as progress bookmark in case of sad process deaths
#file includes a status flag, which is also unpacked and returned
def load_bookmark(base_filename):
	#no bookmark, return empty set and False 
	if file_utils.verify_file(base_filename+"_finished_posts.pkl") == False:
		return set(), False

	#otherwise, load and unpack	
	bookmark_data = file_utils.load_pickle(base_filename+"_finished_posts.pkl")
	#return set of finished posts and status - True if finished, False otherwise
	return bookmark_data['finished_posts'], bookmark_data['complete']
#end save_bookmark	


#save set of finished posts to pickle - acts as progress bookmark in case of sad process deaths
#includes a completion status flag - either True, indicating all posts simulated and results saved, 
#   or False, indicating that there are still posts that must be simulated
def save_bookmark(finished_posts, base_filename, status=False):
	#put post set and flag in a dictionary for unpacking later
	bookmark_data = {"finished_posts": finished_posts, "complete": status}
	#save file
	file_utils.save_pickle(bookmark_data, base_filename+"_finished_posts.pkl")
#end save_bookmark	


#save all sim results to csv file
#one row per simulated post/time pair, with a bunch of data in it
def save_results(base_filename, metrics, observing_time):
	#given a base filename, convert to complete output filename
	filename = base_filename + "_results.csv"

	#dump metrics dict to file, enforcing a semi-meaningful order
	fields = ["post_id", "param_source", "observing_by", "time_observed", "observed_comment_count", "true_comment_count", "simulated_comment_count", "true_root_comments", "sim_root_comments", "true_depth", "true_breadth", "simulated_depth", "simulated_breadth", "true_structural_virality", "sim_structural_virality", "dist", "norm_dist", "norm_dist_exclude_observed", "MEPDL_min", "MEPDL_max", "remove_count", "remove_time", "insert_count", "insert_time", "update_count", "update_time", "match_count", "disconnected", "connecting_edges"]
	if observing_time == False:
		fields.insert(5, "max_observed_comments")

	#if file already exists, append to it
	if file_utils.verify_file(filename):
		file_utils.save_csv(metrics, filename, fields, file_mode='a')
	#otherwise, create from scratch
	else:
		file_utils.save_csv(metrics, filename, fields)

	return
#end save_results

