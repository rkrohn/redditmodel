#functions for paper_model.py - offloading and modularizing all the things

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

#filepath for random test samples, determined by subreddit, number of posts, testing start (year-month), testing length, and min number of comments
#(save these to files so you can have repeated runs of the same random set)
random_sample_list_filepath = "sim_files/%s_%d_test_keys_list_start%d-%d_%dmonths_filter%d.pkl"

#filepaths of output/temporary files - used to pass graph to C++ node2vec for processing
temp_graph_filepath = "sim_files/graph_%s.txt"			#updated graph for this sim run
temp_params_filepath = "sim_files/in_params_%s.txt"		#temporary, filtered params for sim run (if sampled graph)
output_params_filepath = "sim_files/out_params_%s.txt"		#output params from node2vec

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
	parser.add_argument("-o", "--out", dest="outfile", required=True, help="output filename")
	#must pick one of four processing options: a single id, random, all, or sample of size n
	proc_group = parser.add_mutually_exclusive_group(required=True)
	proc_group.add_argument("-id", dest="sim_post", default=None,  help="post id for single-processing")
	proc_group.add_argument("-r", "--rand", dest="sim_post", action="store_const", const="random", help="choose a random post from the subreddit to simulate")
	proc_group.add_argument("-a", "--all", dest="sim_post", action="store_const", const="all", help="simulate all posts in the subreddit")
	proc_group.add_argument("-n", "--n_sample", dest="sim_post", default=100, help="number of posts to test, taken as random sample from testing period")
	#must provide year and month for start of testing data set
	parser.add_argument("-y", "--year", dest="testing_start_year", required=True, help="year to use for test set")
	parser.add_argument("-m", "--month", dest="testing_start_month", required=True, help="month to use for test set")
	#must pick an edge weight computation method: cosine (based on tf-idf) or jaccard
	weight_group = parser.add_mutually_exclusive_group(required=True)
	weight_group.add_argument("-j", "--jaccard", dest="weight_method", action='store_const', const="jaccard", help="compute edge weight between pairs using jaccard index")
	weight_group.add_argument("-c", "--cosine", dest="weight_method", action='store_const', const="cosine", help="compute edge weight between pairs using tf-idf and cosine similarity")
	weight_group.add_argument("-wmd", "--word_mover", dest="weight_method", action='store_const', const="word_mover", help="compute edge weight between pairs using GloVe embeddings and word-mover distance")
	#must pick an edge limit method: top n edges per node, or weight threshold, or both
	parser.add_argument("-topn", dest="top_n", default=False, metavar=('<max edges per node>'), help="limit post graph to n edges per node")
	parser.add_argument("-threshold", dest="weight_threshold", default=False, metavar=('<minimum edge weight>'), help="limit post graph to edges with weight above threshold")

	#optional args	
	parser.add_argument("-t", dest="time_observed", default=[0], help="time of post observation, in hours", nargs='+')
	parser.add_argument("-g", "--graph", dest="max_nodes", default=False, help="max nodes in post graph for parameter infer")
	parser.add_argument("-q", "--qual", dest="min_node_quality", default=False, help="minimum node quality for post graph")
	parser.add_argument("-e", "--esp", dest="estimate_initial_params", action='store_true', help="estimate initial params as inverse quality weighted average of neighbor nodes")
	parser.set_defaults(estimate_initial_params=False)
	parser.add_argument("-l", "--testlen", dest="testing_len", default=1, help="number of months to use for testing")
	parser.add_argument("-p", "--periodtrain", dest="training_len", default=1, help="number of months to use for training (preceding first test month")
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose output")
	parser.set_defaults(verbose=False)
	parser.add_argument("-d", "--default_params", dest="include_default_posts", action='store_true', help="include posts with hardcoded default parameters in infer graph")
	parser.set_defaults(include_default_posts=False)
	parser.add_argument("--err", dest="time_error_margin", default=False, help="allowable time error for evaluation, in minutes")
	parser.add_argument("--err_abs", dest="time_error_absolute", action="store_true", help="use absolute time error, instead of increasing by level")
	parser.set_defaults(time_error_absolute=False)
	parser.add_argument("--topo_err", dest="topological_error", action="store_true", help="compute topological error only, ignoring comment timestamps")
	parser.set_defaults(topological_error=False)
	parser.add_argument("-np", "--norm_param", dest="normalize_parameters", action="store_true", help="min-max normalize paramters for graph inference")
	parser.set_defaults(normalize_parameters=False)
	parser.add_argument("--sanity", dest="sanity_check", action="store_true", help="sanity check: simulate from fitted params instead of inferring")
	parser.set_defaults(sanity_check=False)
	#can also layer in a size filter: only simulate cascades above a certain size 
	#(filter applied before sample/rand)
	parser.add_argument("-sf", "--size_filter", dest="size_filter", default=False, help="minimum cascade size for simulation test set")

	args = parser.parse_args()		#parse the args (magic!)

	#make sure at least one edge-limit option was chosen
	if not (args.top_n or args.weight_threshold):
		parser.error('No edge limit selected, add -topn, -threshold, or both')

	#make sure error settings don't conflict
	if args.topological_error and (args.time_error_absolute or args.time_error_margin != False):
		parser.error('Cannot use topological error method with absolute time error or error margin setting')

	#extract arguments (since want to return individual variables)
	subreddit = args.subreddit
	sim_post = args.sim_post
	time_observed = [float(time) for time in args.time_observed]
	outfile = args.outfile
	max_nodes = args.max_nodes if args.max_nodes == False else int(args.max_nodes)
	min_node_quality = args.min_node_quality if args.min_node_quality == False else float(args.min_node_quality)
	estimate_initial_params = args.estimate_initial_params
	testing_start_month = int(args.testing_start_month)
	testing_start_year = int(args.testing_start_year)
	testing_len = int(args.testing_len)
	training_len = int(args.training_len)
	weight_method = args.weight_method
	include_default_posts = args.include_default_posts
	verbose = args.verbose
	top_n = args.top_n
	time_error_margin = float(args.time_error_margin) if args.time_error_margin != False else 30.0
	time_error_absolute = args.time_error_absolute
	topological_error = args.topological_error
	normalize_parameters = args.normalize_parameters
	sanity_check = args.sanity_check
	size_filter = int(args.size_filter) if args.size_filter != False else False
	if top_n != False:
		top_n = int(top_n)
	weight_threshold = args.weight_threshold
	if weight_threshold != False:
		weight_threshold = float(weight_threshold)
	#extra flags/variables for different processing modes
	sample_num = False
	if sim_post == "all":
		batch = True
	elif sim_post == "random":
		batch = False
	else:
		#is this a number (n_sample), or an id (single post)?
		try:
			#number of posts to sample
			int(sim_post)
			batch = True
			sample_num = int(sim_post)
			sim_post = "sample"
		except ValueError:
			#single specified post
			batch = False
	#and for eval mode
	if topological_error:
		error_method = "topo"
	elif time_error_absolute:
		error_method = "abs"
	else:	#both false, use by-level
		error_method = "level"

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
	vprint("Sim Post: ", sim_post, " %d" % sample_num if sample_num != False else "")
	vprint("Time Observed: ", time_observed)
	vprint("Output: ", outfile)
	vprint("Source subreddit: ", subreddit)
	vprint("Minimum node quality: ", min_node_quality)
	vprint("Max graph size: ", max_nodes)
	vprint("Max edges per node: ", "None" if top_n==False else top_n)
	vprint("Minimum edge weight: ", "None" if weight_threshold==False else weight_threshold)
	if estimate_initial_params:
		vprint("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
	if normalize_parameters:
		vprint("Normalizing params (min-max) for graph infer")
	else:
		vprint("No param normalization for infer step")
	vprint("Testing Period: %d-%d" % (testing_start_month, testing_start_year), " through %d-%d (%d months)" % (monthdelta(testing_start_month, testing_start_year, testing_len, inclusive=True)+(testing_len,)) if testing_len > 1 else " (%d month)" % testing_len)
	vprint("Training Period: %d-%d" % (training_start_month, training_start_year), " through %d-%d (%d months)" % (monthdelta(training_start_month, training_start_year, training_len, inclusive=True)+(training_len,)) if training_len > 1 else " (%d month)" % training_len)
	if weight_method == "jaccard":
		vprint("Using Jaccard index to compute graph edge weights")
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
	if size_filter != False:
		vprint("Only simulating cascades with true size greater than or equal to %d" % size_filter)
	if sanity_check:
		vprint("Simulating from fitted params, skipping graph/infer/refine steps")
	vprint("")

	#return all arguments
	return subreddit, sim_post, time_observed, outfile, max_nodes, min_node_quality, estimate_initial_params, normalize_parameters, batch, sample_num, testing_start_month, testing_start_year, testing_len, training_start_month, training_start_year, training_len, weight_method, top_n, weight_threshold, include_default_posts, time_error_margin, error_method, sanity_check, size_filter, verbose
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
#if cascades = True, also load reconstructed cascades for these posts
#if files don't exist, call methods to perform preprocessing
def load_processed_posts(subreddit, start_month, start_year, num_months, load_params=False, load_cascades=False):
	posts = {}
	params = {}
	failed_fit_posts = []
	cascades = {}

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

	#save both fitted params and list of failed fits to the same file in a wrapping dictionary
	params_out = {"params_dict": cascade_params, "failed_fit_list": failed_fit_posts}
	file_utils.save_pickle(params_out, fitted_params_filepath % (subreddit, subreddit, year, month))
	
	return params_out		#return params + fail list in dict
#end fit_posts


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
		cascades = build_cascades(subreddit, month, year, posts, comments)

	return cascades
#end get_cascades


#filter one dictionary based on list of keys
#returns modified dictionary
#if num_deleted=True, also return number of items removed
def filter_dict_by_list(dict_to_filter, keep_list, num_deleted=False):
	del_keys = set([key for key in dict_to_filter.keys() if key not in keep_list])
	for key in del_keys:
		dict_to_filter.pop(key, None)

	if num_deleted: return dict_to_filter, len(del_keys)
	return dict_to_filter
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


#given a post id, boolean mode flags, and dictionary of posts, ensure post is in this set
#if running in sample mode, pick the sample
#returns modified post dictionary that contains only the posts to be tested
def get_test_post_set(input_sim_post, batch_process, size_filter, sample_num, posts, cascades, subreddit, testing_start_month, testing_start_year, testing_len):
	#apply min-size filter before anything else
	if size_filter != False:
		keys = [post_id for post_id in cascades if cascades[post_id]['comment_count_total'] >= size_filter]
		posts = filter_dict_by_list(posts, keys)
		vprint("Filtered to %d posts with >= %d comments" % (len(keys), size_filter))

	#if processing all posts in test set, return list of ids
	if input_sim_post == "all":		
		vprint("Processing all %d posts in test set" % len(posts))
	#if random post id, pick an id from loaded posts
	elif input_sim_post == "random":
		rand_sim_post_id = random.choice(list(posts.keys()))
		posts = {rand_sim_post_id: posts[rand_sim_post_id]}
		vprint("Choosing random simulation post: %s" % rand_sim_post_id)
	#if sampling, choose random sample of posts
	elif sample_num != False:		
		#if stored random sample exists, load that
		if file_utils.verify_file(random_sample_list_filepath % (subreddit, sample_num, testing_start_year, testing_start_month, testing_len, (size_filter if size_filter != False else 0))):
			vprint("Loading cached sample simulation post set")
			keys = file_utils.load_pickle(random_sample_list_filepath % (subreddit, sample_num, testing_start_year, testing_start_month, testing_len, (size_filter if size_filter != False else 0)))
		#no existing sample file, pick random post id set, and dump list to pickle
		else:
			vprint("Sampling %d random posts for simulation set" % sample_num)
			keys = random.sample(list(posts.keys()), sample_num)
			file_utils.save_pickle(keys, random_sample_list_filepath % (subreddit, sample_num, testing_start_year, testing_start_month, testing_len, (size_filter if size_filter != False else 0)))
		#filter posts to match keys list
		posts = filter_dict_by_list(posts, keys)
	#if single id, make sure given post id is in the dataset
	else:
		#if given not in set, exit
		if input_sim_post not in posts:
			print("Given post id not in group set - exiting.\n")
			exit(0)
		posts = {input_sim_post: posts[input_sim_post]}
		vprint("Using input post id: %s" % input_sim_post)

	return posts
#end get_test_post_set


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
def build_base_graph(posts, params, default_params_list, include_default_posts, max_nodes, min_node_quality, weight_method, min_weight, top_n):

	vprint("\nBuilding param graph for %d posts" % len(posts))
	
	#define post set to use for graph build based on options
	graph_post_ids = filter_post_set(params, default_params_list, min_node_quality, include_default_posts)	
	vprint("Using %d posts for graph" % len(graph_post_ids))

	#do we need to sample the graph? if so, do it now
	#the graph may well end up smaller than the max_nodes limit (because probably not all connected)
	#but that's for the user to deal with!
	if max_nodes != False and len(graph_post_ids) >= max_nodes:
		vprint("\nSampling graph to %d nodes" % max_nodes)
		#sample down posts, true random sample
		graph_post_ids = set(random.sample(graph_post_ids, max_nodes-1))	#-1, leave room for sim_post

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
	graph = defaultdict(lambda: defaultdict(list))
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
		if pair_count % 100000 == 0:
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
	if normalize_parameters:
		min_params, max_params = save_params(numeric_ids, posts, cascades, params, temp_params_filepath % filename_id, param_estimate=(estimated_params if estimate_initial_params else False), normalize=True, display=display)
	else:
		save_params(numeric_ids, posts, cascades, params, temp_params_filepath % filename_id, param_estimate=(estimated_params if estimate_initial_params else False), normalize=False, display=display)

	#clear any previous output params
	if file_utils.verify_file(output_params_filepath):
		os.remove(output_params_filepath)		#clear output to prevent append

	#graph is built and ready - graph file and input params file

	#run node2vec to get embeddings - if we have to infer parameters
	#offload to C++, because I feel the need... the need for speed!:

	#run node2vec on graph and params - with on-the-fly transition probs option, or probably die
	out = subprocess.check_output(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % filename_id), "-ie:"+(temp_params_filepath % filename_id), "-o:"+(output_params_filepath % filename_id), "-d:6", "-l:3", "-w", "-s", "-otf"])
	#subprocess.check_call(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % filename_id), "-ie:"+(temp_params_filepath % filename_id), "-o:"+(output_params_filepath % filename_id), "-d:6", "-l:3", "-w", "-s", "-otf"])
	if display:
		vprint("")

	#load the inferred params (dictionary of numeric id -> params) and extract sim_post inferred params
	if normalize_parameters:
		all_inferred_params = load_inferred_params(output_params_filepath % filename_id, min_params, max_params, display=display)
	else:
		all_inferred_params = load_inferred_params(output_params_filepath % filename_id, display=display)
	inferred_params = all_inferred_params[numeric_ids[sim_post_id]]

	return inferred_params, disconnected
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
def save_params(numeric_ids, posts, cascades, params, filename, param_estimate=False, normalize=False, display=False):
	#if normalizing, find min and max values in each position (not sticky/quality)
	if normalize:
		#start at defaults, so those values included in min/max checks
		min_params = DEFAULT_WEIBULL_NONE + DEFAULT_LOGNORMAL + [DEFAULT_QUALITY]
		max_params = DEFAULT_WEIBULL_SINGLE + DEFAULT_LOGNORMAL + [DEFAULT_QUALITY]
		for post_id, params in params.items():
			for i in range(6):
				if params[i] < min_params[i]:
					min_params[i] = params[i]
				if params[i] > max_params[i] :
					max_params[i] = params[i]
		#max value at index 0 (first weibull) could be highest number of comments for any default post
		for post_id, numeric_id in numeric_ids.items():
			if post_id not in params and post_id in cascades and cascades[post_id]['comment_count_direct'] > max_params[0]:
				max_params[0] = cascades[post_id]['comment_count_direct']
		#compute normalization denominator
		denom = [0] * 6
		for i in range(6):
			denom[i] = max_params[i] - min_params[i]

	#write all params out to file
	with open(filename, "w") as f: 
		for post_id, numeric_id in numeric_ids.items():
			#skip sim post for now
			if numeric_id == 0:
				continue

			f.write(str(numeric_id) + " ")		#write numeric post id
			#fetch params
			if post_id in params:
				post_params = params[post_id]				
				#fill in holes if params not complete
				if not all(post_params):
					post_params = get_complete_params(cascades[post_id], post_params)
			else:
				post_params = get_default_params(cascades[post_id])
			#write params
			if normalize:
				for i in range(6):
					f.write(str((post_params[i] - min_params[i]) / denom[i]) + ' ')
			else:
				for i in range(6):
					f.write(str(post_params[i]) + ' ')
			f.write(str(post_params[6]) + "\n")	#write quality (last value in params)

		#estimated params for sim_post, if we have them	(no quality)
		if param_estimate != False:
			if normalize:
				for i in range(6):
					f.write(str((param_estimate[i] - min_params[i]) / denom[i]) + ' ')
			else:
				f.write("%d %f %f %f %f %f %f\n" % (0, param_estimate[0], param_estimate[1], param_estimate[2], param_estimate[3], param_estimate[4], param_estimate[5]))

	if display:
		vprint("Saved graph params to %s" % filename)

	#if normalized, return min/max values so we can reverse it later
	if normalize:
		return min_params, max_params
#end save_params


#load inferred params from file
def load_inferred_params(filename, min_params=False, max_params=False, display=False):
	#if reversing normalization, compute denominator
	if min_params != False:
		denom = [0] * 6
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
		if min_params != False:
			for i in range(6):
				params[i] = params[i] * denom[i] + min_params[i]
		all_params[post_id] = params

	if display:
		vprint("Loaded %d fitted params from %s" %(len(all_params), filename))

	return all_params
#end load_inferred_params


#given params, simulate a comment tree
#arguments: post object, simulation parameters, actual cascade, observed time
#sim_cascade comment times in utc seconds, time_observed in minutes
#returned simulated cascade has relative comment times in minutes
def simulate_comment_tree(sim_post, sim_params, group, sim_cascade, time_observed, display=False):
	if display:
		vprint("\nSimulating comment tree")
		vprint("Post created at %d" % sim_post['time'])

	#simulate tree structure + comment times!	
	
	#simulate from partially observed tree
	if time_observed != 0:
		#get observed tree
		observed_tree, observed_count = filter_comment_tree(sim_post, sim_cascade, time_observed)
		#simulate from this observed tree
		sim_root, all_times = sim_tree.simulate_comment_tree(sim_params, time_observed*60, observed_tree)
	#simulate entirely new tree from root only
	else:
		sim_root, all_times = sim_tree.simulate_comment_tree(sim_params)
		observed_count = 0

	if display:
		vprint("Generated %d total comments for post (including %d observed)" % (len(all_times), observed_count))
		vprint("   %d actual\n" % sim_cascade['comment_count_total'])

	#return simulated tree, observed comment count, and simulated comment count (counts for output/eval)
	return sim_root, observed_count, len(all_times)		
#end simulate_comment_tree


#given a ground-truth cascade stored as nested dictionary structure, and an observed time, 
#filter tree to only the comments we have observed, offset comment times by root time, 
#and convert relative comment times to minutes
#time_observed given in hours, cascade times in seconds
#if time_observed == False, just time shift and return that
def filter_comment_tree(post, cascade, time_observed=False):
	#build new list/structure of post comments - offset times by post time
	observed_tree = deepcopy(cascade)	#start with given, modify from there

	#grab post time to use as offset
	root_time = post['time']

	#update root
	observed_tree['time'] = 0		#post at time 0

	#traverse the tree, removing unovserved comments and offsetting times
	comments_to_visit = [] + [(observed_tree, reply) for reply in observed_tree['replies']]	#init queue to root replies
	observed_count = 0
	while len(comments_to_visit) != 0:
		parent, curr = comments_to_visit.pop()		#get current comment
		#check time, delete if not within observed window
		if time_observed != False and curr['time'] - root_time > time_observed * 3600:
			parent['replies'].remove(curr)
			continue
		#observed comment time, shift/convert and add replies to queue
		curr['time'] = (curr['time'] - root_time) / 60.0
		observed_count += 1
		comments_to_visit.extend([(curr, reply) for reply in curr['replies']])

	#return post/root
	return observed_tree, observed_count
#end convert_comment_tree


#given simulated and ground-truth cascades, compute the accuracy and precision of the simulation
#both trees given as dictionary-nested structure (returned from simulate_comment_tree and convert_comment_tree)
#return eval results in a metric-coded dictionary
def eval_trees(post_id, sim_tree, true_cascade, simulated_comment_count, observed_comment_count, true_comment_count, time_observed, time_error_margin, error_method, disconnected):
	#get edit distance stats for sim vs truth
	eval_res = tree_edit_distance.compare_trees(sim_tree, true_cascade, error_method, time_error_margin)

	#add more data fields to the results dictionary
	eval_res['post_id'] = post_id
	eval_res['observed_comment_count'] = observed_comment_count
	eval_res['true_comment_count'] = true_comment_count
	eval_res['simulated_comment_count'] = simulated_comment_count
	eval_res['disconnected'] = "True" if disconnected else "False"
	eval_res['time_observed'] = time_observed

	#breakdown of comment counts - root level comments for both true and sim cascades
	eval_res['true_root_comments'] = true_cascade['comment_count_direct']
	eval_res['sim_root_comments'] = len(sim_tree['replies'])
	#can get other = total - root in post-processing

	#true pos = node is in the right place and at the right time (+/- error) = match
	#but don't count the observed comments or the root, because that would be cheating
	eval_res['true_pos'] = eval_res['match_count'] - observed_comment_count - 1
	#false pos = put a node where there shouldn't be one = remove
	eval_res['false_pos'] = eval_res['remove_count']
	#false neg = didn't put a node where there should be one = insert
	eval_res['false_neg'] = eval_res['insert_count']
	#time update nodes - we put a node in the correct topological place, but not the right temporal place
	#consider this both a false pos and a false neg
	eval_res['false_pos'] += eval_res['update_count']
	eval_res['false_neg'] += eval_res['update_count']

	#precision and recall/sensitivity (no accuracy, because no true neg)
	try:
		eval_res['precision'] = eval_res['true_pos'] / (eval_res['true_pos'] +  eval_res['false_pos'])
	except ZeroDivisionError:
		eval_res['precision'] = 0
	try:
		eval_res['recall'] = eval_res['true_pos'] / (eval_res['true_pos'] +  eval_res['false_neg'])
	except ZeroDivisionError:
		eval_res['recall'] = 0

	#f1 score
	try:
		eval_res['f1'] = 2 * (eval_res['recall'] * eval_res['precision']) / (eval_res['recall'] + eval_res['precision'])
	except ZeroDivisionError:
		eval_res['f1'] = 0

	return eval_res
#end eval_trees



#save all sim results to csv file
#one row per simulated post/time pair, with a bunch of data in it
#then, at the bottom, all the settings/arguments, for tracking purposes
def save_results(filename, metrics, avg_metrics, input_sim_post, time_observed, subreddit, min_node_quality, max_graph_size, min_weight, testing_start_month, testing_start_year, testing_len, training_start_month, training_start_year, training_len, edge_weight_method, include_hardcoded_posts, estimate_initial_params, time_error_margin, error_method):
	#dump metrics dict to file, enforcing a semi-meaningful order
	fields = ["post_id", "param_source", "time_observed", "true_comment_count", "observed_comment_count", "simulated_comment_count", "true_root_comments", "sim_root_comments", "true_depth", "true_breadth", "simulated_depth", "simulated_breadth", "f1", "precision", "recall", "true_pos", "false_pos", "false_neg", "dist", "remove_count", "remove_time", "insert_count", "insert_time", "update_count", "update_time", "match_count", "disconnected"]
	file_utils.save_csv(metrics, filename, fields)

	#dump average metrics after that
	with open(filename, 'a') as file:
		file.write("\nAverage by time observed\n")		#header/label
	#convert avg metrics from nested dict to list of dict
	avg_metrics = [avg_metrics[time] for time in avg_metrics.keys()]
	#remove a couple unneeded fields for this dump
	fields.remove("post_id")
	fields.remove("disconnected")
	#save avg metrics
	file_utils.save_csv(avg_metrics, filename, fields, file_mode='a')
	
	#append arguments/settings to the end
	with open(filename, "a") as file:
		file.write("\nSettings\n")
		file.write("sim_post,%s\n" % input_sim_post)
		file.write("time_observed,%s\n" % time_observed)
		file.write("subreddit,%s\n" % subreddit)
		file.write("min_node_quality,%s\n" % min_node_quality)
		file.write("max_graph_size,%s\n" % max_graph_size)
		file.write("min_edge_weight,%s\n" % min_weight)
		file.write("testing_period,%d-%d\n" % (testing_start_month, testing_start_year))
		file.write("test_len,%s\n" % testing_len)
		file.write("training_period,%d-%d\n" % (training_start_month, training_start_year))
		file.write("train_len,%s\n" % training_len)
		file.write("edge_weight_method,%s\n" % edge_weight_method)
		file.write("include_default_params_posts,%s\n" % include_hardcoded_posts)
		file.write("estimate_initial_params,%s\n" % estimate_initial_params)
		file.write("allowable time error,%s\n" % time_error_margin)
		file.write("time error method,%s\n" % error_method)

	return
#end save_results


#OLD LEGACY CODE: not using this method anymore, but keeping it around for now just in case
#given simulated and ground-truth cascades, compute the tree edit distance between them
#both trees given as dictionary-nested structure (returned from simulate_comment_tree and convert_comment_tree)
def eval_trees_edit_dist(post_id, sim_tree, true_cascade, simulated_comment_count, observed_comment_count, true_comment_count, time_observed, disconnected):
	#get edit distance stats for sim vs truth
	eval_res = tree_edit_distance.compare_trees(sim_tree, true_cascade)

	#normalize the distance value: error / max error if sim nothing
	#max error if sim nothing = true comment count - observed comment count
	#ie, can only be wrong by how many comments are missing (in theory)
	norm_error = true_comment_count - observed_comment_count	
	if norm_error == 0:
		norm_error = 1
	eval_res['norm_dist'] = eval_res['dist'] / norm_error

	#add more fields to the results dictionary
	eval_res['post_id'] = post_id
	eval_res['observed_comment_count'] = observed_comment_count
	eval_res['true_comment_count'] = true_comment_count
	eval_res['simulated_comment_count'] = simulated_comment_count
	eval_res['disconnected'] = "True" if disconnected else "False"
	eval_res['time_observed'] = time_observed

	return eval_res
#end eval_trees_edit_dist
