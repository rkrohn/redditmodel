#new model for the cascade prediction/scenario 2 paper

#given a single cascade (top-level post and any observed comments), simulate the remainder of the cascade
#offload node2vec to c++, because speed


import file_utils
import functions_gen_cascade_model
import socsim_data_functions_gen_cascade_model
import cascade_manip
import fit_partial_cascade

import time
from collections import defaultdict
from copy import deepcopy


#parse all command-line arguments
subreddit, input_sim_post, observing_time, observed_list, outfile, max_nodes, min_node_quality, binary_quality, estimate_initial_params, normalize_parameters, batch, testing_num, testing_start_month, testing_start_year, training_num, weight_method, remove_stopwords, top_n, weight_threshold, include_default_posts, time_error_margin, error_method, sanity_check, min_size, max_size, get_training_stats, get_testing_stats, get_sub_stats, socsim_data, graph_downsample_ratio, large_cascade_demarcation, verbose, preprocess, output_timestamps = functions_gen_cascade_model.parse_command_args()

#hackery: declare a special print function for verbose output
if verbose:
	def vprint(*args):
		# Print each argument separately so caller doesn't need to
		# stuff everything to be printed into a single string
		for arg in args:
			print(arg, end='')
		print("")
else:   
	vprint = lambda *a: None      # do-nothing function

#ensure working directory exists - for saving of intermediate graph/param files for node2vec
file_utils.verify_dir("sim_files")	
#ensure data directory for this subreddit exists - for saving posts, cascades, params, etc
file_utils.verify_dir("reddit_data/%s" % subreddit)

#get sub stats if desired
if get_sub_stats:
	functions_gen_cascade_model.output_subreddit_stats(subreddit, testing_start_year)
	exit(0)

#if using socsim data, special load process (no time-defined sets)
if socsim_data:
	socsim_data_functions_gen_cascade_model.define_vprint(verbose)		#define vprint for that function class\
	#load all the training and testing data for this domain
	train_posts, train_cascades, train_params, train_fit_fail_list, test_posts, test_cascades, test_params, test_fit_fail_list = socsim_data_functions_gen_cascade_model.load_data(subreddit)

#otherwise, standard data load (use month-year and lengths to define training and testing sets)
else:
	#load pre-processed posts and their fitted params for training period
	if not sanity_check:
		vprint("Loading processed training data")
		train_posts, train_cascades, train_params, train_fit_fail_list = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, training_num, load_params=True, load_cascades=True, load_forward=False)

	vprint("\nLoading processed testing data")
	#load pre-processed posts and their reconstructed cascades for testing period (no params here!)
	if not sanity_check:
		test_posts, test_cascades = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_num, load_cascades=True)
	#if simming from fitted params, load testing params
	else:
		test_posts, test_cascades, test_params, test_fit_fail_list = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_num, load_params=True, load_cascades=True)

#ensure post id is in dataset (and filter test_posts set down to processing group only)
vprint("")
test_posts = functions_gen_cascade_model.get_test_post_set(input_sim_post, min_size, max_size, testing_num, test_posts, test_cascades, subreddit, testing_start_month, testing_start_year)
#reduce cascades to match this set
if len(test_posts) != len(test_cascades):
	test_cascades = functions_gen_cascade_model.filter_dict_by_list(test_cascades, list(test_posts.keys()))

#also sample/filter the train set to the desired number
vprint("Sampling %d last posts (from %d) for simulation set" % (training_num, len(train_posts.keys())))
#get list of n keys to keep - last n posts in loaded train data
train_keys = functions_gen_cascade_model.sample_chronologically(train_posts, training_num, forward=False)
#filter posts to match corresponding keys list
train_posts = functions_gen_cascade_model.filter_dict_by_list(train_posts, train_keys)
#and reduce cascades and params to match this set
train_cascades = functions_gen_cascade_model.filter_dict_by_list(train_cascades, train_keys)
train_params = functions_gen_cascade_model.filter_dict_by_list(train_params, train_keys)
train_fit_fail_list = [post_id for post_id in train_fit_fail_list if post_id in train_keys]

#drop a random training cascade to a pickle (for dev/testing purposes)
'''
import random
rand_id, rand_cascade = random.choice(list(train_cascades.items()))
while rand_cascade['comment_count_total'] < 100:
	rand_id, rand_cascade = random.choice(list(train_cascades.items()))
file_utils.save_pickle(rand_cascade, "test_cascade.pkl")
'''

#if want training data stats, get those now
if get_training_stats:
	vprint("Computing training data stats")
	functions_gen_cascade_model.output_post_set_stats(train_cascades, subreddit, testing_start_year, testing_start_month, "train", training_num, observed_list, train_params)
#if want testing data stats, get those now
if get_testing_stats:
	vprint("Computing testing data stats")
	functions_gen_cascade_model.output_post_set_stats(test_cascades, subreddit, testing_start_year, testing_start_month, "test", testing_num, observed_list)

#build base graph for training set - will add infer posts later
if not sanity_check:
	base_graph, graph_post_ids = functions_gen_cascade_model.build_base_graph(train_cascades, train_posts, train_params, train_fit_fail_list, subreddit, testing_start_year, testing_start_month, training_num, include_default_posts, max_nodes, min_node_quality, weight_method, remove_stopwords, weight_threshold, top_n, graph_downsample_ratio, large_cascade_demarcation)
vprint("")

if preprocess:
	vprint("Finished graph build, exiting")
	exit(0)

all_metrics = []		#keep all metrics, separate for each post/observed time run, dump them all at the end
filename_id = str(time.time())		#unique temp file identifier for this run - node2vec graph/param files

#for outputting comment timestamps
timestamps = {} 	#post_id -> time (or true) -> list of timestamps

#how often do we want to dump? every 20 tests or so
#20 / number of observation settings = number of posts to finish before dumping
dump_count = 20 // len(observed_list) + (20 % len(observed_list) > 0) 
if dump_count == 0: dump_count = 20 	#make sure not modding by 0 if small run

#load list of finished posts for this run, so we can skip ones that are already done
#(if no bookmark, will get back empty set and False flag)
finished_posts, complete = functions_gen_cascade_model.load_bookmark(outfile)
#if finished all posts already, exit
if complete:
	vprint("Entire post set already simulated, exiting")
	exit(0)
else: vprint("Skipping %d already simulated posts" % len(finished_posts))

#sort the observed list - largest to smallest
observed_list = sorted(observed_list, reverse=True)

#process all posts (or just one, if doing that)
post_count = 0
disconnected_count = 0
vprint("Processing %d post" % len(test_posts) + ("s" if len(test_posts) > 1 else ""))
for sim_post_id, sim_post in test_posts.items():

	#skip this post if we've already done it
	if sim_post_id in finished_posts:
		continue

	if batch == False:
		vprint("Simulation post has %d comments" % test_cascades[sim_post_id]['comment_count_total'])

	#GRAPH INFER
	if not sanity_check:
		inferred_params, disconnected, new_edges = functions_gen_cascade_model.graph_infer(sim_post, sim_post_id, weight_method, weight_threshold, base_graph, graph_post_ids, train_posts, train_cascades, train_params, train_fit_fail_list, top_n, estimate_initial_params, normalize_parameters, binary_quality, filename_id, display= not batch)
		if batch == False:
			vprint("Inferred params: ", inferred_params, "\n")

		#keep count of disconnected (bad results, probably) posts
		if disconnected:
			disconnected_count += 1
	else:
		new_edges = None

	#get time-shifted ground-truth cascade (same for all observation periods)
	true_cascade, true_comment_count = functions_gen_cascade_model.shift_comment_tree(test_cascades[sim_post_id])
	#and compute the structural virality of this cascade
	true_structural_virality = functions_gen_cascade_model.get_structural_virality(true_cascade)

	if output_timestamps:
		#get list of true comment timestamps in minutes
		true_comment_timestamps = sorted(functions_gen_cascade_model.get_list_of_comment_times(true_cascade))
		#add to output list
		timestamps[sim_post_id] = {}
		timestamps[sim_post_id]["true"] = true_comment_timestamps

	#duplicate the true cascade - will use as a working copy for different observed trees
	observed_tree = deepcopy(true_cascade)

	#if observing by comments, reduce observation list if necessary - so we can make sure not to sim from a complete tree more than once
	if observing_time == False and observed_list[0] >= true_comment_count and len(observed_list) > 1 and observed_list[1] >= true_comment_count:
		#reduce list by removing all but the smallest observation setting that observes the whole tree
		#(yes, this is ugly)
		post_observed_list = deepcopy(observed_list)
		while post_observed_list[0] >= true_comment_count and len(post_observed_list) > 1 and post_observed_list[1] >= true_comment_count:
			del post_observed_list[0]
	#not observing the whole post more than once, or observing by time, use the unedited list
	else:	
		post_observed_list = deepcopy(observed_list)

	#use the same inferred params for all the time_observed values
	#loop observed settings from largest to smallest - so we shrink the observed tree
	#(list should already be sorted - but just to be sure...)
	for observed in sorted(post_observed_list, reverse=True):

		#remove unobserved comments from base tree, so we can simulate from partially observed tree
		#observation defined by time
		if observing_time:
			#get observed tree based on observation time and comment timestamps
			observed_tree, observed_count = functions_gen_cascade_model.filter_comment_tree(observed_tree, observed*60)	#pass in time in minutes
			#set observed time equal to given for sim
			time_observed = observed
		#observation defined by number of comments
		else:
			observed_tree, observed_count, time_observed = functions_gen_cascade_model.filter_comment_tree_by_num_comments(observed_tree, observed)

		#which params are we using for simulation?

		#if sanity check, grab fitted or default params
		if sanity_check:
			if sim_post_id in test_params:
				sim_params = test_params[sim_post_id]
				#if hole in fitted params (part of fit failed), fill that in with default
				if not all(sim_params):
					sim_params = functions_gen_cascade_model.get_complete_params(test_cascades[sim_post_id], sim_params)
			else:
				sim_params = functions_gen_cascade_model.get_default_params(test_cascades[sim_post_id])
			disconnected = False 		#set this for output
			
		#real sim, get refined params
		else:
			#REFINE PARAMS - for partial observed trees
			partial_fit_params, observed_comment_count = fit_partial_cascade.fit_partial_cascade(observed_tree, inferred_params, verbose=(verbose if batch==False else False))
			#verify we got good params back - if not, skip this post entirely
			if partial_fit_params == False:
				vprint("Partial fit failed for ", sim_post_id, " - skipping post")
				post_count -= 1		#don't count this post (counter incremented at bottom of loop)
				break
			sim_params = partial_fit_params			#refined params from partial fit for sim

		if not batch:
			vprint("Simulation params: ", sim_params)

		#SIMULATE COMMENT TREE
		sim_tree, simulated_count = functions_gen_cascade_model.simulate_comment_tree(sim_params, observed_tree, observed_count, time_observed*60.0, not batch)

		#don't try to eval if sim failed (aborted infinite sim)
		if sim_tree == False:
			print("infinite sim aborted, skipping post", sim_post_id)
			continue

		#EVAL

		#already got ground-truth cascade above

		#get sim cascade as networkx graph
		#sim_graph = functions_gen_cascade_model.cascade_to_graph(sim_tree)

		#compute tree edit distance between ground-truth and simulated cascades
		eval_res = functions_gen_cascade_model.eval_trees(sim_post_id, sim_tree, true_cascade, simulated_count, observed_count, true_comment_count, true_structural_virality, time_observed, observing_time, time_error_margin, error_method, disconnected, new_edges, (observed if observing_time==False else None))
		#add a column indicating where the params for this sim came from
		if not sanity_check:
			if observed == 0:
				eval_res['param_source'] = "infer"
			else:
				eval_res['param_source'] = "infer+observed_fit"
		elif sim_post_id in test_params and all(test_params[sim_post_id]):
			eval_res['param_source'] = "fitted"
		elif sim_post_id in test_params:
			eval_res['param_source'] = "incomplete_fitted"
		else:
			eval_res['param_source'] = "default"

		#append eval data to overall list
		all_metrics.append(eval_res)

		#tracking timestamps? handle that here
		if output_timestamps:
			#already have true times - get simulated
			sim_comment_timestamps = sorted(functions_gen_cascade_model.get_list_of_comment_times(sim_tree))
			#add to dict
			timestamps[sim_post_id][observed] = sim_comment_timestamps

	#counter and periodic prints
	post_count += 1
	finished_posts.add(sim_post_id)
	if batch and post_count % 100 == 0:
		vprint("   finished %d posts (%d disconnected)" % (post_count, disconnected_count))

	#dump results every 10%, to save memory
	if batch and post_count % dump_count == 0:
		vprint("   saving results so far (%d posts)" % post_count)
		#append new results to running csv
		functions_gen_cascade_model.save_results(outfile, all_metrics, observing_time)
		all_metrics.clear()		#clear out what we already saved
		#and save pickle bookmark: set of finished posts and current status
		functions_gen_cascade_model.save_bookmark(finished_posts, outfile)
		#don't clear that list, want it to contain everything


#all done, print final disconnected count
vprint("Finished simulating %d posts (%d disconnected)" % (post_count, disconnected_count))

if post_count == 0:
	vprint("\nNo posts simulated, no results to save\n")
	exit(0)

#save metrics + settings to output file
functions_gen_cascade_model.save_results(outfile, all_metrics, observing_time)

#all done, update bookmark to "finished"
functions_gen_cascade_model.save_bookmark(finished_posts, outfile, status=(True if len(finished_posts) == len(test_posts) else False))

#if outputting timestamps, dump to pickle (hackery)
if output_timestamps:
	file_utils.save_pickle(timestamps, outfile+"_timestamps.pkl")

vprint("All done, all results saved\n")
