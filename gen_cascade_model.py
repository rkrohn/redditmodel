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


#parse all command-line arguments
subreddit, input_sim_post, observing_time, observed_list, outfile, max_nodes, min_node_quality, estimate_initial_params, normalize_parameters, batch, sample_num, testing_start_month, testing_start_year, testing_len, training_start_month, training_start_year, training_len, weight_method, remove_stopwords, top_n, weight_threshold, include_default_posts, time_error_margin, error_method, sanity_check, min_size, max_size, get_training_stats, get_testing_stats, socsim_data, graph_downsample_ratio, large_cascade_demarcation, verbose, preprocess = functions_gen_cascade_model.parse_command_args()

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
		train_posts, train_cascades, train_params, train_fit_fail_list = functions_gen_cascade_model.load_processed_posts(subreddit, training_start_month, training_start_year, training_len, load_params=True, load_cascades=True)

	vprint("\nLoading processed testing data")
	#load pre-processed posts and their reconstructed cascades for testing period (no params here!)
	if not sanity_check:
		test_posts, test_cascades = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_len, load_cascades=True)
	#if simming from fitted params, load testing params
	else:
		test_posts, test_cascades, test_params, test_fit_fail_list = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_len, load_params=True, load_cascades=True)

#ensure post id is in dataset (and filter test_posts set down to processing group only)
vprint("")
test_posts = functions_gen_cascade_model.get_test_post_set(input_sim_post, batch, min_size, max_size, sample_num, test_posts, test_cascades, subreddit, testing_start_month, testing_start_year, testing_len)
#reduce cascades to match this set
if len(test_posts) != len(test_cascades):
	test_cascades = functions_gen_cascade_model.filter_dict_by_list(test_cascades, list(test_posts.keys()))

#if want training data stats, get those now
if get_training_stats:
	vprint("Computing training data stats")
	functions_gen_cascade_model.output_post_set_stats(train_cascades, subreddit, training_start_year, training_start_month, training_len)
#if want testing data stats, get those now
if get_testing_stats:
	vprint("Computing testing data stats")
	functions_gen_cascade_model.output_post_set_stats(test_cascades, subreddit, testing_start_year, testing_start_month, testing_len)

#build base graph for training set - will add infer posts later
if not sanity_check:
	base_graph, graph_post_ids = functions_gen_cascade_model.build_base_graph(train_cascades, train_posts, train_params, train_fit_fail_list, subreddit, training_start_year, training_start_month, training_len, include_default_posts, max_nodes, min_node_quality, weight_method, remove_stopwords, weight_threshold, top_n, graph_downsample_ratio, large_cascade_demarcation)
vprint("")

if preprocess:
	vprint("Finished graph build, exiting")
	exit(0)

all_metrics = []		#keep all metrics, separate for each post/observed time run, dump them all at the end
filename_id = str(time.time())		#unique temp file identifier for this run

#process all posts (or just one, if doing that)
post_count = 0
disconnected_count = 0
vprint("Processing %d post" % len(test_posts) + ("s" if len(test_posts) > 1 else ""))
for sim_post_id, sim_post in test_posts.items():

	if batch == False:
		vprint("Simulation post has %d comments" % test_cascades[sim_post_id]['comment_count_total'])

	#GRAPH INFER
	if not sanity_check:
		inferred_params, disconnected, new_edges = functions_gen_cascade_model.graph_infer(sim_post, sim_post_id, weight_method, weight_threshold, base_graph, graph_post_ids, train_posts, train_cascades, train_params, train_fit_fail_list, top_n, estimate_initial_params, normalize_parameters, filename_id, display= not batch)
		if batch == False:
			vprint("Inferred params: ", inferred_params, "\n")

		#keep count of disconnected (bad results, probably) posts
		if disconnected:
			disconnected_count += 1
	else:
		new_edges = None

	#get time-shifted ground-truth cascade (same for all observation periods)
	true_cascade, true_comment_count = functions_gen_cascade_model.filter_comment_tree(test_cascades[sim_post_id])
	#and compute the structural virality of this cascade
	true_structural_virality = functions_gen_cascade_model.get_structural_virality(true_cascade)

	#use the same inferred params for all the time_observed values
	for observed in observed_list:			

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
			partial_fit_params, observed_comment_count = fit_partial_cascade.fit_partial_cascade(test_cascades[sim_post_id], observed, observing_time, inferred_params, verbose=(verbose if batch==False else False))
			#verify we got good params back - if not, skip this post entirely
			if partial_fit_params == False:
				vprint("Partial fit failed for", sim_post_id, "- skipping post")
				post_count -= 1		#don't count this post (counter incremented at bottom of loop)
				break

			#if we have observed the whole cascade and max observed is not 0, 
			#don't bother simming for this observed setting
			if observed != 0 and observed_comment_count >= test_cascades[sim_post_id]['comment_count_total']:
				continue

			sim_params = partial_fit_params			#refined params from partial fit for sim

		if not batch:
			vprint("Simulation params: ", sim_params)

		#SIMULATE COMMENT TREE
		sim_tree, observed_count, observed_time, simulated_count = functions_gen_cascade_model.simulate_comment_tree(sim_params, subreddit, test_cascades[sim_post_id], observed, observing_time, not batch)

		#don't try to eval if sim failed (aborted infinite sim)
		if sim_tree == False:
			continue

		#EVAL

		#already got ground-truth cascade above

		#get sim cascade as networkx graph
		#sim_graph = functions_gen_cascade_model.cascade_to_graph(sim_tree)

		#compute tree edit distance between ground-truth and simulated cascades
		eval_res = functions_gen_cascade_model.eval_trees(sim_post_id, sim_tree, true_cascade, simulated_count, observed_count, true_comment_count, true_structural_virality, observed_time, observing_time, time_error_margin, error_method, disconnected, new_edges, (observed if observing_time==False else None))
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

	#counter and periodic prints
	post_count += 1
	if batch and post_count % 50 == 0:
		vprint("   finished %d posts (%d disconnected)" % (post_count, disconnected_count))

#all done, print final disconnected count
vprint("Finished simulating %d posts (%d disconnected)" % (post_count, disconnected_count))

if post_count == 0:
	vprint("\nNo posts simulated, no results to save\n")
	exit(0)

#save metrics + settings to output file
functions_gen_cascade_model.save_results(outfile, all_metrics, input_sim_post, sample_num, observed_list, observing_time, subreddit, min_node_quality, max_nodes, weight_threshold, testing_start_month, testing_start_year, testing_len, training_start_month, training_start_year, training_len, weight_method, include_default_posts, estimate_initial_params, time_error_margin, error_method, min_size, max_size)
