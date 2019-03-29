#new model for the cascade prediction/scenario 2 paper

#given a single cascade (top-level post and any observed comments), simulate the remainder of the cascade
#offload node2vec to c++, because speed


import file_utils
import functions_gen_cascade_model
import cascade_manip
import fit_partial_cascade

import time
from collections import defaultdict

print("")

#parse all command-line arguments
subreddit, input_sim_post, time_observed_list, outfile, max_nodes, min_node_quality, estimate_initial_params, batch, sample_num, testing_start_month, testing_start_year, testing_len, training_start_month, training_start_year, training_len, weight_method, top_n, weight_threshold, include_default_posts, verbose = functions_gen_cascade_model.parse_command_args()

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

#load pre-processed posts and their fitted params for training period
vprint("Loading processed training data")
train_posts, train_cascades, train_params, train_fit_fail_list = functions_gen_cascade_model.load_processed_posts(subreddit, training_start_month, training_start_year, training_len, load_params=True, load_cascades=True)

#load pre-processed posts and their reconstructed cascades for testing period (no params here!)
vprint("\nLoading processed testing data")
test_posts, test_cascades = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_len, load_cascades=True)

#ensure post id is in dataset (and filter test_posts set down to processing group only)
vprint("")
test_posts = functions_gen_cascade_model.get_test_post_set(input_sim_post, batch, sample_num, test_posts, subreddit, testing_start_month, testing_start_year, testing_len)
#reduce cascades to match this set
if len(test_posts) != len(test_cascades):
	test_cascades = functions_gen_cascade_model.filter_dict_by_list(test_cascades, list(test_posts.keys()))

#build base graph for training set - will add infer posts later
base_graph, graph_post_ids = functions_gen_cascade_model.build_base_graph(train_posts, train_params, train_fit_fail_list, include_default_posts, max_nodes, min_node_quality, weight_method, weight_threshold, top_n)
vprint("")

all_metrics = []		#keep all metrics, separate for each post/observed time run, dump them all at the end
filename_id = str(time.time())		#unique temp file identifier for this run

#process all posts (or just one, if doing that)
post_count = 0
disconnected_count = 0
vprint("Processing %d post" % len(test_posts), "s" if len(test_posts) > 1 else "")
for sim_post_id, sim_post in test_posts.items():

	if batch == False:
		vprint("Simulation post has %d comments" % test_cascades[sim_post_id]['comment_count_total'])

	#GRAPH INFER
	inferred_params, disconnected = functions_gen_cascade_model.graph_infer(sim_post, sim_post_id, weight_method, weight_threshold, base_graph, graph_post_ids, train_posts, train_cascades, train_params, train_fit_fail_list, top_n, estimate_initial_params, filename_id, display= not batch)
	if batch == False:
		vprint("Inferred params: ", inferred_params, "\n")

	#keep count of disconnected (bad results, probably) posts
	if disconnected:
		disconnected_count += 1

	#use the same inferred params for all the time_observed values
	for time_observed in time_observed_list:

		#REFINE PARAMS - for partial observed trees
		partial_fit_params = fit_partial_cascade.fit_partial_cascade(sim_post, test_cascades[sim_post_id], time_observed, inferred_params, verbose=(verbose if batch==False else False))
		if batch == False:
			vprint("Refined params: ", partial_fit_params)

		#which params are we using for simulation?
		#sim_params = inferred_params
		sim_params = partial_fit_params			#for now, always the refined params from partial fit


		#SIMULATE COMMENT TREE
		sim_tree, observed_count, simulated_count = functions_gen_cascade_model.simulate_comment_tree(sim_post, sim_params, subreddit, test_cascades[sim_post_id], time_observed, not batch)

		#EVAL

		#get time-shifted ground-truth cascade
		true_cascade, true_comment_count = functions_gen_cascade_model.filter_comment_tree(sim_post, test_cascades[sim_post_id])

		#compute tree edit distance between ground-truth and simulated cascades
		eval_res = functions_gen_cascade_model.eval_trees(sim_post_id, sim_tree, true_cascade, simulated_count, observed_count, true_comment_count, time_observed, disconnected)

		#append eval data to overall list
		all_metrics.append(eval_res)

		if batch == False and len(time_observed_list) == 1:
			vprint("Tree edit distance: ", eval_res['dist'])
			vprint("   normalized distance: ", eval_res['norm_dist'])
			vprint("   update: ", eval_res['update_count'], " ", eval_res['update_time'])
			vprint("   insert: ", eval_res['insert_count'], " ", eval_res['insert_time'])
			vprint("   remove: ", eval_res['remove_count'], " ", eval_res['remove_time'])
			vprint("   match: ", eval_res['match_count'])

	#counter and periodic prints
	post_count += 1
	if batch and post_count % 100 == 0:
		vprint("   finished %d posts (%d disconnected)" % (post_count, disconnected_count))

#if mode == all, print metric totals
if batch or len(time_observed_list) > 1:
	print("\nAll done\n")
	print("Number of posts:", len(test_posts))
	print("Time Observed:", time_observed)
	print("Source subreddit:", subreddit)
	if min_node_quality != -1:
		print("Minimum node quality:", min_node_quality)
	else:
		print("No minimum node quality")
	print("Max graph size:", max_nodes)
	if estimate_initial_params:
		print("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")

	print("")

	#dump metrics dict to file, enforcing a semi-meaningful order
	fields = ["post_id", "time_observed", "true_comment_count", "observed_comment_count", "simulated_comment_count", "dist", "norm_dist", "remove_count", "remove_time", "insert_count", "insert_time", "update_count", "update_time", "match_count", "disconnected"]
	file_utils.verify_dir(outfile)
	file_utils.save_csv(all_metrics, outfile + ("%s_%d_eval_res_start%d-%d_%d_months.csv" % (subreddit, len(test_posts), testing_start_year, testing_start_month, testing_len)), fields)
