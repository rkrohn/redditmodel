#new model for the cascade prediction/scenario 2 paper

#given a single cascade (top-level post and any observed comments), simulate the remainder of the cascade
#offload node2vec to c++, because speed

#requires the following command-line args: subreddit (or hackernews or cve), id of cascade post to predict (or "random"), output filename for simulation results (no extension), max number of nodes for infer graph, minimum node quality for graph inference (set to -1 for no filter), esp (optional, for estimating initial params based on surrouding)

#for example:
#	python3 paper_model.py pivx random 4 sim_tree 2000 -1
#	python3 paper_model.py pivx 26RPcnyIuA0JyQpTqEui7A 1 sim_tree 500 -1			(4 comments)
#	paper_model.py pivx ZeuF7ZTDw3McZUOaosvXdA 5 sim_tree 250 -1					(11 comments)
#	paper_model.py compsci qOjspbLmJbLMVFxYbjB1mQ 200 sim_tree 250 -1				(58 comments)


import file_utils
import functions_gen_cascade_model
import cascade_manip
import fit_partial_cascade

import time
from collections import defaultdict

print("")

#parse all command-line arguments
subreddit, input_sim_post_id, time_observed_list, outfile, max_nodes, min_node_quality, estimate_initial_params, batch, random, testing_start_month, testing_start_year, testing_len, training_start_month, training_start_year, training_len, weight_method, top_n, weight_threshold, include_default_posts, verbose = functions_gen_cascade_model.parse_command_args()

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
test_posts = functions_gen_cascade_model.verify_post_set(input_sim_post_id, batch, random, test_posts)
#reduce cascades to match this set
if len(test_posts) != len(test_cascades):
	test_cascades = functions_gen_cascade_model.filter_dict_by_list(test_cascades, list(test_posts.keys()))

#build graph for training set - will add infer posts later
base_graph, graph_post_ids = functions_gen_cascade_model.build_base_graph(train_posts, train_params, train_fit_fail_list, include_default_posts, max_nodes, min_node_quality, weight_method, weight_threshold, top_n)
vprint("")

#if running in mode all, keep total of all metrics, dump at end
if batch or len(time_observed_list) > 1:
	total_metrics = defaultdict(lambda: defaultdict(int))		#time observed -> metrics dictionary

filename_id = str(time.time())		#unique temp file identifier for this run

#process all posts (or just one, if doing that)
vprint("Processing", len(test_posts), "post", "s" if len(test_posts) > 1 else "")
for sim_post_id, sim_post in test_posts.items():

	if batch == False:
		vprint("Simulation post has %d comments" % test_cascades[sim_post_id]['comment_count_total'])


	#GRAPH INFER
	inferred_params = functions_gen_cascade_model.graph_infer(sim_post, sim_post_id, weight_method, weight_threshold, base_graph, graph_post_ids, train_posts, train_cascades, train_params, train_fit_fail_list, top_n, estimate_initial_params, filename_id)
	#inferred_params = [1.73166, 0.651482, 1.08986, 0.762604, 2.49934, 0.19828]		#placeholder if skipping the infer
	if batch == False:
		vprint("Inferred params: ", inferred_params, "\n")


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
		sim_tree = functions_gen_cascade_model.simulate_comment_tree(sim_post, sim_params, subreddit, test_cascades[sim_post_id], time_observed)


		#OUTPUT TREES

		#for now, only output if doing a single post
		'''
		if batch == False:
			#save groundtruth cascade to csv
			functions_gen_cascade_model.save_groundtruth(sim_post, post_comments, outfile)

			#save sim results to json - all simulated events plus some simulation parameters
			functions_gen_cascade_model.save_sim_json(subreddit, sim_post_id, random_post, time_observed, min_node_quality, max_nodes, estimate_initial_params, sim_events, outfile)

			#save sim results to second output file - csv, one event per row, columns 'rootID', 'nodeID', and 'parentID' for now
			print("Saving results to", outfile + ".csv...")  
			file_utils.save_csv(sim_events, outfile+".csv", fields=['rootID', 'nodeID', 'parentID'])
			print("")
		'''


		#EVAL

		#get time-shifted ground-truth cascade
		true_cascade, true_comment_count = functions_gen_cascade_model.filter_comment_tree(sim_post, test_cascades[sim_post_id])

		#compute tree edit distance between ground-truth and simulated cascades
		eval_res = functions_gen_cascade_model.eval_trees(sim_tree, true_cascade)

		if batch == False and len(time_observed_list) == 1:
			vprint("Tree edit distance:", eval_res['dist'])
			vprint("   update:", eval_res['update_count'], eval_res['update_time'])
			vprint("   insert:", eval_res['insert_count'], eval_res['insert_time'])
			vprint("   remove:", eval_res['remove_count'], eval_res['remove_time'])
			vprint("   match:", eval_res['match_count'])

		#if running in mode all, or multiple times, keep total of all these metrics, dump at end
		else:
			for metric, value in eval_res.items():
				total_metrics[time_observed][metric] += value

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

	for time_observed in time_observed_list:
		print("\nObserved time: %f" % time_observed)
		print("Tree edit distance:", total_metrics[time_observed]['dist'])
		print("   update:", total_metrics[time_observed]['update_count'], total_metrics[time_observed]['update_time'])
		print("   insert:", total_metrics[time_observed]['insert_count'], total_metrics[time_observed]['insert_time'])
		print("   remove:", total_metrics[time_observed]['remove_count'], total_metrics[time_observed]['remove_time'])
		print("   match:", total_metrics[time_observed]['match_count'])
	print("")

	#dump metrics dict to file
	file_utils.verify_dir(outfile)
	file_utils.dict_to_csv(total_metrics, ['time_observed']+list(total_metrics[time_observed_list[0]].keys()), outfile+("/%s_metric_results.csv"%filename_id))
