#comparative model for the cascade prediction/scenario 2 paper

#given a single cascade (top-level post and any observed comments), simulate the remainder of the cascade

#instead of doing param infer and fit refine, fit params to just the observed comments, and sim from there


import file_utils
import functions_gen_cascade_model
import socsim_data_functions_gen_cascade_model
import functions_baseline_model
import functions_comparative_hawkes

import random
from copy import deepcopy


#parse all command-line arguments
subreddit, input_sim_post, observing_time, observed_list, outfile, batch, testing_num, testing_start_month, testing_start_year, training_num, time_error_margin, error_method, min_size, max_size, socsim_data, verbose = functions_baseline_model.parse_command_args()

if observing_time == False:
	print("Can't observe by number of comments now, sorry")
	exit(0)

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
#and make sure the regular model functions have this too
functions_gen_cascade_model.define_vprint(verbose)	

#ensure data directory for this subreddit exists - for saving posts, cascades, params, etc
file_utils.verify_dir("reddit_data/%s" % subreddit)

#if using socsim data, special load process (no time-defined sets)
if socsim_data:
	socsim_data_functions_gen_cascade_model.define_vprint(verbose)		#define vprint for that function class\
	#load all the training and testing data for this domain
	train_posts, train_cascades, train_params, train_fit_fail_list, test_posts, test_cascades, test_params, test_fit_fail_list = socsim_data_functions_gen_cascade_model.load_data(subreddit)

#otherwise, standard data load (use month-year and lengths to define testing set)
else:
	#no training data, since just using observed comments for this cascade

	vprint("\nLoading processed testing data")
	#load pre-processed posts and their reconstructed cascades for testing period (no params here!)
	test_posts, test_cascades = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_num, load_cascades=True)

#ensure post id is in dataset (and filter test_posts set down to processing group only)
vprint("")
test_posts = functions_gen_cascade_model.get_test_post_set(input_sim_post, min_size, max_size, testing_num, test_posts, test_cascades, subreddit, testing_start_month, testing_start_year)
#reduce cascades to match this set
if len(test_posts) != len(test_cascades):
	test_cascades = functions_gen_cascade_model.filter_dict_by_list(test_cascades, list(test_posts.keys()))

all_metrics = []		#keep all metrics, separate for each post/observed time run, dump them all at the end

#how often do we want to dump? every 100 tests or so
#100 / number of observation settings = number of posts to finish before dumping
dump_count = 100 // len(observed_list) + (100 % len(observed_list) > 0) 
if dump_count == 0: dump_count = 100	#make sure not modding by 0 if small run

#load list of finished posts for this run, so we can skip ones that are already done
#(if no bookmark, will get back empty set and False flag)
finished_posts, complete = functions_gen_cascade_model.load_bookmark(outfile)
#if finished all posts already, exit
if complete:
	vprint("Entire post set already simulated, exiting")
	exit(0)
else: vprint("Skipping %d already simulated posts" % len(finished_posts))

#process all posts (or just one, if doing that)
post_count = 0
vprint("Processing %d post" % len(test_posts) + ("s" if len(test_posts) > 1 else ""))
for sim_post_id, sim_post in test_posts.items():

	#skip this post if we've already done it
	if sim_post_id in finished_posts:
		continue

	if batch == False:
		vprint("Simulation post has %d comments" % test_cascades[sim_post_id]['comment_count_total'])

	#get time-shifted ground-truth cascade (same for all observation periods)
	true_cascade, true_comment_count = functions_gen_cascade_model.shift_comment_tree(test_cascades[sim_post_id])
	#and compute the structural virality of this cascade
	true_structural_virality = functions_gen_cascade_model.get_structural_virality(true_cascade)

	#duplicate the true cascade - will use as a working copy for different observed trees
	observed_tree = deepcopy(true_cascade)

	#convert cascade to networkx graph
	graph = functions_gen_cascade_model.cascade_to_graph(test_cascades[sim_post_id])

	#pull root of tree
	root, root_creation_time = functions_comparative_hawkes.get_root(graph)

	#loop observed
	for observed in sorted(observed_list, reverse=True):

		print("observed", observed, "h")

		#get truncated graph, so we know how many comments observed
		#remove unobserved comments from base tree, so we can simulate from partially observed tree
		#observation defined by time
		if observing_time:
			#get observed graph based on observation time and comment timestamps
			given_tree = functions_comparative_hawkes.get_trunc_tree_no_relabel(graph, observed*60)		#pass observed time in minutes
			observed_count = len(given_tree) - 1	#observed comments = graph nodes - 1
			print(observed_count, "comments")
			#set observed time equal to given for sim
			time_observed = observed
		#observation defined by number of comments
		else:
			print("no observing by comments, sorry")
			exit(0)

		#if we have observed the whole cascade and max observed is not 0, 
		#don't bother simming for this observed setting
		if observed != 0 and observed_count >= test_cascades[sim_post_id]['comment_count_total']:
			continue

		#SIMULATE COMMENT TREE
		#sim_tree, simulated_count = functions_gen_cascade_model.simulate_comment_tree(sim_params, observed_tree, time_observed, not batch)

		#break if size of the observed tree is too small for prediction at that moment
		if len(given_tree) <= 10:  
			print("Not enough data for parameters estimation!")
			negative_result = 0
			#list_hawkes_sizes[t].append(negative_result)
			continue

		#fit the weibull based on root comment times
		root_comment_times = functions_comparative_hawkes.get_root_comment_times(given_tree)
		mu_params = functions_comparative_hawkes.mu_parameters_estimation(root_comment_times)		
		if mu_params == None:  # if loglikelihood estimation fails - use curve_fit
			mu_params = functions_comparative_hawkes.mu_func_fit_weibull(root_comment_times)
		print("Mu_params:", mu_params)

		#fit log-normal based on all other comment times
		other_comment_times = functions_comparative_hawkes.get_other_comment_times(given_tree)
		phi_params = functions_comparative_hawkes.phi_parameters_estimation(other_comment_times)
		print("Phi_params:", phi_params)

		#estimate branching factor (average number of replies per comment)
		n_b = functions_comparative_hawkes.nb_parameters_estimation(given_tree, root)
		print("n_b:", n_b)
	    
		hawkes_times = []
		given_tree = functions_comparative_hawkes.get_trunc_tree(graph, observed*60)	#filter tree, but this time set timestamps to minutes

		sim_graph, success = functions_comparative_hawkes.simulate_comment_tree(given_tree, observed*60, mu_params, phi_params, n_b)	#simulate!
		simulated_count = len(sim_graph) - 1	#number of sim comments = size of graph - 1

		#don't try to eval if sim failed
		if success == False:
			print('Generation failed! Too many nodes')
			print("RUN " + str(i) + ": T_learn: "+ t_learn_list[t] + ': Generation HAWKES failed! Too many nodes')
			#list_hawkes_sizes[t] = [-1]


		#EVAL

		#already got ground-truth cascade above

		#get sim networkx graph as cascade
		sim_tree = functions_gen_cascade_model.graph_to_cascade(sim_graph)

		#compute tree edit distance between ground-truth and simulated cascades
		eval_res = functions_baseline_model.eval_trees(sim_post_id, sim_tree, true_cascade, simulated_count, observed_count, true_comment_count, true_structural_virality, time_observed, observing_time, time_error_margin, error_method, (observed if observing_time==False else None))
		#add a column indicating where the params for this sim came from
		eval_res['param_source'] = "fitted"
		#dummy columns (graph infer) so output format matches real model exactly
		eval_res['disconnected'] = "N/A"
		eval_res['connecting_edges'] = "N/A"

		#append eval data to overall list
		all_metrics.append(eval_res)

	#counter and periodic prints
	post_count += 1
	finished_posts.add(sim_post_id)
	if batch and post_count % 100 == 0:
		vprint("   finished %d posts" % post_count)

	#dump results every 10%, to save memory
	'''
	if batch and post_count % dump_count == 0:
		vprint("   saving results so far (%d posts)" % post_count)
		#append new results to running csv
		functions_gen_cascade_model.save_results(outfile, all_metrics, observing_time)
		all_metrics.clear()		#clear out what we already saved
		#and save pickle bookmark: set of finished posts and current status
		functions_gen_cascade_model.save_bookmark(finished_posts, outfile)
		#don't clear that list, want it to contain everything
	'''	

#all done, print final disconnected count
vprint("Finished simulating %d posts" % post_count)

if post_count == 0:
	vprint("\nNo posts simulated, no results to save\n")
	exit(0)

#save metrics + settings to output file
functions_gen_cascade_model.save_results(outfile, all_metrics, observing_time)

#all done, update bookmark to "finished"
functions_gen_cascade_model.save_bookmark(finished_posts, outfile, status=(True if len(finished_posts) == len(test_posts) else False))

vprint("All done, all results saved\n")
