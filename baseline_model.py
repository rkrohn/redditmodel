#baseline model for the cascade prediction/scenario 2 paper

#given a single cascade (top-level post and any observed comments), simulate the remainder of the cascade

#instead of doing param infer and fit refine, draw random params from training set and sim from those 
#for all observation settings


import file_utils
import functions_gen_cascade_model
import socsim_data_functions_gen_cascade_model
import functions_baseline_model

import random
from copy import deepcopy


#parse all command-line arguments
mode, subreddit, input_sim_post, observing_time, observed_list, outfile, batch, testing_num, testing_start_month, testing_start_year, training_num, time_error_margin, error_method, min_size, max_size, socsim_data, verbose = functions_baseline_model.parse_command_args()

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

#otherwise, standard data load (use month-year and lengths to define training and testing sets)
else:
	#load pre-processed posts and their fitted params for training period
	vprint("Loading processed training data")
	train_posts, train_cascades, train_params, train_fit_fail_list = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, training_num, load_params=True, load_cascades=True, load_forward=False)

	vprint("\nLoading processed testing data")
	#load pre-processed posts and their reconstructed cascades for testing period (no params here!)
	test_posts, test_cascades = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_num, load_cascades=True)

#ensure post id is in dataset (and filter test_posts set down to processing group only)
vprint("")
test_posts = functions_gen_cascade_model.get_test_post_set(input_sim_post, min_size, max_size, testing_num, test_posts, test_cascades, subreddit, testing_start_month, testing_start_year)
#reduce cascades to match this set
if len(test_posts) != len(test_cascades):
	test_cascades = functions_gen_cascade_model.filter_dict_by_list(test_cascades, list(test_posts.keys()))

#also sample/filter the train set to the desired number
vprint("Sampling %d last posts (from %d) for simulation set\n" % (training_num, len(train_posts.keys())))
#get list of n keys to keep - last n posts in loaded train data
train_keys = functions_gen_cascade_model.sample_chronologically(train_posts, training_num, forward=False)
#filter posts to match corresponding keys list
train_posts = functions_gen_cascade_model.filter_dict_by_list(train_posts, train_keys)
#and reduce params to match this set
train_params = functions_gen_cascade_model.filter_dict_by_list(train_params, train_keys)
train_fit_fail_list = [post_id for post_id in train_fit_fail_list if post_id in train_keys]


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

avg_params = None

#process all posts (or just one, if doing that)
post_count = 0
vprint("Processing %d post" % len(test_posts) + ("s" if len(test_posts) > 1 else ""))
for sim_post_id, sim_post in test_posts.items():

	#skip this post if we've already done it
	if sim_post_id in finished_posts:
		continue

	if batch == False:
		vprint("Simulation post has %d comments" % test_cascades[sim_post_id]['comment_count_total'])

	#if simulating - get sim params from somewhere, either random or average
	#will use these same params for all simulations of this post, regardless of the observed setting

	#draw random simulation params from training set
	if mode == "rand_sim":
		rand_train_id = random.choice(train_keys)	#random training post
		#get params: either fitted or default
		if rand_train_id in train_params:
			sim_params = train_params[rand_train_id]
			param_source = "random_fitted"
			#if any are false here, fill the holes with default
			if sim_params[0] == False or sim_params[3] == 0:
				sim_params = functions_gen_cascade_model.get_complete_params(train_cascades[rand_train_id], sim_params)
				param_source = "random_fitted_default_combo"
		else:
			sim_params = functions_gen_cascade_model.get_default_params(train_cascades[rand_train_id])
			param_source = "random_default"
	#compute average params over entire training set (default, partial fitted, and fully fitted)
	if mode == "avg_sim" and avg_params is None:
		vprint("Computing average params for training set")
		avg_params = [0, 0, 0, 0, 0, 0]		#init to 0
		#loop all training posts
		for train_id in train_keys:
			#fitted params, fill any holes
			if train_id in train_params:
				train_post_params = train_params[train_id]
				#fill holes in fitted params
				if train_post_params[0] == False or train_post_params[3] == 0:
					train_post_params = functions_gen_cascade_model.get_complete_params(train_cascades[train_id], train_post_params)
			#no fitted params, get default and add to total	
			else:
				train_post_params = functions_gen_cascade_model.get_default_params(train_cascades[train_id])
			#add this post's params to runnign total	
			for i in range(6): avg_params[i] += train_post_params[i]
		#finish the average
		for i in range(6): avg_params[i] /= len(train_keys)
		print("Average params: ", avg_params)
		param_source = "avg_fitted"
		#set the average as the sim_params
		sim_params = avg_params.copy()

	#get time-shifted ground-truth cascade (same for all observation periods)
	true_cascade, true_comment_count = functions_gen_cascade_model.shift_comment_tree(test_cascades[sim_post_id])
	#and compute the structural virality of this cascade
	true_structural_virality = functions_gen_cascade_model.get_structural_virality(true_cascade)

	#duplicate the true cascade - will use as a working copy for different observed trees
	observed_tree = deepcopy(true_cascade)

	#use the same sim params for all the time_observed values
	for observed in sorted(observed_list, reverse=True):
		#if returning a random tree instead of simulating, pick one and skip to eval
		#this doesn't take the observed time/comments into account at all, so results will be pretty poor indeed
		if mode == "rand_tree":
			#draw a random id
			rand_train_id = random.choice(train_keys)	#random training post
			#get time-shifted cascade for this post, to use as "simulated" tree
			sim_tree, simulated_count = functions_gen_cascade_model.shift_comment_tree(train_cascades[rand_train_id])
			#ignore some eval fields, since we're only doing one random tree per test post
			observed_count = 0
			time_observed = 0
			param_source = "rand_tree"

		#simulating from random params or average params, generate a tree
		else:
			if not batch:
				vprint("Simulation params: ", sim_params)

			#get truncated cascade, so we know how many comments observed
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

			#SIMULATE COMMENT TREE
			sim_tree, simulated_count = functions_gen_cascade_model.simulate_comment_tree(sim_params, observed_tree, observed_count, time_observed*60.0, not batch)
			if not batch:
				vprint("Simulated cascade has ", simulated_count, " comments")

			#don't try to eval if sim failed (aborted infinite sim)
			if sim_tree == False:
				print("infinite sim aborted, skipping post", sim_post_id)
				continue

		#EVAL

		#already got ground-truth cascade above

		#get sim cascade as networkx graph
		#sim_graph = functions_gen_cascade_model.cascade_to_graph(sim_tree)

		#compute tree edit distance between ground-truth and simulated cascades
		eval_res = functions_baseline_model.eval_trees(sim_post_id, sim_tree, true_cascade, simulated_count, observed_count, true_comment_count, true_structural_virality, time_observed, observing_time, time_error_margin, error_method, (observed if observing_time==False else None))
		#add a column indicating where the params for this sim came from
		eval_res['param_source'] = param_source
		#dummy columns (graph infer) so output format matches real model exactly
		eval_res['disconnected'] = "N/A"
		eval_res['connecting_edges'] = "N/A"

		#if running in random tree mode, null out the observed stats now that eval is done
		if mode == "rand_tree":
			eval_res['observed_comment_count'] = "N/A"
			eval_res['time_observed'] = "N/A"

		#append eval data to overall list
		all_metrics.append(eval_res)

		#if running in random tree mode, skip all other observed settings
		if mode == "rand_tree":
			break

	#counter and periodic prints
	post_count += 1
	finished_posts.add(sim_post_id)
	if batch and post_count % 100 == 0:
		vprint("   finished %d posts" % post_count)

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
vprint("Finished simulating %d posts" % post_count)

if post_count == 0:
	vprint("\nNo posts simulated, no results to save\n")
	exit(0)

#save metrics + settings to output file
functions_gen_cascade_model.save_results(outfile, all_metrics, observing_time)

#all done, update bookmark to "finished"
functions_gen_cascade_model.save_bookmark(finished_posts, outfile, status=(True if len(finished_posts) == len(test_posts) else False))

vprint("All done, all results saved\n")
