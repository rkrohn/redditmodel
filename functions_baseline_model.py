#functions for baseline_model.py
#very similar to versions in functions_gen_cascade_model.py, with a few minor modifications

import file_utils
import tree_edit_distance
import functions_gen_cascade_model

from argparse import *


#parse out all command line arguments and return results
def parse_command_args(baseline = True):
	#arg parser
	parser = ArgumentParser(description="BASELINE MODEL: Simulate reddit cascades from partially-observed posts.")

	#required arguments (still with -flags, because clearer that way, and don't want to impose an order)
	parser.add_argument("-s", "--sub", dest="subreddit", required=True, help="subreddit to process")
	parser.add_argument("-o", "--out", dest="outfile", required=True, help="base output filename")
	#must pick one of four processing options: a single id, random, all, or sample of size n
	proc_group = parser.add_mutually_exclusive_group(required=True)
	proc_group.add_argument("-id", dest="sim_post", default=None,  help="post id for single-processing")
	proc_group.add_argument("-r", "--rand", dest="sim_post", action="store_const", const="random", help="choose a random post from the subreddit to simulate")
	proc_group.add_argument("-a", "--all", dest="sim_post", action="store_const", const="all", help="simulate all posts in the subreddit")
	proc_group.add_argument("-n", "--n_sample", dest="sim_post", default=None, help="number of posts to test, taken as first n posts in the testing period")
	#must pick one baseline model mode, if running a baseline model
	if baseline:
		mode_group = parser.add_mutually_exclusive_group(required=True)
		mode_group.add_argument("-rand_sim", dest="mode", action="store_const", const="rand_sim", help="simulate cascade using fitted params for a random post in training set")
		mode_group.add_argument("-rand_tree", dest="mode", action="store_const", const="rand_tree", help="return a random cascade from training set (no simulation)")
		mode_group.add_argument("-avg_sim", dest="mode", action="store_const", const="avg_sim", help="simulate cascade using average params of training set")

	#must select an observation type (time or comments) and provide at least one time/count
	observed_group = parser.add_mutually_exclusive_group(required=True)
	observed_group.add_argument("-t", dest="time_observed", default=False, help="time of post observation, in hours", nargs='+')
	observed_group.add_argument("-nco", "--num_comments_observed", dest="num_comments_observed", default=False, help="number of comments observed", nargs='+')

	#must provide year and month for start of testing data set - unless running off crypto, cve, or cyber
	parser.add_argument("-y", "--year", dest="testing_start_year", default=False, help="year to use for test set")
	parser.add_argument("-m", "--month", dest="testing_start_month", default=False, help="month to use for test set")

	#optional args
	if baseline:
		parser.add_argument("-n_train", dest="training_num", default=10000, help="number of posts to use for training (immediately preceding test month")
	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose output")
	parser.set_defaults(verbose=False)
	parser.add_argument("--err", dest="time_error_margin", default=False, help="allowable time error for evaluation, in minutes")
	parser.add_argument("--err_abs", dest="time_error_absolute", action="store_true", help="use absolute time error, instead of increasing by level")
	parser.set_defaults(time_error_absolute=False)
	parser.add_argument("--topo_err", dest="topological_error", action="store_true", help="compute topological error only, ignoring comment timestamps")
	parser.set_defaults(topological_error=False)
	#can also layer in a size filter: only simulate cascades within a size range (or meeting some min/max size)
	#(filter applied before sample/rand)
	parser.add_argument("-min", "--min_size", dest="min_size", default=None, help="minimum cascade size for simulation test set")
	parser.add_argument("-max", "--max_size", dest="max_size", default=None, help="maximum cascade size for simulation test set (exclusive)")

	args = parser.parse_args()		#parse the args (magic!)

	#must provide year and month for start of testing data set - unless running off crypto, cve, or cyber
	if not (args.testing_start_month and args.testing_start_year) and args.subreddit != "cve" and args.subreddit != "crypto" and args.subreddit != "cyber":
		parser.error('Must specify year and month for start of testing period, -m and -y')

	#make sure error settings don't conflict
	if args.topological_error and (args.time_error_absolute or args.time_error_margin != False):
		parser.error('Cannot use topological error method with absolute time error or error margin setting')

	#extract arguments (since want to return individual variables)
	subreddit = args.subreddit
	sim_post = args.sim_post
	if baseline: mode = args.mode
	else: mode = "comparative"
	if args.time_observed != False:
		observed_list = [float(time) for time in args.time_observed]
		observing_time = True
	elif args.num_comments_observed != False:
		observed_list = [int(count) for count in args.num_comments_observed]
		observing_time = False
	outfile = args.outfile
	testing_start_month = int(args.testing_start_month)
	testing_start_year = int(args.testing_start_year)
	if baseline:
		training_num = int(args.training_num)
	verbose = args.verbose
	time_error_margin = float(args.time_error_margin) if args.time_error_margin != False else 30.0
	time_error_absolute = args.time_error_absolute
	topological_error = args.topological_error
	min_size = int(args.min_size) if args.min_size is not None else None
	max_size = int(args.max_size) if args.max_size is not None else None

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

	#if running socsim data, set that flag
	if subreddit == "cve" or subreddit == "cyber" or subreddit == "crypto":
		socsim_data = True
		#and set counts to something for when it shows up in filenames
		training_num = 0
		testing_num = 0
	else:
		socsim_data = False

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

	#print all arguments, so we know what this run was
	vprint("\n", args, "\n")	

	#print some log-ish stuff in case output being piped and saved
	vprint("Sim Post: ", sim_post, " %d" % sample_num if sample_num != False else "")
	if baseline: vprint("Baseline model mode: ", mode)
	else: vprint("Comparative model")
	if observing_time:
		vprint("Time Observed: ", observed_list)
	else:
		vprint("Comments Observed: ", observed_list)
	vprint("Output: ", outfile)
	vprint("Source subreddit: ", subreddit)
	vprint("Test Set: first %d posts starting at %d-%d" % (testing_num, testing_start_month, testing_start_year))
	if baseline: vprint("Training Set: %d posts immediately preceding %d-%d" % (training_num, testing_start_month, testing_start_year))
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
	vprint("")

	#return all arguments
	if baseline:
		return mode, subreddit, sim_post, observing_time, observed_list, outfile, batch, testing_num, testing_start_month, testing_start_year, training_num, time_error_margin, error_method, min_size, max_size, socsim_data, verbose
	else:
		return subreddit, sim_post, observing_time, observed_list, outfile, batch, testing_num, testing_start_month, testing_start_year, time_error_margin, error_method, min_size, max_size, socsim_data, verbose
#end parse_command_args


#given simulated and ground-truth cascades, compute the accuracy and precision of the simulation
#both trees given as dictionary-nested structure (returned from simulate_comment_tree and convert_comment_tree)
#return eval results in a metric-coded dictionary
#pretty much the same as the real model output, but without a couple columns that don't make sense
def eval_trees(post_id, sim_cascade, true_cascade, simulated_comment_count, observed_comment_count, true_comment_count, true_virality, time_observed, observing_time, time_error_margin, error_method,max_observed_comment_count=None):
	#get edit distance stats for sim vs truth
	#eval_res = tree_edit_distance.compare_trees(sim_cascade, true_cascade, error_method, time_error_margin)

	#no more tree edit distance, to expensive - and large trees die
	#so just get the tree stats
	eval_res = tree_edit_distance.tree_stats(sim_cascade, true_cascade)

	#compute structural virality of sim cascade
	sim_virality = functions_gen_cascade_model.get_structural_virality(sim_cascade)

	#add more data fields to the results dictionary
	eval_res['post_id'] = post_id
	eval_res['observed_comment_count'] = observed_comment_count
	eval_res['true_comment_count'] = true_comment_count
	eval_res['simulated_comment_count'] = simulated_comment_count
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
	#eval_res['norm_dist'] = eval_res['dist'] / eval_res['true_comment_count']	
	#divide by number of unobserved comments
	#eval_res['norm_dist_exclude_observed'] = eval_res['dist'] / (eval_res['true_comment_count'] - eval_res['observed_comment_count'])

	#structural virality
	eval_res['true_structural_virality'] = true_virality
	eval_res['sim_structural_virality'] = sim_virality

	#mean error per distance layer (both min and max)
	eval_res['MEPDL_min'], eval_res['MEPDL_max'] = functions_gen_cascade_model.mean_error_per_distance_layer(true_cascade, sim_cascade)

	#cascade lifetime - ie, time of last comment relative to root time
	true_comments = functions_gen_cascade_model.get_list_of_comment_times(true_cascade)
	eval_res['true_lifetime'] = 0 if len(true_comments) == 0 else true_comments[-1]
	sim_comments = functions_gen_cascade_model.get_list_of_comment_times(sim_cascade)
	eval_res['sim_lifetime'] = 0 if len(sim_comments) == 0 else sim_comments[-1]	

	return eval_res
#end eval_trees