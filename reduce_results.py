#given results for some set of n posts, reduce files (csv and pkl) for use with a set of m posts, where m < n
#ie, reduce a large set of results to a small one

from argparse import *
import glob
import csv

import functions_gen_cascade_model
import file_utils

#parse out all command line arguments and return results
def parse_command_args():
	#arg parser
	parser = ArgumentParser(description="Simulate reddit cascades from partially-observed posts.")

	#required arguments (still with -flags, because clearer that way, and don't want to impose an order)
	parser.add_argument("-s", "--sub", dest="subreddit", required=True, help="subreddit to process")

	parser.add_argument("-n", "--n_sample", dest="test_size", default=None, help="number of posts to test, taken as first n posts in the testing period")

	#must provide year and month for start of testing data set - unless running off crypto, cve, or cyber
	parser.add_argument("-y", "--year", dest="testing_start_year", default=2017, help="year to use for test set")
	parser.add_argument("-m", "--month", dest="testing_start_month", default=12, help="month to use for test set")

	parser.add_argument("-n_train", dest="training_num", required=True, help="number of posts to use for training (immediately preceding test month")

	args = parser.parse_args()		#parse the args (magic!)

	#extract arguments (since want to return individual variables)
	subreddit = args.subreddit
	testing_num = int(args.test_size)
	testing_start_month = int(args.testing_start_month)
	testing_start_year = int(args.testing_start_year)
	training_num = int(args.training_num)

	#print some log-ish stuff in case output being piped and saved
	print("Target Size: ", testing_num)
	print("Source subreddit: ", subreddit)
	print("Test Set: first %d posts starting at %d-%d" % (testing_num, testing_start_month, testing_start_year))
	print("Training Set: %d posts immediately preceding %d-%d" % (training_num, testing_start_month, testing_start_year))

	#return all arguments
	return subreddit, testing_num, testing_start_month, testing_start_year, training_num
#end parse_command_args


#---MAIN EXECUTION BEGINS HERE---#

functions_gen_cascade_model.define_vprint(True)

#get the defining command args (none of the detailed ones, since running all the same configurations now)
#just need subreddit, test set size, training set size, and test date (for data load)
subreddit, testing_num, testing_start_month, testing_start_year, training_num = parse_command_args()

#load the test data
test_posts, test_cascades = functions_gen_cascade_model.load_processed_posts(subreddit, testing_start_month, testing_start_year, testing_num, load_cascades=True)

#get target test post set
#ensure post id is in dataset (and filter test_posts set down to processing group only)
test_posts = functions_gen_cascade_model.get_test_post_set("sample", None, None, testing_num, test_posts, test_cascades, subreddit, testing_start_month, testing_start_year)
#reduce cascades to match this set
if len(test_posts) != len(test_cascades):
	test_cascades = functions_gen_cascade_model.filter_dict_by_list(test_cascades, list(test_posts.keys()))

test_ids = list(test_posts.keys())	#list of test posts ids to include in target set

#now we have the target test set - are there any files in the results directory for a larger test set, that we can
#pull results from? for any of the models?

print("")

#for comparative and baseline files
#results filenames look like this: science_<model>_<#train>train_<#test>test_<year>-<month>_run<#>_results.csv
#and bookmark filenames look like this: science_<model>_<#train>train_<#test>test_<year>-<month>_run<#>_finished_posts.pkl
#where the model options are: model, comparative, baseline_rand_tree, baseline_rand_sim, and baseline_avg_sim

#format strings for results and bookmark filenames
results_format = "sim_results/%s/run_results/%s_%s_%dtrain_%stest_%d-%d_run%s_results.csv"
bookmark_format = "sim_results/%s/run_results/%s_%s_%dtrain_%stest_%d-%d_run%s_finished_posts.pkl"

#loop all the model types:
for model in ["model", "comparative", "baseline_rand_tree", "baseline_rand_sim", "baseline_avg_sim"]:
	print("Processing", model)

	#get list of matching results files for this model with same training set size and testing period
	results_files = glob.glob(results_format % (subreddit, subreddit, model, training_num, "*", testing_start_year, testing_start_month, "*"))
	print("   Found", len(results_files), "matching files")

	#get correct index of test size token, based on model name
	index = 7 if "baseline" in model else 5

	#process each file
	for results_file in results_files:

		#what testing size is this file for?
		file_tokens = results_file.split('_')
		file_test_size = int(file_tokens[index][:-4])
		file_run = int(file_tokens[index+2][-1])
		
		#if test size larger than target, process this file (and its associated pkl)
		if file_test_size > testing_num:
			#if bookmark and results already exist at target size for this run, skip reduction
			if file_utils.verify_file(bookmark_format % (subreddit, subreddit, model, training_num, testing_num, testing_start_year, testing_start_month, file_run)) and file_utils.verify_file(results_format % (subreddit, subreddit, model, training_num, testing_num, testing_start_year, testing_start_month, file_run)):
				print("   target exists, skipping", results_file)
				continue
				
			print("   reducing", results_file)

			#get corresponding bookmark filename
			bookmark_file = bookmark_format % (subreddit, subreddit, model, training_num, file_test_size, testing_start_year, testing_start_month, file_run)
			
			#load the pickle bookmark
			bookmark = file_utils.load_pickle(bookmark_file)

			#build a new one, including only the posts in our reduced target set
			finished_set = set([post_id for post_id in bookmark['finished_posts'] if post_id in test_ids])
			print("     ", len(finished_set), "posts finished in bookmark")

			#save new bookmark - if doesn't already exist (don't want to overwrite stuff!)
			file_utils.save_pickle({"finished_posts": finished_set, 'complete': True if len(finished_set) == testing_num else False}, bookmark_format % (subreddit, subreddit, model, training_num, testing_num, testing_start_year, testing_start_month, file_run))

			#edit the results csv to match this post set - if correctly sized results don't already exist
			count = 0
			first = True
			with open(results_file, 'r') as inp, open(results_format % (subreddit, subreddit, model, training_num, testing_num, testing_start_year, testing_start_month, file_run), 'w') as out:
				writer = csv.writer(out)
				for row in csv.reader(inp):
					if first:
						writer.writerow(row)
						first = False
					if row[0] in test_ids:
						writer.writerow(row)
						count += 1
			print("      Copied", count, "rows from results file")		

print("\nAll done")