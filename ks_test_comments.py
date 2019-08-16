#given true and simulated comment times, perform a KS-test on those times for each post

import file_utils

from scipy import stats
from collections import defaultdict


subreddit = "worldnews" 

model_filename = "sim_results/%s/run_results/%s_model_10000train_1000test_2017-12_10comm_run%d_timestamps.pkl"
comp_filename = "sim_results/%s/run_results/%s_comparative_1000test_2017-12_10comm_run%d_timestamps.pkl"
rand_tree_filename = "sim_results/%s/run_results/%s_baseline_rand_tree_10000train_1000test_2017-12_10comm_run%d_timestamps.pkl"
rand_sim_filename = "sim_results/%s/run_results/%s_baseline_rand_sim_10000train_1000test_2017-12_10comm_run%d_timestamps.pkl"
avg_filename = "sim_results/%s/run_results/%s_baseline_avg_sim_10000train_1000test_2017-12_10comm_run%d_timestamps.pkl"

ks_test_res_filename = "sim_results/all_1000test_ks-test_results_%s.csv"

ks_test_res = []

#process each model separately
for model, curr_filename in [("model", model_filename), ("comp", comp_filename), ("rand_tree", rand_tree_filename), ("rand_sim", rand_sim_filename), ("avg_sim", avg_filename)]:

	print("Processing", model, "and", subreddit)

	#process each subreddit
	for run in range(5):
		print("  run", run)

		#load timestamps data
		timestamps = file_utils.load_pickle(curr_filename % (subreddit, subreddit, run))

		#loop posts
		for post_id, post_timestamps in timestamps.items():

			#grab true timestamps
			true_timestamps = post_timestamps['true']

			#loop observation times
			for time in post_timestamps.keys():
				#skip true
				if time == "true": continue

				#pull sim comment times
				sim_timestamps = post_timestamps[time]

				#ks-test those distributions - if sim list is non-empty
				if len(sim_timestamps) != 0:
					D, p_val = stats.ks_2samp(true_timestamps, sim_timestamps)
				else:
					D = None
					p_val = None

				#add results to running total
				observed_count = len([timestamp for timestamp in true_timestamps if timestamp < time*60]) if model != "rand_tree" else None
				res = {'model': model, 'post': post_id, 'run': run, 'true_count': len(true_timestamps), "sim_count": len(sim_timestamps), 'observed_count': observed_count, 'time_observed': time,'D': D, 'p_val': p_val}
				ks_test_res.append(res)

#save ks-test results to file
file_utils.save_csv(ks_test_res, ks_test_res_filename % subreddit, ["model", 'post', 'run', 'true_count', 'sim_count', 'observed_count', 'time_observed', "D", "p_val"])


