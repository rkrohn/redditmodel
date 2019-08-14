#given true lifetimes of test posts, and simulated lifetimes of same post set,
#perform a KS-test

import file_utils

from scipy import stats
from collections import defaultdict


subreddits = ["aww", "changemyview", "explainlikeimfive", "IAmA", "nottheonion", "science", "Showerthoughts", "sports", "worldnews"]

true_stats_filename = "sim_results/post_set_stats_%s_test2017-12_1000_test_posts_lifetime_v_size.csv"

model_sim_res_filename = "sim_results/%s/%s_model_10000train_1000test_2017-12_all_results.csv"
comp_sim_res_filename = "sim_results/%s/%s_comparative_1000test_2017-12_all_results.csv"
rand_tree_res_filename = "sim_results/%s/%s_baseline_rand_tree_10000train_1000test_2017-12_all_results.csv"
rand_sim_res_filename = "sim_results/%s/%s_baseline_rand_sim_10000train_1000test_2017-12_all_results.csv"
avg_sim_res_filename = "sim_results/%s/%s_baseline_avg_sim_10000train_1000test_2017-12_all_results.csv"

lifetime_lists_filename = "sim_results/all_1000test_lifetimelists.csv"
ks_test_res_filename = "sim_results/all_1000test_ks-test_results.csv"

ks_test_res = []

output_lists = []
output_fields = []

#process each model separately
for model, curr_filename in [("model", model_sim_res_filename), ("comp", comp_sim_res_filename), ("rand_tree", rand_tree_res_filename), ("rand_sim", rand_sim_res_filename), ("avg_sim", avg_sim_res_filename)]:

	print("Processing", model)

	#aggregate all subreddit data together for this model
	all_true_lifetime = []
	all_sim_lifetime = defaultdict(list)		#time observed -> list of lifetimes

	#process each subreddit
	for subreddit in subreddits:
		print("  loading", subreddit)

		#load true cascade lifetime data
		data = file_utils.load_csv_pandas(true_stats_filename % subreddit)
		#pull lifetime list
		true_lifetime = list(data['lifetime(minutes)'])

		#add true data to aggregate
		all_true_lifetime.extend(true_lifetime)

		#load sim results data
		data = file_utils.load_csv_pandas(curr_filename % (subreddit, subreddit))

		#rand tree doesn't have times, just get all the sim lifetimes
		if model == "rand_tree":
			all_sim_lifetime["all"] = list(data['sim_lifetime'])
		else:
			#loop observation times - if there are times
			for time in data['time_observed'].unique():

				#pull sim lifetime list (5x as many entries as true, but it doesn't matter)
				#but only pull for the current observation time
				sim_lifetime = list(data.loc[data['time_observed'] == time]['sim_lifetime'])

				#add sim data to aggregate
				all_sim_lifetime[time].extend(sim_lifetime)

	#add true data to output list
	output_lists.append(all_true_lifetime)
	output_fields.append("true_lifetime")

	#loop observation times (again), this time to perform ks-test
	for time in all_sim_lifetime.keys():
		print("   testing", time)

		#ks-test those distributions
		D, p_val = stats.ks_2samp(all_true_lifetime, all_sim_lifetime[time])

		#add results to running total
		res = {'model':model, 'time_observed':time,'D':D, 'p_val':p_val}
		ks_test_res.append(res)

		#add lifetime list for this time to running list
		output_lists.append(all_sim_lifetime[time])
		output_fields.append("%s_%dh_sim" % (model, time) if model != "rand_tree" else "%s_%sh_sim" % (model, "all"))

#save all lists to file
file_utils.multi_lists_to_csv(output_lists, output_fields, lifetime_lists_filename)

#save ks-test results to file
file_utils.save_csv(ks_test_res, ks_test_res_filename, ["model", 'time_observed', "D", "p_val"])



#this is all the old version, which is done for only the model, per subreddit
#new version (above) aggregates all the subreddits together, and does each model
'''
#process each subreddit
for subreddit in subreddits:
	#load true cascade lifetime data
	data = file_utils.load_csv_pandas(true_stats_filename % subreddit)
	#pull lifetime list
	true_lifetime = list(data['lifetime(minutes)'])

	#add true data to output lists
	output_lists.append(true_lifetime)
	output_fields.append("%s_true" % subreddit)

	#load sim results data
	data = file_utils.load_csv_pandas(model_sim_res_filename % (subreddit, subreddit))

	#loop observation times
	for time in data['time_observed'].unique():

		#pull sim lifetime list (5x as many entries as true, but it doesn't matter)
		#but only pull for the current observation time
		sim_lifetime = list(data.loc[data['time_observed'] == time]['sim_lifetime'])

		#ks-test those distributions
		D, p_val = stats.ks_2samp(true_lifetime, sim_lifetime)

		#add results to running total
		res = {'subreddit':subreddit, 'time_observed':time,'D':D, 'p_val':p_val}
		ks_test_res.append(res)

		#add lifetime list to running list
		output_lists.append(sim_lifetime)
		output_fields.append("%s_%dh_sim" % (subreddit, time))

#save all lists to file
file_utils.multi_lists_to_csv(output_lists, output_fields, lifetime_lists_filename)

#save ks-test results to file
file_utils.save_csv(ks_test_res, ks_test_res_filename, ["subreddit", 'time_observed', "D", "p_val"])
'''