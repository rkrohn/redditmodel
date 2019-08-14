#given true lifetimes of test posts, and simulated lifetimes of same post set,
#perform a KS-test

import file_utils

from scipy import stats
from collections import defaultdict


subreddits = ["aww", "changemyview", "explainlikeimfive", "IAmA", "nottheonion", "science", "Showerthoughts", "sports", "worldnews"]

true_stats_filename = "sim_results/post_set_stats_%s_test2017-12_1000_test_posts_lifetime_v_size.csv"
model_sim_res_filename = "sim_results/%s/%s_model_10000train_1000test_2017-12_all_results.csv"

lifetime_lists_filename = "sim_results/all_1000test_lifetimelists.csv"
ks_test_res_filename = "sim_results/all_1000test_ks-test_results.csv"

ks_test_res = []

output_lists = []
output_fields = []

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
