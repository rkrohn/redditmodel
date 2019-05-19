import glob
import sys
import pandas as pd
from collections import defaultdict

import file_utils

#compute percent error for the given sim and true columns of the dataframe
#adds as a new column to the dataframe
def percent_error(df, true, sim, err_name):
	df[err_name] = abs(df[true]+1 - df[sim]+1) / (df[true]+1)
#end percent_error

#compute average of a column based on grouping of another
#returns a new dataframe with groupby as the index
def avg_column(df, col, groupby):
	avg_data = df.groupby([groupby])[col].mean().reset_index()
	avg_data.set_index(groupby, inplace=True)
	avg_data.rename(index=str, columns={col: "avg_"+col}, inplace=True)		#prepend avg to col name
	return avg_data
#end avg_column

#compute std dev of a column based on grouping of another
#returns a new dataframe with groupby as the index
#for now, this is population std dev - because some groups only have 1 post
def std_dev_column(df, col, groupby):
	avg_data = df.groupby([groupby])[col].std(ddof=0).reset_index()
	avg_data.set_index(groupby, inplace=True)
	avg_data.rename(index=str, columns={col: "std_"+col}, inplace=True)		#prepend std to col name
	return avg_data
#end std_dev_column

#counts number of records based on grouping of another
#returns a new dataframe with groupby as the index
def count_column(df, col, groupby):
	avg_data = df.groupby([groupby])[col].size().reset_index()
	avg_data.set_index(groupby, inplace=True)
	avg_data.rename(index=str, columns={col: "count"}, inplace=True)		#simple name
	return avg_data
#end count_column

subreddits = sys.argv[1:]

for subreddit in subreddits:
	print("\nProcessing", subreddit)

	#find different run groups based on the timestamp and test period
	all_file_list = glob.glob('sim_results/%s/%s_*_*_test_*_results.csv' % (subreddit, subreddit))	#all files

	#parse into the different test runs - will each generate a separate eval file
	runs = defaultdict(list)	#(test date, timestamp) -> list of files
	for file in all_file_list:
		#pull test period and run timestamp
		chunks = file.split('_')
		test_date = chunks[2]
		timestamp = chunks[5]
		#add this file to corresponding list
		runs[(test_date, timestamp)].append(file)

	print("Found", len(runs), "eval group"+("s" if len(runs) > 1 else ""))

	#process each eval group separately
	for run, file_list in runs.items():
		print("Processing run", run[0], run[1])

		file_data = []
		for file in file_list:
			#build list of dataframes
			file_data.append(file_utils.load_csv_pandas(file, index_col='post_id'))
			print("  ", file_data[-1].shape[0], "records from", file)

		#concat into one
		all_data = pd.concat(file_data)
		print(all_data.shape[0], "total records")

		#drop the columns we don't care about (can put them back later)
		all_data.drop(['param_source', 'observing_by', 'dist', 'norm_dist', 'norm_dist_exclude_observed', 'MEPDL_min', 'MEPDL_max', 'remove_count', 'remove_time', 'insert_count', 'insert_time', 'update_count', 'update_time', 'match_count', 'disconnected', 'connecting_edges'], axis=1, inplace=True)

		#new eval columns - percent difference on size, depth, breadth
		percent_error(all_data, 'true_comment_count', 'simulated_comment_count', 'size_err')
		percent_error(all_data, 'true_depth', 'simulated_depth', 'depth_err')
		percent_error(all_data, 'true_breadth', 'simulated_breadth', 'breadth_err')

		#average evals by observed
		averages = []
		averages.append(avg_column(all_data, 'size_err', 'max_observed_comments'))
		averages.append(avg_column(all_data, 'depth_err', 'max_observed_comments'))
		averages.append(avg_column(all_data, 'breadth_err', 'max_observed_comments'))
		#and the std devs
		averages.append(std_dev_column(all_data, 'size_err', 'max_observed_comments'))
		averages.append(std_dev_column(all_data, 'depth_err', 'max_observed_comments'))
		averages.append(std_dev_column(all_data, 'breadth_err', 'max_observed_comments'))
		#and the counts
		averages.append(count_column(all_data, 'size_err', 'max_observed_comments'))

		avg_data = pd.concat(averages, axis=1)

		#dump to file
		save_filename = 'sim_results/%s/%s_%s_%s_agg_results.csv' % (subreddit, subreddit, run[0], run[1])
		file_utils.save_csv_pandas(avg_data, save_filename)
		print(avg_data.shape[0], "different observed averages saved to", save_filename)

	#end run group

#end subreddit