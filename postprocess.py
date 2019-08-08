#finds matching .csv files, and combines them into a single .xlsx file, with one per sheet
#also add some columns and do some post-processing of data, so we don't have to do it manually

import pandas as pd
import glob
import re
import numpy as np

#find matching files to combine
#want all results files for the same model and testing size, but any subreddit or training set size

test_size = 1000		#desired test size to combine
year = 2017
month = 12

#format of combined results files, for glob
results_format = "sim_results/%s/%s_%s_%strain_%stest_%d-%d_all_results.csv"	#(subreddit, subreddit, model, trainsize, testsize, year, month)
#and for comparative files - no training value
comparative_results_format = "sim_results/%s/%s_%s_%stest_%d-%d_all_results.csv"

#loop all the model types:
for model in ["model", "comparative", "baseline_rand_tree", "baseline_rand_sim", "baseline_avg_sim"]:
#for model in ["baseline_avg_sim"]:
	#which format string? and fill in the glob *s
	if model == "comparative": curr_format = comparative_results_format % ("*", "*", model, test_size, year, month)
	else: curr_format = results_format % ("*", "*", model, "*", test_size, year, month)

	#get list of matching results files for this model with same testing set size and testing period
	results_files = glob.glob(curr_format)
	print("\n" + model + ": found", len(results_files), "matching files")

	#get correct index of subreddit and train size tokens, based on model name
	sub_index = 2
	train_index = 7 if "baseline" in model else (False if model == "comparative" else 5)

	#start up the excel writer for this model
	excel_file = 'sim_results/%s_all_%dtest_results.xlsx' % (model, test_size)
	writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
	print(excel_file, "contains:")

	#process each file
	for results_file in results_files:

		#what training size is this file for? and subreddit?
		file_tokens = re.split("_|/", results_file)
		if train_index != False: train_size = file_tokens[train_index][:-5]
		else: train_size = False
		subreddit = file_tokens[sub_index]
		
		if train_size != False: print("   ", subreddit+train_size)
		else: print("   ", subreddit)

		#load this file
		df = pd.read_csv(results_file)

		#postprocessing - do things before we write this out

		#add column for subreddit, and one for training size
		df.insert(0, 'subreddit', subreddit)
		if train_size != False: df.insert(1, 'train_size', int(train_size))

		#absolute and relative error for size, depth, breadth, and structural virality
		#columns: true_comment_count	simulated_comment_count	true_root_comments	sim_root_comments	true_depth	true_breadth	simulated_depth	simulated_breadth	true_structural_virality	sim_structural_virality	true_lifetime	sim_lifetime	MEPDL_min	MEPDL_max	disconnected	connecting_edges

		#size error
		df['abs_size_err'] = df['simulated_comment_count'] - df['true_comment_count']
		#want error of total size, not total number of comments
		df['rel_size_err'] = df['abs_size_err'] / (df['true_comment_count'] + 1)

		#depth error
		df['abs_depth_err'] = df['simulated_depth'] - df['true_depth']
		df['rel_depth_err'] = df['abs_depth_err'] / df['true_depth']

		#breadth error
		df['abs_breadth_err'] = df['simulated_breadth'] - df['true_depth']
		df['rel_breadth_err'] = df['abs_breadth_err'] / df['true_breadth']

		#structural virality error - only if both true and simulated trees have at least one comment
		#ie, total nodes n > 1
		df['abs_struct_vir_err'] = np.where(((df['true_comment_count'] > 0) & (df['simulated_comment_count'] > 0)), df['sim_structural_virality'] - df['true_structural_virality'], np.nan)
		df['rel_struct_vir_err'] = np.where((np.isnan(df['abs_struct_vir_err'])), np.nan, df['abs_struct_vir_err'] / df['true_structural_virality'])

		#lifetime error - absolute only
		df['abs_lifetime_err(min)'] = df['sim_lifetime'] - df['true_lifetime']

		#add as new sheet to running excel file - not outputting the index
		sheet_name = subreddit+train_size if train_size != False else subreddit
		df.to_excel(writer, sheet_name=sheet_name, index=False)

	#save spreadsheet for this model
	print("Saving file...")
	writer.save()
	exit(0)

exit(0)


'''
writer = pd.ExcelWriter('yourfile.xlsx', engine='xlsxwriter')
df = pd.read_csv('originalfile.csv')
df.to_excel(writer, sheet_name='sheetname')
writer.save()
'''