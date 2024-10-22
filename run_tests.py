import subprocess
from collections import defaultdict
import sys

import file_utils
from functions_gen_cascade_model import load_bookmark


#given a list of bookmark files, check to see if they have all completed
def check_completion(file_list):
	#loop all bookmarks
	for bookmark in file_list:
		finished_posts, complete = load_bookmark(bookmark)
		#if this run didn't finish, return false
		if complete == False:
			return False
	return True
#end check_completion


#dictionary of arguments with values, list of flag/boolean arguments
arguments = {}
arguments_list = []

#boolean flags - what do you want to run?
run_model = True
run_baseline = True
run_comparative = True

#REQUIRED ARGUMENTS

repeat_runs = 5			#number of repeated runs to do for each subreddit/size class

#list of subreddits separately, since one output file per subreddit
#subreddits = ['explainlikeimfive']	#list of subreddits, -s
#converted this to a command line arg for parallel processing of multiple subreddits,
#even during the preprocessing phase
#but it could be manual without too much work
subreddits = sys.argv[1:]

#list of baseline model modes to run
#each of them will produce a separate output file
#options are: -rand_sim, -rand_tree, -avg_sim
baseline_modes = ['-rand_sim', '-rand_tree', '-avg_sim']

#list of times/comment counts for observation, along with the selected option
observation_option = '-t'		#-nco or -t
observation_list = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24]
#if observing by comments, this will be filtered depending on the size class, so you don't try to observe more than can possibly exist for that run
#if observing by time, list will be used unedited, even if observe the entire cascade more than once
#select an edge weight method
arguments_list.append('-j')		#-j, -c, or -wmd

#define the test set: -n <sample size>, -id <post id>, -r, or -a
#arguments_list.append('-a')		#list for -a or -r
arguments['-n'] = 1000			#dict for -n or -id

#define the size of the training set
arguments['-n_train'] = 10000

#choose at least one graph limit option, but can use both
arguments['-topn'] = 20			#max connected neighbors (pick highest weighted edges)
#arguments['-threshold'] = 0.1		#min edge weight to be included in graph

#define testing set - used unless running crypto or cyber
arguments['-y'] = 2017		#year
arguments['-m'] = 12

#pick an error method - --topo_err, --err_abs, or define a margin using --err
arguments_list.append('--topo_err')

#size classes - put a list of size category breaks here, and the loop will process them all
#first will just be used as a max, last as just a min
#for options -min and -max
#size_breaks = [1, 5, 10, 30, 50, 100, 200, 300, 500, 1000]
size_breaks = []

#OPTIONAL ARGUMENTS: only add to dictionary/list if not using default

#arguments['-g'] = [500]		#max nodes in graph
#arguments['-q'] = [0.1]		#minimum node quality
#arguments['-l'] = 1			#number of months for testing (default 1)
#arguments['-p'] = 1			#number of months for training
#arguments['-np'] = 'ln'		#optional normalization, ln (natural log) or mm (min-max)
#arguments['-threshold'] = 0.1	#minimum edge weight for inclusion in graph

#must do both of these together, if you want any
#arguments['-down_ratio'] = 1	#ratio of large:small posts for graph downsampling, 
								#where large/small border defined by -large_req
#arguments['-large_req'] = 10	#divider between large and small cascades for downsample

#arguments_list.append('-e')		#estimate initial params
arguments_list.append('-v')			#verbose output
arguments_list.append('-d')		#include default posts in graph
#arguments_list.append('--sanity')	#simulate from fitted params
#arguments_list.append('--train_stats')		#compute and output training set stats
#arguments_list.append('--test_stats')		#compute and output testing set stats
#arguments_list.append('-b')		#force all training node qualities to 1, so learning rate is 0
#arguments_list.append('-timestamps')

#can only use this option if doing jaccard
#arguments_list.append('-stopwords')			#remove stopwords from titles for jaccard edge weight calc

#END ARGUMENTS

#make sure output dir for each subreddit exists
for subreddit in subreddits:
	file_utils.verify_dir("sim_results/%s" % subreddit)
	file_utils.verify_dir("sim_results/%s/run_results" % subreddit)

#prepend/append a None to size breaks list for easier looping
size_breaks = [None] + size_breaks + [None]

#count runs for each subreddit (hackery incoming)
sub_counts = defaultdict(int)

#keep list of background processes, so we can wait for them at the end
background_procs = []

#outfile list
#for each model, keep a list of base outfiles, so we can check the bookmarks at the end
outfile_lists = {"model": [], "comparative": [], '-rand_sim': [], '-rand_tree': [], '-avg_sim': []}

#loop repeated runs
for run in range(repeat_runs):

	#loop size classes
	for min_size, max_size in zip(size_breaks, size_breaks[1:]):

		#string representation of this size class
		if min_size is not None and max_size is not None:
			size_class = "_%d-%d" % (min_size, max_size)
		elif min_size is not None:
			size_class = "_>=%d" % min_size
		elif max_size is not None:
			size_class = "_<%d" % max_size
		else:
			size_class = ""

		#and set the min and max in the arguments dict accordingly
		#if using size classes, remove old settings
		if min_size is None and max_size is None:
			if '-min' in arguments: del arguments['-min']
			if '-max' in arguments: del arguments['-max']
			#save new
			if min_size is not None: arguments['-min'] = min_size
			if max_size is not None: arguments['-max'] = max_size
		#not using size classes, leave hard-set min/max arguments alone - do nothing

		#filter the observed list for this run
		if max_size is not None:
			run_observed_list = [str(count) for count in observation_list if count <= max_size]
		else:
			run_observed_list = [str(count) for count in observation_list]

		#loop subreddits
		for subreddit in subreddits:

			#is this the first run for this subreddit? 
			#if yes, make sure all the preprocessing is done and the graph exists first

			#define our base output filename - keep it simple, will have all the settings in the output files
			outfile = "sim_results/%s/run_results/%s_model_%dtrain_%dtest_%d-%d%s%s%s" % (subreddit, subreddit, arguments['-n_train'], arguments['-n'], arguments['-y'], arguments['-m'], size_class, "_fixed_qual" if '-b' in arguments_list else "", "_run%d" % run if repeat_runs > 1 else "")
			outfile_lists['model'].append(outfile)

			#build command arguments list
			#base first
			model_command = ['time', 'python3', 'gen_cascade_model.py', '-s', subreddit, '-o', outfile]
			#add all the dict args
			for key, value in arguments.items():
				model_command.append(key)
				model_command.append(str(value))
			#all the list args
			model_command = model_command + arguments_list
			#observation list
			model_command.append(observation_option)
			model_command = model_command + run_observed_list

			#wait for this graph-build-only run to finish before doing more
			if sub_counts[subreddit] == 0:
				print("Preprocessing", subreddit)
				f = open("sim_results/%s/run_results/%s_%dtrain_%dtest_%d-%dgraph.txt" % (subreddit, subreddit, arguments['-n_train'], arguments['-n'], arguments['-y'], arguments['-m']), "a")
				subprocess.call(model_command+['-preprocess'], stdout=f, stderr=f)

			#run corresponding baseline models in background - if don't have results already
			if run_baseline:
				for mode in baseline_modes:
					#define output filename for baseline model
					baseline_outfile = "sim_results/%s/run_results/%s_baseline_%s_%dtrain_%dtest_%d-%d%s%s" % (subreddit, subreddit, mode[1:], arguments['-n_train'], arguments['-n'], arguments['-y'], arguments['-m'], size_class, "_run%d" % run if repeat_runs > 1 else "")
					outfile_lists[mode].append(baseline_outfile)

					#no data for this baseline configuration, run the test
					#check the bookmark saved by the model to know if finished or not
					finished_posts, complete = load_bookmark(baseline_outfile)
					if complete:
						print("skipping", baseline_outfile)

					else:
						#build command arguments list
						#base first
						baseline_command = ['time', 'python3', 'baseline_model.py', '-s', subreddit, '-o', baseline_outfile, mode, '-v']
						#add the dict args - but only the ones that make sense for the baseline model
						for arg in ['-n', '-n_train', '-m', '-y', '-min', '-max', '-timestamps']:
							if arg in arguments:
								baseline_command.append(arg)
								baseline_command.append(str(arguments[arg]))
							if arg in arguments_list:
								baseline_command.append(arg)
						#observation list
						baseline_command.append(observation_option)
						baseline_command = baseline_command + run_observed_list

						print(baseline_outfile)
						
						#run the thing, piping output to file
						f = open(baseline_outfile+".txt", "w")
						f.write(' '.join(baseline_command)+'\n')		#write arguments to first line of file
						f.flush()  #make sure arguments get written first
						process = subprocess.Popen(baseline_command, stdout=f, stderr=f)
						#no wait, run in background
						background_procs.append(process)
			#end if run_baseline

			#also run comparative model (reddit paper Hawkes) in background - if not already done
			if run_comparative:
				#define output filename for comparative model
				comparative_outfile = "sim_results/%s/run_results/%s_comparative_%dtest_%d-%d%s%s" % (subreddit, subreddit, arguments['-n'], arguments['-y'], arguments['-m'], size_class, "_run%d" % run if repeat_runs > 1 else "")
				outfile_lists['comparative'].append(comparative_outfile)

				#no data for this configuration, run the test
				#check bookmark saved by the model to know if finished or not
				finished_posts, complete = load_bookmark(comparative_outfile)
				if complete:
					print("skipping", comparative_outfile)

				else:
					#build command arguments list
					#base first
					comparative_command = ['time', 'python3', 'comparative_model.py', '-s', subreddit, '-o', comparative_outfile, '-v']
					#add the dict args - but only the ones that make sense for the comparative model
					for arg in ['-n', '-m', '-y', '-min', '-max', '-timestamps']:
						if arg in arguments:
							comparative_command.append(arg)
							comparative_command.append(str(arguments[arg]))						
						if arg in arguments_list:
							comparative_command.append(arg)
					#observation list
					comparative_command.append(observation_option)
					comparative_command = comparative_command + run_observed_list

					print(comparative_outfile)
					
					#run the thing, piping output to file
					f = open(comparative_outfile+".txt", "w")
					f.write(' '.join(comparative_command)+'\n')		#write arguments to first line of file
					f.flush()  #make sure arguments get written first
					process = subprocess.Popen(comparative_command, stdout=f, stderr=f)
					#no wait, run in background
					background_procs.append(process)
			#end if run_comparative

			#and then run the regular model, and wait for it to finish
			if run_model:
				#outfile name defined above

				#this run already done? skip
				#check the bookmark saved by the model to know if finished or not
				finished_posts, complete = load_bookmark(outfile)
				if complete:
					print("skipping", outfile)
					sub_counts[subreddit] += 1	#sub done, add to counter
					continue

				#model command defined above

				print(outfile)
				
				#run the thing, piping output to file
				f = open(outfile+".txt", "w")
				f.write(' '.join(model_command)+'\n')		#write arguments to first line of file
				f.flush()  #make sure arguments get written first
				process = subprocess.Popen(model_command, stdout=f, stderr=f)
				#process.wait()		#wait for it to finish before we do more, if you want
				background_procs.append(process)

				'''
				#did this actually finish? if not, try again (just once)
				#check the bookmark saved by the model to know if finished or not
				finished_posts, complete = load_bookmark(outfile)
				if complete == False:
					print("failed, restarting", outfile)
					#append to file this time, don't write arguments
					f = open(outfile+".txt", "a")
					process = subprocess.Popen(model_command, stdout=f, stderr=f)
					#process.wait()		#wait for it to finish before we do more, if you want
					background_procs.append(process)
				'''
			#end if run_model

			#add to subreddit counter
			sub_counts[subreddit] += 1

print("\nTest counts by subreddit:")
total_count = 0
for sub, count in sub_counts.items():
	print("  ", sub, ":", count)
	total_count += count
print("Total:", total_count, "(plus", len(subreddits), "preprocessing)\n")

print("\nWaiting on", len(background_procs), "background processes")
exit_codes = [p.wait() for p in background_procs]
print("Finished\n")

#combine multiple runs into a single results file - if all those runs *actually* finished
print("Creating combined results files")
if repeat_runs != 1 or len(size_breaks) != 0:
	#loop subreddits
	for subreddit in subreddits:
		#path to subreddit results directory (top-level)
		subreddit_dir = "sim_results/%s/" % subreddit
		#path to subreddit dir for individual results
		run_dir = subreddit_dir + "run_results/"

		#baseline results
		for mode in baseline_modes:
			#if all finished, combine
			if run_baseline and check_completion(outfile_lists[mode]):
				#redefine output filename - without run identifier or subreddit directory
				baseline_outfile = "%s_baseline_%s_%dtrain_%dtest_%d-%d%s" % (subreddit, mode[1:], arguments['-n_train'], arguments['-n'], arguments['-y'], arguments['-m'], size_class)	
				#combine matching files from multiple runs together
				file_utils.combine_csv(subreddit_dir+baseline_outfile+"_all_results.csv", run_dir+baseline_outfile + ("*" if len(size_breaks) != 0 else "") + "*.csv", display=True)
			elif run_baseline == False: print("Skipped", mode[1:], "baseline runs")
			else: print("Not all runs finished, skipping", mode[1:], "baseline combine")

		#comparative model results - if all runs finished, combine
		if run_comparative and check_completion(outfile_lists['comparative']):
			#redefine output filename - without run identifier or subreddit directory
			comparative_outfile = "%s_comparative_%dtest_%d-%d%s" % (subreddit, arguments['-n'], arguments['-y'], arguments['-m'], size_class)
			#combine matching files from multiple runs together
			file_utils.combine_csv(subreddit_dir+comparative_outfile+"_all_results.csv", run_dir+comparative_outfile + ("*" if len(size_breaks) != 0 else "") + "*.csv", display=True)
		elif run_comparative == False: print("Skipped comparative runs")	
		else: print("Not all runs finished, skipping comparative combine")

		#test results - if all runs finished, combine
		if run_model and check_completion(outfile_lists['model']):
			#redefine output filename - without run identifier or subreddit directory
			outfile = "%s_model_%dtrain_%dtest_%d-%d%s%s" % (subreddit, arguments['-n_train'], arguments['-n'], arguments['-y'], arguments['-m'], size_class, "_fixed_qual" if '-b' in arguments_list else "")
			#combine matching files from multiple runs together
			file_utils.combine_csv(subreddit_dir+outfile+"_all_results.csv", run_dir+outfile + ("*" if len(size_breaks) != 0 else "") + ".csv", display=True)
		elif run_model == False: print("Skipped model runs")
		else: print("Not all runs finished, skipping model combine")

