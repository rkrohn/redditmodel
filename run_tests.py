import datetime
import subprocess
from collections import defaultdict
import sys

import file_utils
from functions_gen_cascade_model import load_bookmark

#dictionary of arguments with values, list of flag/boolean arguments
arguments = {}
arguments_list = []

#REQUIRED ARGUMENTS

repeat_runs = 1			#number of repeated runs to do for each subreddit/size class

#list of subreddits separately, since one output file per subreddit
#subreddits = ['explainlikeimfive']	#list of subreddits, -s
#converted this to a command line arg for parallel processing of multiple subreddits,
#even during the preprocessing phase
#but it could be manual without too much work
subreddits = sys.argv[1:]

#list of times/comment counts for observation, along with the selected option
observation_option = '-nco'		#-nco or -t
observation_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
#this will be filtered depending on the size class, so you don't try to observe more than can possibly exist for that run

#select an edge weight method
arguments_list.append('-j')		#-j, -c, or -wmd

#define the test set: -n <sample size>, -id <post id>, -r, or -a
arguments_list.append('-a')		#list for -a or -r
#arguments['-n'] = 50			#dict for -n or -id

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
size_breaks = [1, 5, 10, 30, 50, 100, 200, 300, 500, 1000]

#OPTIONAL ARGUMENTS: only add to dictionary/list if not using default

#arguments['-g'] = [500]		#max nodes in graph
#arguments['-q'] = [0.1]		#minimum node quality
#arguments['-l'] = 1			#number of months for testing (default 1)
#arguments['-p'] = 1			#number of months for training
arguments['-np'] = 'ln'		#optional normalization, ln (natural log) or mm (min-max)

#must do both of these together, if you want any
#arguments['-down_ratio'] = 3	#ratio of large:small posts for graph downsampling, 
								#where large/small border defined by -large_req
#arguments['-large_req'] = 20	#divider between large and small cascades for downsample

#arguments_list.append('-e')		#estimate initial params
arguments_list.append('-v')			#verbose output
arguments_list.append('-d')		#include default posts in graph
#arguments_list.append('--sanity')	#simulate from fitted params
#arguments_list.append('--train_stats')		#compute and output training set stats
#arguments_list.append('--test_stats')		#compute and output testing set stats

#can only use this option if doing jaccard
#arguments_list.append('-stopwords')			#remove stopwords from titles for jaccard edge weight calc

#END ARGUMENTS

#make sure output dir for each subreddit exists
for subreddit in subreddits:
	file_utils.verify_dir("sim_results/%s" % subreddit)

#timestamp for this set of runs
timestamp = '{:%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
#timestamp = "05-15-21:46:48"		#hardcode to fill holes caused by process death

#prepend/append a None to size breaks list for easier looping
size_breaks = [None] + size_breaks + [None]

#count runs for each subreddit (hackery incoming)
sub_counts = defaultdict(int)

#loop repeated runs
for run in range(repeat_runs):

	#loop size classes
	for min_size, max_size in zip(size_breaks, size_breaks[1:]):

		#string representation of this size class
		if min_size is not None and max_size is not None:
			size_class = "%d-%d" % (min_size, max_size)
		elif min_size is not None:
			size_class = ">=%d" % min_size
		else:
			size_class = "<%d" % max_size

		#and set the min and max in the arguments dict accordingly
		#remove old settings
		if '-min' in arguments: del arguments['-min']
		if '-max' in arguments: del arguments['-max']
		#save new
		if min_size is not None: arguments['-min'] = min_size
		if max_size is not None: arguments['-max'] = max_size

		#filter the observed list for this run
		if max_size is not None:
			run_observed_list = [str(count) for count in observation_list if count < max_size]
		else:
			run_observed_list = [str(count) for count in observation_list]

		#loop subreddits
		for subreddit in subreddits:

			#define our base output filename - keep it simple, will have all the settings in the output files
			outfile = "sim_results/%s/%s_%d-%d_%s_test_%s%s" % (subreddit, subreddit, arguments['-y'], arguments['-m'], size_class, timestamp, "_run%d" % run if repeat_runs > 1 else "")

			#this run already done? skip
			#check the bookmark saved by the model to know if finished or not
			finished_posts, complete = load_bookmark(outfile)
			if complete:
				print("skipping", outfile)
				continue

			#build command arguments list
			#base first
			command = ['time', 'python3', 'gen_cascade_model.py', '-s', subreddit, '-o', outfile]
			#add all the dict args
			for key, value in arguments.items():
				command.append(key)
				command.append(str(value))
			#all the list args
			command = command + arguments_list
			#observation list
			command.append(observation_option)
			command = command + run_observed_list

			#is this the first run for this subreddit? 
			#if yes, make sure all the preprocessing is done and the graph exists first
			#wait for this graph-build-only run to finish before doing more
			if sub_counts[subreddit] == 0:
				print("Preprocessing", subreddit)
				f = open("sim_results/%s/%s_%d-%d_%sgraph.txt" % (subreddit, subreddit, arguments['-y'], arguments['-m'], timestamp), "a")
				subprocess.call(command+['-preprocess'], stdout=f, stderr=f)
				print("Done")

			print(outfile)
			
			#run the thing, piping output to file
			f = open(outfile+".txt", "w")
			f.write(' '.join(command)+'\n')		#write arguments to first line of file
			f.flush()  #make sure arguments get written first
			process = subprocess.Popen(command, stdout=f, stderr=f)
			#process.wait()		#wait for it to finish before we do more, if you want

			#add to subreddit counter
			sub_counts[subreddit] += 1

print("\nTest counts by subreddit:")
total_count = 0
for sub, count in sub_counts.items():
	print("  ", sub, ":", count)
	total_count += count
print("Total:", total_count, "(plus", len(subreddits), "preprocessing)")