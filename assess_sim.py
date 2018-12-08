#stripped-down version of hybrid_model, only simulates cascades for which we have fitted parameters and does not save output
#used to examine simulated cascade against actual cascade

import file_utils
import sim_tree
import fit_cascade
from functions_hybrid_model import *
import cascade_manip

from shutil import copyfile
import sys
import subprocess


#filepaths of input files
posts_filepath = "model_files/posts/%s_posts.pkl"		#processed post data for each post, one file per subreddit
														#each post maps original post id to numeric id, set of tokens, and user id
params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per subreddit
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)


DISPLAY = False

#verify command line args
if len(sys.argv) != 3:
	print("Incorrect command line arguments\nUsage: python3 hybrid_model.py <seed filename> <domain>")
	exit(0)

#extract arguments
infile = sys.argv[1]
domain = sys.argv[2]

#print some log-ish stuff in case output being piped and saved
print("Input", infile)

#load post seeds
raw_post_seeds = load_reddit_seeds(infile)

#convert to dictionary of subreddit->list of post objects
post_seeds = defaultdict(list)
for post in raw_post_seeds:
	post_seeds[post['subreddit']].append(post)
print({key : len(post_seeds[key]) for key in post_seeds})

post_counter = 1	#counter of posts to simulate, across all subreddits

#process each subreddit
for subreddit, seeds in post_seeds.items():
	
	#TESTING ONLY!!!!
	if subreddit != "Lisk":
		continue
	

	print("\nProcessing", subreddit, "with", len(seeds), "posts to simulate")

	#load preprocessed posts for this subreddit
	if file_utils.verify_file(posts_filepath % subreddit):
		posts = file_utils.load_pickle(posts_filepath % subreddit)
		print("Loaded", len(posts), "processed posts from", posts_filepath % subreddit)
	else:
		print("Cannot simulate for subreddit", subreddit, "without processed posts file", posts_filepath % subreddit)
		exit(0)

	#correct key format of seed posts, if necessary
	for seed_post in seeds:
		#if post id contains the t3_ prefix, strip it off so we don't have to change everything
		if seed_post['id_h'].startswith(POST_PREFIX):
			seed_post['id_h'] = seed_post['id_h'][3:]

	#load fitted params for this subreddit
	#fitted_params, quality = load_params(params_filepath % subreddit, posts, quality=True)

	#if have cached cascades/comments for this set of posts, load them
	if file_utils.verify_file("%s_test_cascades.pkl" % subreddit):
		cascades = file_utils.load_pickle("%s_test_cascades.pkl" % subreddit)
		comments = file_utils.load_pickle("%s_test_comments.pkl" % subreddit)
		print("Loaded", len(cascades), "filtered test cascades")
	#ptherwise, load filtered cascades for this subreddit, and build filtered list
	else:
		all_cascades, all_comments = cascade_manip.load_filtered_cascades(domain, subreddit)
		seed_ids = [post['id_h'] for post in seeds]
		cascades = {post_id: post for post_id, post in all_cascades.items() if post_id in seed_ids}
		file_utils.save_pickle(cascades, "%s_test_cascades.pkl" % subreddit)
		cascades, comments = cascade_manip.filter_comments_by_posts(cascades, all_comments)
		file_utils.save_pickle(comments, "%s_test_comments.pkl" % subreddit)

	#node2vec finished, on to the simulation!
	#for each post, infer parameters and simulate
	print("Simulating comment trees...")
	for seed_post in seeds:

		#grap real post for this cascade
		test_post = cascades[seed_post['id_h']]

		'''
		#if we can, use fitted params
		if test_post['id'] in fitted_params:
			post_params = fitted_params[test_post['id']]
			post_quality = quality[test_post['id']]
		#otherwise, skip this cascade
		else:
			print("No fitted params for this post - skipping")
			continue
		'''

		#fit params instead, because reasons
		res = fit_cascade.fit_cascade_model(test_post, comments)
		post_params = res[:6]
		post_quality = res[6]

		#print some stuff
		print("\nTitle:", test_post['title_m'])
		print("Params:", post_params)
		print("Quality:", post_quality)

		#some data on actual cascade
		actual_root_replies = fit_cascade.get_root_comment_times(test_post, comments)
		actual_all_replies = sorted(actual_root_replies + fit_cascade.get_other_comment_times(test_post, comments))
		print("Actual cascade has", len(actual_root_replies), "replies and", len(actual_all_replies), "total comments")
		actual_levels = fit_cascade.count_nodes_per_level(test_post, comments)
		for level, count in actual_levels.items():
			#print("   ", str(level) + ":", count)
			print(count, end=" ")
		print("")

		#simulate a comment tree!
		#actually, simulate a few
		for i in range(5):
			sim_root, all_times = sim_tree.simulate_comment_tree(post_params, display=True)
			sim_levels = sim_tree.sim_count_nodes_per_level(sim_root)
			for level, count in sim_levels.items():
				#print("   ", str(level) + ":", count)
				print(count, end=" ")
			print("")

#finished all posts across all subreddit
print("Finished all simulations")
