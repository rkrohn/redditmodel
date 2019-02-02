#create and save all the necessary files for the new c++/python hybrid model for hackernews
#save everything to model_files for easy server loading later

#handle hackernews specially since it doesn't get broken up by subreddit

import cascade_manip
import cascade_analysis
import fit_cascade
import file_utils
from functions_prebuild_model import *

from itertools import count
import sys


#filepaths of output files
subreddits_filepath = "model_files/subreddits.pkl"		#dictionary of subreddit -> domain code
posts_filepath = "model_files/posts/%s_posts.pkl"			#processed post data for each post, one file per subreddit
														#each post maps original post id to numeric id, set of tokens, and user id
params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per subreddit
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)
graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this subreddit
users_filepath = "model_files/users/%s_users.txt"	#list of users seen in posts/comments, one file per subreddit

#verify directories for output files
file_utils.verify_dir("model_files/params")
file_utils.verify_dir("model_files/posts")
file_utils.verify_dir("model_files/graphs")
file_utils.verify_dir("model_files/users")

code = "hackernews"

#load raw hackernews cascades if no cached reconstructed cascades
if os.path.exists("data_cache/%s_cascades/%s_cascade_posts.pkl" % (code, code)) == False or os.path.exists("data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code)) == False:
	raw_stories = file_utils.load_json("../hackernews/data/hackernews_stories.json")
	raw_comments = file_utils.load_json("../hackernews/data/hackernews_comments.json")
else:
	raw_stories = {}
	raw_comments = {}

#build/load cascades (auto-load as a result, either raw data or cached cascades)
cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(code, raw_stories, raw_comments)
#optional: filter out cascades with any missing elements (posts or comments)
cascades, comments = cascade_manip.remove_missing(code, cascades, comments)

print("\nHave total of", len(cascades), "cascades and", len(comments), "comments for hackernews")

#build processed post file
#assign numeric ids to each post for node2vec input files
#get set of tokens
#extract and maintain user
if file_utils.verify_file(posts_filepath % code):
	print("Processed post file already exists.")
	posts = file_utils.load_pickle(posts_filepath % code)
else:
	c = count()
	posts = {key: {'user': value['author_h'], 'tokens': extract_tokens(value), 'id': next(c)} for key, value in cascades.items()}
	#save this to file
	file_utils.save_pickle(posts, posts_filepath % code)
	print("Saved", len(posts), "processed posts to", posts_filepath % code)

#build list of users active in this subreddit - list, not set, so more active users are more likely to get drawn in the simulation
if file_utils.verify_file(users_filepath % code):
	print("Active users exist in", users_filepath % code)
else:
	#build active users list
	active_users = []
	for post_id, post in cascades.items():
		active_users.append(post['author_h'])
	for comment_id, comment in comments.items():
		active_users.append(comment['author_h'])
	file_utils.save_pickle(active_users, users_filepath % code)
	print("Saved", len(active_users), "active users to", users_filepath % code)

#fit params to all of the cascades, if no file
#no need to load if we have them, won't use them again
if file_utils.verify_file(params_filepath % code):
	print("Params exist in", params_filepath % code)
else:
	#fit params to all cascades
	all_params, fit_fail_list = cascade_analysis.fit_all_cascades(code, cascades, comments, True)

	#save list of failed fits (if exists)
	if len(fit_fail_list) != 0 and file_utils.verify_file("model_files/params/%s_failed_param_fit.txt" % code) == False:
		file_utils.save_json(fit_fail_list, "model_files/params/%s_failed_param_fit.txt" % code)
		print("Saved list of fit-fail stories to model_files/params/%s_failed_param_fit.txt" % code)

	#save to text file now
	with open(params_filepath % code, "w") as f: 
		for post_id, params in all_params.items():
			f.write(str(posts[post_id]['id']) + " ")		#write numeric post id
			for i in range(len(params)):
				f.write((' ' if i > 0 else '') + str(params[i]))
			f.write("\n")
	print("Saved text-readable params to", params_filepath % code)

#don't build graph, would be way too big
'''
#check for graph
if file_utils.verify_file(graph_filepath % code):
	print("Post-graph exists in", graph_filepath % code)
else:
	#now we have to build the graph... ugh
	build_graph(posts, graph_filepath % code)
'''		

