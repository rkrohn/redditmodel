#create and save all the necessary files for the new c++/python hybrid model
#save everything to model_files for easy server loading later

import cascade_analysis
import fit_cascade
import file_utils
from functions_prebuild_model import *

from itertools import count
import sys


#optional command line argument to limit to a single subreddit
if len(sys.argv) == 2:
	subreddit_filter = sys.argv[1]


#filepaths of output files
subreddits_filepath = "model_files/subreddits.pkl"		#dictionary of subreddit -> domain code
posts_filepath = "model_files/posts/%s_posts.pkl"			#processed post data for each post, one file per subreddit
														#each post maps original post id to numeric id, set of tokens, and user id
params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per subreddit
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)
graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this subreddit
users_filepath = "model_files/users/%s_users.txt"	#list of users seen in posts/comments, one file per subreddit


#load the subreddit distribution for all cascades (just need a list of subreddits)
if file_utils.verify_file(subreddits_filepath):
	print("Loading subreddit list from", subreddits_filepath)
	subreddit_dict = file_utils.load_pickle(subreddits_filepath)
#file doesn't exist, build it
else:	
	#load all three domain breakdown files
	crypto_subreddit_dist = file_utils.load_json("results/crypto_post_subreddit_dist.json")
	cve_subreddit_dist = file_utils.load_json("results/cve_post_subreddit_dist.json")
	cyber_subreddit_dist = file_utils.load_json("results/cyber_post_subreddit_dist.json")
	#combine into single dictionary of subreddit -> list of corresponding domain codes
	subreddit_dict = build_domain_dict([set(crypto_subreddit_dist.keys()), set(cve_subreddit_dist.keys()), set(cyber_subreddit_dist.keys())], ["crypto", "cve", "cyber"])
	#now, kill all the duplicates! crypto and cyber scraped entire subreddits, 
	#so any overlap is redudant and can be thrown away
	#(yes, there are neater ways to do this, but I don't care!)
	for item in subreddit_dict.keys():
		if len(subreddit_dict[item]) > 1:
			#crypto and cyber drowns out cve, so remove it
			if ("crypto" in subreddit_dict[item] or "cyber" in subreddit_dict[item]) and "cve" in subreddit_dict[item]:
				subreddit_dict[item].remove("cve")
		subreddit_dict[item] = subreddit_dict[item][0]

	#save as pickle for later
	print("Saving subreddit->domain mapping to", subreddits_filepath)
	file_utils.save_pickle(subreddit_dict, subreddits_filepath)

#verify directories for output files
file_utils.verify_dir("model_files/params")
file_utils.verify_dir("model_files/posts")
file_utils.verify_dir("model_files/graphs")
file_utils.verify_dir("model_files/users")

#loop all subreddits
for subreddit, domain in subreddit_dict.items():
	
	if subreddit != subreddit_filter:
		continue
	
	'''
	if domain != "crypto":
		continue
	'''

	print("\nProcessing", subreddit, "in", domain, "domain")

	#load filtered cascades for this subreddit
	#cascades, comments = load_subreddit_cascades(subreddit, domain)
	#delay loading until we actually need them
	cascades = None
	comments = None

	#load processed version of posts
	if file_utils.verify_file(posts_filepath % subreddit):
		posts = file_utils.load_pickle(posts_filepath % subreddit)
		print("Loaded", len(posts), "processed posts from", posts_filepath % subreddit)
	#doesn't exist, build processed posts file
	else:	
		#load cascades if not yet loaded
		cascades, comments = load_cascades(subreddit, domain, cascades, comments)
		#assign numeric ids to each post for node2vec input files
		#get set of tokens
		#extract and maintain user
		c = count()
		posts = {key: {'user': value['author_h'], 'tokens': extract_tokens(value), 'id': next(c)} for key, value in cascades.items()}
		#save this to file
		file_utils.save_pickle(posts, posts_filepath % subreddit)
		print("Saved", len(posts), "processed posts to", posts_filepath % subreddit)

	#build list of users active in this subreddit - list, not set, so more active users are more likely to get drawn in the simulation
	if file_utils.verify_file(users_filepath % subreddit):
		print("Active users exist in", users_filepath % subreddit)
	else:
		#load cascades if not yet loaded
		cascades, comments = load_cascades(subreddit, domain, cascades, comments)
		#build active users list
		active_users = []
		for post_id, post in cascades.items():
			active_users.append(post['author_h'])
		for comment_id, comment in comments.items():
			active_users.append(comment['author_h'])
		file_utils.save_pickle(active_users, users_filepath % subreddit)
		print("Saved", len(active_users), "active users to", users_filepath % subreddit)

	#fit params to all of the cascades, if no file
	#no need to load if we have them, won't use them again
	if file_utils.verify_file(params_filepath % subreddit):
		print("Params exist in", params_filepath % subreddit)
	else:
		#load cascades if not yet loaded
		cascades, comments = load_cascades(subreddit, domain, cascades, comments)

		#fit params to all cascades
		all_params = cascade_analysis.fit_all_cascades(domain, cascades, comments, False, subreddit)

		#save to text file now
		with open(params_filepath % subreddit, "w") as f: 
			for post_id, params in all_params.items():
				f.write(str(posts[post_id]['id']) + " ")		#write numeric post id
				for i in range(len(params)):
					f.write((' ' if i > 0 else '') + str(params[i]))
				f.write("\n")
		print("Saved text-readable params to", params_filepath % subreddit)

	#check for graph
	if file_utils.verify_file(graph_filepath % subreddit):
		print("Post-graph exists in", graph_filepath % subreddit)
	else:
		#now we have to build the graph... ugh
		build_graph(posts, graph_filepath % subreddit)

