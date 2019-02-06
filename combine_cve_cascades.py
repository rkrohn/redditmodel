import file_utils
import glob

code = "cve"

#load all cve cascades and comments
files = glob.glob('data_cache/filtered_cascades/cve_*_cascades.pkl')
posts = {}
for file in files:
	posts.update(file_utils.load_pickle(file))
files = glob.glob('data_cache/filtered_cascades/cve_*_comments.pkl')
comments = {}
for file in files:
	comments.update(file_utils.load_pickle(file))

#filenames of filtered cascades and comments
cascades_filepath = "data_cache/filtered_cascades/%s_%s_cascades.pkl"	#domain and subreddit cascades
comments_filepath = "data_cache/filtered_cascades/%s_%s_comments.pkl"	#domain and subreddit comments

#save to same place as other filtered cascades - use hackernews as domain and subreddit
file_utils.save_pickle(posts, cascades_filepath % (code, code))
file_utils.save_pickle(comments, comments_filepath % (code, code))

#add cve to subreddit -> domain mapping
subs = file_utils.load_pickle("model_files/domain_mapping.pkl")
if code not in subs:
	subs[code] = code
	file_utils.save_pickle(subs, "model_files/domain_mapping.pkl")
