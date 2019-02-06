import file_utils
import cascade_manip

code = "hackernews"

#load list of posts that fit-failed
fit_fail = set(file_utils.load_json("model_files/params/hackernews_failed_param_fit.txt"))

#load hackernews cascades
posts = file_utils.load_pickle("data_cache/hackernews_cascades/hackernews_cascade_posts.pkl")
comments = file_utils.load_pickle("data_cache/hackernews_cascades/hackernews_cascade_comments.pkl")
print("Loaded", len(posts), "posts and", len(comments), "comments")

#remove missing
posts, comments = cascade_manip.remove_missing(code, posts, comments)

#remove posts for which the fit failed
posts = {key:value for (key,value) in posts.items() if key not in fit_fail}
posts, comments = cascade_manip.filter_comments_by_posts(posts, comments)
print("Down to", len(posts), "posts and", len(comments), "comments")

#filenames of filtered cascades and comments
cascades_filepath = "data_cache/filtered_cascades/%s_%s_cascades.pkl"	#domain and subreddit cascades
comments_filepath = "data_cache/filtered_cascades/%s_%s_comments.pkl"	#domain and subreddit comments

#save to same place as other filtered cascades - use hackernews as domain and subreddit
file_utils.save_pickle(posts, cascades_filepath % (code, code))
file_utils.save_pickle(comments, comments_filepath % (code, code))

#add hackernews to subreddit -> domain mapping
subs = file_utils.load_pickle("model_files/subreddits.pkl")
if code not in subs:
	subs[code] = code
	file_utils.save_pickle("model_files/domain_mapping.pkl")
