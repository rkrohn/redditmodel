
import file_utils
import cascade_analysis
import cascade_manip

import glob


code = "cyber"

#load cascades and comments from pickle
#cascades, comments, missing_posts, missing_comments = build_cascades(code, posts = False, comments = False)

print("Loading cascades from data_cache")
cascades = file_utils.load_pickle("data_cache/%s_cascades/%s_cascade_posts.pkl" % (code, code))

#comments: across multiple files
print("Loading comments from data_cache")		
comments = {}
files = sorted(glob.glob('data_cache/%s_cascades/%s_cascade_comments*' % (code, code)))
for file in files:
	print("Loading", file)
	new_comments = file_utils.load_pickle(file)
	comments.update(new_comments)

#missing posts and comments
missing_posts = file_utils.load_json("data_cache/%s_cascades/%s_cascade_missing_posts.json" % (code, code))
missing_comments = file_utils.load_json("data_cache/%s_cascades/%s_cascade_missing_comments.json" % (code, code))

#yay! loaded
print("   Loaded", len(cascades), "cascades with", len(comments), "comments")
print("     ", len(missing_posts), "missing posts", len(missing_comments), "missing comments")

cascades, comments = cascade_manip.remove_missing(code, cascades, comments)

#load subreddit distribution (just want list of subreddits)
cyber_subreddit_dist = file_utils.load_json("results/cyber_post_subreddit_dist.json")
print(sorted(list(cyber_subreddit_dist.keys())))

#filter posts/comments for each subreddit
for subreddit, count in cyber_subreddit_dist.items():
	print("Filtering for", subreddit)
	#filter cascades by a particular subreddit
	filtered_cascades = cascade_manip.filter_cascades_by_subreddit(cascades, subreddit)
	#and filter comments to match those posts
	filtered_cascades, filtered_comments = cascade_manip.filter_comments_by_posts(filtered_cascades, comments)
	#save these filtered posts/comments for easier loading later
	cascade_manip.save_cascades(code, filtered_cascades, subreddit)
	cascade_manip.save_comments(code, filtered_comments, subreddit)
	print("Files saved")
