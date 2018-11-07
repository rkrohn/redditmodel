import load_model_data
import cascade_analysis
import cascade_manip
import fit_cascade
import file_utils
import sim_tree
from ParamGraph import ParamGraph

import random

#driver for all the other things


code = "crypto"			#set use case/domain: must be crypto, cyber, or cve
						#crypto for dry run
						#cyber takes forever
						#cve fastest

subreddit = 'Lisk'		#process a particular subreddit if desired

print("\nProcessing", code)

#load data and build cascades
#posts, comments = load_reddit_data(code)
#cascades, comments, missing_posts, missing_comments = cascades.build_cascades(posts, comments, code)

#load all exogenous data for this use case/domain
#load_exogenous_data(code)

#build/load cascades (auto-load as a result, either raw data or cached cascades)
'''
cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(code)
#optional: filter out cascades with any missing elements (posts or comments)
cascades, comments = cascade_manip.remove_missing(code, cascades, comments)
'''

#get subreddit distribution
#cascade_analysis.get_subreddits(code, cascades)

#get/plot top-level comment response time distribution
#cascade_analysis.top_level_comment_response_dist(code, cascades, comments)		#1 minute bins
#cascade_analysis.top_level_comment_response_dist(code, cascades, comments, bin_minutes = 30)	#30 minute bins

#look at number of top-level comments from two sources
#cascade_analysis.check_comment_count(code, cascades)	


#filter cascades by a particular subreddit
'''
subreddit = "Lisk"
filtered_cascades = cascade_manip.filter_cascades_by_subreddit(cascades, subreddit)
#and filter comments to match those posts
filtered_comments = cascade_manip.filter_comments_by_posts(filtered_cascades, comments)
#save these filtered potss/comments for easier loading later
cascade_manip.save_cascades(code, filtered_cascades, subreddit)
cascade_manip.save_comments(code, filtered_comments, subreddit)
'''

#or, load cached filtered posts/comments
#cascades, comments = cascade_manip.load_filtered_cascades(code, subreddit)


#load cascades and fit params to all of them, loading checkpoints if they exist
'''
cascades, comments = cascade_manip.load_filtered_cascades(code, subreddit)	#load posts + comments
cascade_analysis.fit_all_cascades(code, cascades, comments, subreddit)		#load saved fits, and fit the rest
'''


#or, load specific saved cascade params from file

cascades, comments = cascade_manip.load_filtered_cascades(code, subreddit)	#load posts + comments
cascade_params = cascade_manip.load_cascade_params(code, subreddit + "50")
#filter cascades/comments to fitted posts (for testing)
cascades = {post_id : post for post_id, post in cascades.items() if post_id in cascade_params}
print("Filtered to", len(cascades), "posts with fitted parameters")
cascade_manip.filter_comments_by_posts(cascades, comments)


#build a ParamGraph for these posts
pgraph = ParamGraph()
pgraph.build_graph(cascades, cascade_params)


#simulate cascade based on fitted params of a single (possibly random) post
'''
random_post_id = "BitkI6YOhOueIKiphn5okA" #random.choice(list(cascade_params.keys()))
random_post = cascades[random_post_id]
print(random_post_id)
print("Random post has", len(random_post['replies']), "replies and", random_post['comment_count_total'], "total comments\n")

#pull params and simulate
post_params = cascade_params[random_post_id]
root, all_replies = sim_tree.simulate_comment_tree(post_params)

actual_post_comment_times = sorted(fit_cascade.get_root_comment_times(random_post, comments) + fit_cascade.get_other_comment_times(random_post, comments))
sim_tree.plot_all(all_replies, actual_post_comment_times, "gen_tree_replies.png")
sim_tree.plot_root_comments([child['time'] for child in root['children']], fit_cascade.get_root_comment_times(random_post, comments), "gen_tree_root_replies.png", params = post_params[:3])
#maybe want to break this plot down, at least into root/deeper, if not by level

#visualize the simulated tree
sim_tree.viz_tree(root, "gen_tree.png")
'''
