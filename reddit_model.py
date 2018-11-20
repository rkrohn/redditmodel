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
#save these filtered posts/comments for easier loading later
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
cascade_params = cascade_manip.load_cascade_params(code, subreddit + "100")


#filter cascades/comments to fitted posts (for testing)
cascades = {post_id : post for post_id, post in cascades.items() if post_id in cascade_params}
print("Filtered to", len(cascades), "posts with fitted parameters")
cascade_manip.filter_comments_by_posts(cascades, comments)


#pull out one random cascade from those loaded for testing, remove from all cascades

test_post_id = "WYNap8ZYQ6kc0lKZRAX3tA" #"BitkI6YOhOueIKiphn5okA" #"kRl5UtFpGFEaAQ374AREfw" #random.choice(list(cascades.keys()))
test_post = cascades.pop(test_post_id)
test_post_params = cascade_params.pop(test_post_id)
print("Random post:", test_post_id, "\n   " + test_post['title_m'], "\n  ", test_post_params)



#build a ParamGraph for set of posts
pgraph = ParamGraph()
pgraph.build_graph(cascades, cascade_params)
#file_utils.save_pickle(pgraph, "class_pickle_test.pkl")	#save class instance for later

#or, load a saved class instance - skipping cascade loads and graph construction
#pgraph = file_utils.load_pickle("class_pickle_test.pkl")

#infer parameters for the random post
pgraph.add_post(test_post)	#add post to graph
pgraph.run_node2vec()		#node2vec
test_post_inferred_params = pgraph.infer_params(test_post, mode='weighted', skip_default=True)	#infer
print("\nFitted params:", test_post_params)
print("Inferred params:", test_post_inferred_params)
#simulate from inferred params
print("\nSimulating inferred: ", end='')
infer_root, infer_all_replies = sim_tree.simulate_comment_tree(test_post_inferred_params)

#compare to both the actual cascade, and a simulation based on the fitted parameters
print("Simulating fitted: ", end='')
fit_root, fit_all_replies = sim_tree.simulate_comment_tree(test_post_params)
actual_root_replies = fit_cascade.get_root_comment_times(test_post, comments)
actual_all_replies = sorted(actual_root_replies + fit_cascade.get_other_comment_times(test_post, comments))
print("Actual cascade has", len(actual_root_replies), "replies and", len(actual_all_replies), "total comments")
sim_tree.plot_three_comparison(fit_all_replies, infer_all_replies, actual_all_replies, "gen_tree_replies.png")


#simulate cascade based on fitted params of a single (possibly random) post
'''
random_post_id = "BitkI6YOhOueIKiphn5okA" #random.choice(list(cascade_params.keys()))
random_post = cascades[random_post_id]
print("Random post", random_post_id, "has", len(random_post['replies']), "replies and", random_post['comment_count_total'], "total comments\n")

#pull params and simulate from fitted params
post_params = cascade_params[random_post_id]
root, all_replies = sim_tree.simulate_comment_tree(post_params)

#pull actual cascade comment times for comparison, plot
actual_post_comment_times = sorted(fit_cascade.get_root_comment_times(random_post, comments) + fit_cascade.get_other_comment_times(random_post, comments))
sim_tree.plot_two_comparison(all_replies, actual_post_comment_times, "gen_tree_replies.png")
sim_tree.plot_root_comments([child['time'] for child in root['children']], fit_cascade.get_root_comment_times(random_post, comments), "gen_tree_root_replies.png", params = post_params[:3])
#maybe want to break this plot down, at least into root/deeper, if not by level

#visualize the simulated tree
sim_tree.viz_tree(root, "gen_tree.png")
'''
