import fit_cascade		#requires fit_weibull, fit_lognormal
import sim_tree

import pickle


#load one post and associated comments as an example
#example_post is a single dictionary object containing all post data fields
#example_comments is a dictionary of comment_id -> comment object, where each comment object is a dictionary
with open("example_post.pkl", "rb") as f:
	data = pickle.load(f)
example_post = data[0]
example_comments = data[1]
print("Example post", example_post['id_h'], "has", len(example_post['replies']), "replies and", example_post['comment_count_total'], "total comments\n")

#fit parameters for this cascade
example_post_params = fit_cascade.fit_cascade_model(example_post, example_comments, display=True)

#simulate a new comment tree based on fitted parameters
root, all_replies = sim_tree.simulate_comment_tree(example_post_params)

#some qualitative visualization
actual_post_comment_times = sorted(fit_cascade.get_root_comment_times(example_post, example_comments) + fit_cascade.get_other_comment_times(example_post, example_comments))	#get actual comment times of example_post
#plot distribution of actual vs simulated comment times (all comments)
sim_tree.plot_all(all_replies, actual_post_comment_times, "gen_tree_replies.png")
#plot distribution of actual vs simulated root comments
sim_tree.plot_root_comments([child['time'] for child in root['children']], fit_cascade.get_root_comment_times(example_post, example_comments), "gen_tree_root_replies.png", params = example_post_params[:3])

#visualize the simulated tree
sim_tree.viz_tree(root, "gen_tree.png")
