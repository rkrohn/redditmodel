#new model for the cascade prediction/scenario 2 paper

#given a single cascade (top-level post and any observed comments), simulate the remainder of the cascade
#offload node2vec to c++, because speed

#requires the following command-line args: group (or hackernews or cve), id of cascade post to predict (or "random"), output filename for simulation results (no extension), max number of nodes for infer graph, minimum node quality for graph inference (set to -1 for no filter), esp (optional, for estimating initial params based on surrouding)

#for example:
#	python3 paper_model.py pivx random 4 sim_tree 2000 -1
#	python3 paper_model.py pivx 26RPcnyIuA0JyQpTqEui7A 1 sim_tree 500 -1			(4 comments)
#	paper_model.py pivx ZeuF7ZTDw3McZUOaosvXdA 5 sim_tree 250 -1					(11 comments)
#	paper_model.py compsci qOjspbLmJbLMVFxYbjB1mQ 200 sim_tree 250 -1				(58 comments)


import file_utils
import functions_paper_model
import cascade_manip
import fit_partial_cascade


print("")

#parse all command-line arguments
group, sim_post_id, time_observed, outfile, max_nodes, min_node_quality, estimate_initial_params = functions_paper_model.parse_command_args()

#ensure working directory exists
file_utils.verify_dir("sim_files")		

#load posts and comments for this group
raw_posts, raw_comments = functions_paper_model.load_group_data(group)

#ensure post id is in dataset
sim_post_id, random_post = functions_paper_model.verify_post_id(sim_post_id, list(raw_posts.keys()))

#pull out just the post (and associated comments) we care about
sim_post = raw_posts[sim_post_id]
#and filter to comments
junk, post_comments = cascade_manip.filter_comments_by_posts({sim_post_id: sim_post}, raw_comments, False)
print("Simulation post has", len(post_comments), "comments\n")


#GRAPH INFER
inferred_params = functions_paper_model.graph_infer(sim_post, sim_post_id, group, max_nodes, min_node_quality, estimate_initial_params)
#inferred_params = [1.73166, 0.651482, 1.08986, 0.762604, 2.49934, 0.19828]		#placeholder if skipping the infer
print("Inferred params:", inferred_params, "\n")


#REFINE PARAMS - for partial observed trees
partial_fit_params = fit_partial_cascade.fit_partial_cascade(sim_post, post_comments, time_observed, inferred_params, display=False)
print("Refined params:", partial_fit_params)


#which params are we using for simulation?
#sim_params = inferred_params
sim_params = partial_fit_params			#for now, always the refined params from partial fit

#SIMULATE COMMENT TREE
sim_events, sim_tree = functions_paper_model.simulate_comment_tree(sim_post, sim_params, group, post_comments, time_observed)


#OUTPUT TREES

#save groundtruth cascade to csv
functions_paper_model.save_groundtruth(sim_post, post_comments, outfile)

#save sim results to json - all simulated events plus some simulation parameters
functions_paper_model.save_sim_json(group, sim_post_id, random_post, time_observed, min_node_quality, max_nodes, estimate_initial_params, sim_events, outfile)

#save sim results to second output file - csv, one event per row, columns 'rootID', 'nodeID', and 'parentID' for now
print("Saving results to", outfile + ".csv...")  
file_utils.save_csv(sim_events, outfile+".csv", fields=['rootID', 'nodeID', 'parentID'])

print("All done\n")
