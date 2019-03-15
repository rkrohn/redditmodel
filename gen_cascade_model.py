#new model for the cascade prediction/scenario 2 paper

#given a single cascade (top-level post and any observed comments), simulate the remainder of the cascade
#offload node2vec to c++, because speed

#requires the following command-line args: subreddit (or hackernews or cve), id of cascade post to predict (or "random"), output filename for simulation results (no extension), max number of nodes for infer graph, minimum node quality for graph inference (set to -1 for no filter), esp (optional, for estimating initial params based on surrouding)

#for example:
#	python3 paper_model.py pivx random 4 sim_tree 2000 -1
#	python3 paper_model.py pivx 26RPcnyIuA0JyQpTqEui7A 1 sim_tree 500 -1			(4 comments)
#	paper_model.py pivx ZeuF7ZTDw3McZUOaosvXdA 5 sim_tree 250 -1					(11 comments)
#	paper_model.py compsci qOjspbLmJbLMVFxYbjB1mQ 200 sim_tree 250 -1				(58 comments)


import file_utils
import functions_gen_cascade_model
import cascade_manip
import fit_partial_cascade

print("")

#parse all command-line arguments
subreddit, input_sim_post_id, time_observed, outfile, max_nodes, min_node_quality, estimate_initial_params, batch, testing_start_month, testing_start_year, testing_len, training_len = functions_gen_cascade_model.parse_command_args()

#ensure working directory exists
file_utils.verify_dir("sim_files")		

#load posts and comments for this subreddit
raw_posts, raw_comments = functions_gen_cascade_model.load_subreddit_data(subreddit)

#ensure post id is in dataset (gets list of all post ids if running all)
sim_post_id_list, random_post = functions_gen_cascade_model.verify_post_id(input_sim_post_id, batch, list(raw_posts.keys()))

#if running in mode all, keep total of all metrics, dump at end
if batch:
	total_dist = 0
	total_update_count = 0
	total_update_time = 0
	total_insert_count = 0
	total_insert_time = 0
	total_remove_count = 0
	total_remove_time = 0
	total_match_count = 0

#process all posts (or just one, if doing that)
print("Processing", len(sim_post_id_list), "post", "s" if len(sim_post_id_list) > 1 else "")
for sim_post_id in sim_post_id_list:

	#pull out just the post (and associated comments) we care about
	sim_post = raw_posts[sim_post_id]
	#and filter to comments
	junk, post_comments = cascade_manip.filter_comments_by_posts({sim_post_id: sim_post}, raw_comments, False)
	if batch == False:
		print("Simulation post has", len(post_comments), "comments\n")


	#GRAPH INFER
	inferred_params = functions_gen_cascade_model.graph_infer(sim_post, sim_post_id, subreddit, max_nodes, min_node_quality, estimate_initial_params)
	#inferred_params = [1.73166, 0.651482, 1.08986, 0.762604, 2.49934, 0.19828]		#placeholder if skipping the infer
	if batch == False:
		print("Inferred params:", inferred_params, "\n")


	#REFINE PARAMS - for partial observed trees
	partial_fit_params = fit_partial_cascade.fit_partial_cascade(sim_post, post_comments, time_observed, inferred_params, display=False)
	if batch == False:
		print("Refined params:", partial_fit_params)


	#which params are we using for simulation?
	#sim_params = inferred_params
	sim_params = partial_fit_params			#for now, always the refined params from partial fit

	#SIMULATE COMMENT TREE
	sim_events, sim_tree = functions_gen_cascade_model.simulate_comment_tree(sim_post, sim_params, subreddit, post_comments, time_observed)


	#OUTPUT TREES

	#for now, only output if doing a single post
	if batch == False:
		#save groundtruth cascade to csv
		functions_gen_cascade_model.save_groundtruth(sim_post, post_comments, outfile)

		#save sim results to json - all simulated events plus some simulation parameters
		functions_gen_cascade_model.save_sim_json(subreddit, sim_post_id, random_post, time_observed, min_node_quality, max_nodes, estimate_initial_params, sim_events, outfile)

		#save sim results to second output file - csv, one event per row, columns 'rootID', 'nodeID', and 'parentID' for now
		print("Saving results to", outfile + ".csv...")  
		file_utils.save_csv(sim_events, outfile+".csv", fields=['rootID', 'nodeID', 'parentID'])
		print("")


	#EVAL

	#compute tree edit distance between ground-truth and simulated cascades
	dist, update_count, update_time, insert_count, insert_time, remove_count, remove_time, match_count = functions_gen_cascade_model.eval_trees(sim_tree, sim_post, post_comments)
	if batch == False:
		print("Tree edit distance:", dist)
		print("   update:", update_count, update_time)
		print("   insert:", insert_count, insert_time)
		print("   remove:", remove_count, remove_time)
		print("   match:", match_count)

	#if running in mode all, keep total of all these metrics, dump at end
	if batch:
		total_dist += dist
		total_update_count += update_count
		total_update_time += update_time
		total_insert_count += insert_count
		total_insert_time += insert_time
		total_remove_count += remove_count 
		total_remove_time += remove_time
		total_match_count += match_count

print("\nAll done\n")

#if mode == all, print metric totals
if batch:
	print("Number of posts:", len(sim_post_id_list))
	print("Time Observed:", time_observed)
	print("Source subreddit:", subreddit)
	if min_node_quality != -1:
		print("Minimum node quality:", min_node_quality)
	else:
		print("No minimum node quality")
	print("Max graph size:", max_nodes, "from file" if limit_from_file else "from argument")
	if estimate_initial_params:
		print("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
	
	print("Tree edit distance:", total_dist)
	print("   update:", total_update_count, total_update_time)
	print("   insert:", total_insert_count, total_insert_time)
	print("   remove:", total_remove_count, total_remove_time)
	print("   match:", total_match_count)
	print("")
