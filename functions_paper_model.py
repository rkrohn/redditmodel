#functions for paper_model.py - offloading and modularizing all the things

import file_utils
import functions_hybrid_model

from shutil import copyfile
import subprocess
import os


#filepaths of pre-computed model files
posts_filepath = "model_files/posts/%s_posts.pkl"		#processed post data for each post, one file per group
														#each post maps original post id to numeric id, set of tokens, and user id

params_filepath = "model_files/params/%s_params.txt"	#text file of fitted cascade params, one file per group
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)

graph_filepath = "model_files/graphs/%s_graph.txt"		#edgelist of post graph for this group

#filepaths of output/temporary files - used to pass graph to C++ node2vec for processing
temp_graph_filepath = "sim_files/%s_graph.txt"			#updated graph for this sim run
temp_params_filepath = "sim_files/%s_in_params.txt"		#temporary, filtered params for sim run (if sampled graph)
output_params_filepath = "sim_files/%s_params.txt"		#output params from node2vec


#for a given post, infer parameters using post graph
def graph_infer(sim_post, sim_post_id, group, max_nodes, min_node_quality, estimate_initial_params):
	#load preprocessed posts for this group
	if file_utils.verify_file(posts_filepath % group):
		posts = file_utils.load_pickle(posts_filepath % group)
		print("Loaded", len(posts), "processed posts from", posts_filepath % group)
	else:
		print("Cannot simulate for group", group, "without processed posts file", posts_filepath % group)
		exit(0)

	#if seed post not in posts file - we're gonna have a bad time
	if sim_post['id_h'] not in posts:
		print("Simulation post not in dataset - exiting\n")
		exit(0)

	#grab numeric/graph id of sim post
	numeric_sim_post_id = posts[sim_post_id]['id']

	#load in fitted simulation params - need these for graph build
	fitted_params, fitted_quality = functions_hybrid_model.load_params(params_filepath % group, posts, False, True)	

	#remove sim post from graph params - no cheating! (pop based on numeric id)
	res = fitted_params.pop(numeric_sim_post_id)
	res = fitted_quality.pop(numeric_sim_post_id)

	#graph stuff - sample graph if necessary, add new nodes, etc
	graph = {}
	isolated_nodes = []
	added_count = 0
	sample_graph = False

	#do we need to sample the graph? sample if whole graph too big, imposing a min node quality, need to estimate initial params, we don't have a precomputed graph file
	if len(posts) > max_nodes or file_utils.verify_file(graph_filepath % group) == False or min_node_quality != -1 or estimate_initial_params:
		print("\nSampling graph to", max_nodes, "nodes")
		#sample down posts
		graph_posts = functions_hybrid_model.user_sample_graph(posts, [sim_post], max_nodes, group, min_node_quality, fitted_quality)
		#build graph, getting initial param estimate if required
		if estimate_initial_params:
			estimated_params = functions_hybrid_model.build_graph_estimate_node_params(graph_posts, fitted_params, fitted_quality, numeric_sim_post_id, temp_graph_filepath % group)
		else:
			functions_hybrid_model.build_graph(graph_posts, temp_graph_filepath % group)
		
		sample_graph = True
	#no graph sample, use the full set and copy graph file to temp location
	else:
		graph_posts = posts
		copyfile(graph_filepath % group, temp_graph_filepath % group)
		print("Copied complete post-graph to", temp_graph_filepath % group)

	#ALWAYS sample down params to match whatever graph we have - because we can't use the previously fitted params!
	if estimate_initial_params:
		functions_hybrid_model.get_graph_params(graph_posts, numeric_sim_post_id, fitted_params, fitted_quality, temp_params_filepath % group, estimated_params)
	else:
		functions_hybrid_model.get_graph_params(graph_posts, numeric_sim_post_id, fitted_params, fitted_quality, temp_params_filepath % group)

	#graph is built and ready - graph file and input params file

	#run node2vec to get embeddings - if we have to infer parameters
	#offload to C++, because I feel the need... the need for speed!:

	if file_utils.verify_file(output_params_filepath % group):
		os.remove(output_params_filepath % group)		#clear output to prevent append

	#run node2vec on graph and params
	subprocess.check_call(["./c_node2vec/examples/node2vec/node2vec", "-i:"+(temp_graph_filepath % group), "-ie:"+(temp_params_filepath % group), "-o:"+(output_params_filepath % group), "-d:6", "-l:3", "-w", "-s"])
	print("")

	#load the inferred params (dictionary of numeric id -> params)
	all_inferred_params = functions_hybrid_model.load_params(output_params_filepath % group, posts, inferred=True)
	inferred_params = all_inferred_params[numeric_sim_post_id]

	return inferred_params
#end graph_infer

#given a list of events, check event times to make sure they are in sorted order
def verify_sorted(events):
	prev_time = -1
	for event in events:
		curr_time = event['nodeTime']
		if prev_time != -1 and curr_time < prev_time:
			print("out of order!", prev_time, curr_time)
			return False
		prev_time = curr_time
	print("Events are sorted")
	return True
#end verify_sorted