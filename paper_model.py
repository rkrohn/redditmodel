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
import sim_tree
from functions_hybrid_model import *
import functions_paper_model
import cascade_manip
import fit_partial_cascade

import sys
import random


#filepaths of pre-computed model files
users_filepath = "model_files/users/%s_users.txt"		#list of users seen in posts/comments, one file per group

limit_filepath = "model_files/group_size_limits.txt"	#per-group graph size limits

#filenames of filtered cascades and comments
cascades_filepath = "data_cache/filtered_cascades/%s_%s_cascades.pkl"	#domain and group cascades
comments_filepath = "data_cache/filtered_cascades/%s_%s_comments.pkl"	#domain and group comments


DISPLAY = True


print("")

#verify command line args
if len(sys.argv) < 7:
	print("Incorrect command line arguments\nUsage: python3 paper_model.py <source group> <seed post id or \"random\"> <time post observed (hours)> <output filename> <max nodes for infer graph (if none given in group_size_limits.txt> <min_node_quality (set to -1 for no filter)> esp(optional, for estimating initial params)")
	exit(0)

#extract arguments
group = sys.argv[1]
sim_post_id = sys.argv[2]
time_observed = float(sys.argv[3])
outfile = sys.argv[4]
max_nodes = int(sys.argv[5])
min_node_quality = float(sys.argv[6])
if len(sys.argv) > 7 and sys.argv[7] == "esp":
	estimate_initial_params = True
else:
	estimate_initial_params = False

#read group-specific size limits from file
with open(limit_filepath, 'r') as f:
	lines = f.readlines()
sub_limits = {}
#process each line, extract group and limit
for line in lines:
	values = line.split()
	sub_limits[values[0]] = int(values[1])
#use limit from file, if it exists
limit_from_file = False
if group in sub_limits:
	max_nodes = sub_limits[group]
	limit_from_file = True

#print some log-ish stuff in case output being piped and saved
print("Post ID:", sim_post_id)
print("Time Observed:", time_observed)
print("Output:", outfile)
print("Source group:", group)
if min_node_quality != -1:
	print("Minimum node quality:", min_node_quality)
else:
	print("No minimum node quality")
print("Max graph size:", max_nodes, "from file" if limit_from_file else "from argument")
if estimate_initial_params:
	print("Estimating initial params for seed posts based on inverse quality weighted average of neighbors")
print("")

file_utils.verify_dir("sim_files")		#ensure working directory exists

#read group (group) -> domain mapping for later file loads
domain_mapping = file_utils.load_pickle("model_files/domain_mapping.pkl")
if group not in domain_mapping:
	print(group, "not in domain mapping - exiting.\n")
	exit(0)
domain = domain_mapping[group]

#load cascades and comments for this group
print("")
raw_posts = file_utils.load_pickle(cascades_filepath % (domain, group))
raw_comments = file_utils.load_pickle(comments_filepath % (domain, group))
print("Loaded", len(raw_posts), "posts and", len(raw_comments), "comments\n")

#if random post id, pick an id from loaded posts
if sim_post_id == "random":
	sim_post_id = random.choice(list(raw_posts.keys()))
	print("Choosing random simulation post:", sim_post_id)

#if given post id not in set, exit
if sim_post_id not in raw_posts:
	print("Given post id not in group set - exiting.\n")
	exit(0)

#pull out just the post we care about
sim_post = raw_posts[sim_post_id]
#and filter to comments
junk, all_comments = cascade_manip.filter_comments_by_posts({sim_post_id: sim_post}, raw_comments, False)
print("Simulation post has", len(all_comments), "comments\n")

#convert ground_truth from given format to eval format
truth_events = []
#include post
truth_events.append({'rootID': "t3_"+sim_post['id_h'], 'nodeID': "t3_"+sim_post['id_h'], 'parentID': "t3_"+sim_post['id_h']})
#and all comments, sorted by time
for comment in sorted(all_comments.values(), key=lambda k: k['created_utc']): 
	truth_events.append({'rootID': comment['link_id_h'], 'nodeID': "t1_"+comment['id_h'], 'parentID': comment['parent_id_h']})

#save ground-truth of this cascade
print("Saving groundtruth as", outfile+"_groundtruth.csv")
file_utils.save_csv(truth_events, outfile+"_groundtruth.csv", fields=['rootID', 'nodeID', 'parentID'])


#GRAPH INFER
inferred_params = functions_paper_model.graph_infer(sim_post, sim_post_id, group, max_nodes, min_node_quality, estimate_initial_params)
#inferred_params = [1.73166, 0.651482, 1.08986, 0.762604, 2.49934, 0.19828]		#placeholder if skipping the infer
print("Inferred params:", inferred_params, "\n")


#REFINE PARAMS - for partial observed trees

partial_fit_params = fit_partial_cascade.fit_partial_cascade(sim_post, all_comments, time_observed, inferred_params, display=False)
print("Refined params:", partial_fit_params, "\n")

#END REFINE PARAMS


#which params are we using for simulation?
#sim_params = inferred_params
sim_params = partial_fit_params			#for now, always the refined params from partial fit


#COMMENT TREE SIM

#load active users list to draw from when assigning users to comments
user_ids = file_utils.load_pickle(users_filepath % group)

#node2vec finished, on to the simulation!
print("\nSimulating comment tree")
sim_root, all_times = sim_tree.simulate_comment_tree(sim_params)

#convert that to desired output format
sim_events = build_cascade_events(sim_root, sim_post, user_ids, group)
sim_events = sorted(sim_events, key=lambda k: k['nodeTime']) 

print("Generated", len(sim_events)-1, "comments for post", sim_post_id)
print("   ", len(all_comments), "actual")

#END COMMENT TREE SIM


#save sim results to output file - json with events and run settings
print("\nSaving results to", outfile + ".json...")      
    
#write to json, include some run info
output = {'group'    				: group,
          'post_id'  				: sim_post_id,
          'time_observed'   		: time_observed,
          'min_node_quality' 		: min_node_quality,
          'max_graph_size' 			: max_nodes,
          'estimate_initial_params' : estimate_initial_params,
          'data'     				: sim_events}
file_utils.save_json(output, outfile+".json")


#save sim results to second output file - csv, one event per row, columns 'rootID', 'nodeID', and 'parentID' for now
print("\nSaving results to", outfile + ".csv...")  
file_utils.save_csv(sim_events, outfile+".csv", fields=['rootID', 'nodeID', 'parentID'])

print("All done\n")
