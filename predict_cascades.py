#given a set of seed posts, predict them all and create submission output file
#requires three command line args: domain, input filename of seed posts, and output filename for simulation json results

import file_utils
from newParamGraph import ParamGraph
import cascade_manip

import sys
import json
from collections import defaultdict


#creates submission json file
#domain - simulation domain, must be one of 'crypto', 'cyber', or 'CVE'
#events - list of simulation events, each a dictionary with the following fields: 
'''
	"rootID": id of root post for cascade
	"actionType": 'post' or 'comment'
	"parentID": id of parent (comment or post), null/self if this object is post 
	"nodeTime": timestamp of event
	"nodeUserID": id of user posting this event
	"nodeID": id of post/comment
	"communityID": id of subreddit			???
'''
#outfile - location of output json file
#generates file of the following form:
"""
	{"team"     : 'usc',
	 "scenario" : 2,
	 "domain"   : domain,
	 "platform" : reddit,
	 "data":[
	 JSON_datapoint,
	 JSON_datapoint,
	 :
	 :
	 :
	 JSON_datapoint,
	 JSON_datapoint]
	 }
"""
def reddit_sim_to_json(domain, events, outfile):
	#header values
    team_name = 'usc'
    scenario = 2
    platform = 'reddit'

    #crypto, cyber, or CVE domain
    if domain != 'crypto' and domain != 'cyber' and domain != 'CVE':
        print("Invalid domain")
        return False        
        
    #write to json
    submission = {'team'     : team_name,
                  'scenario' : scenario,
                  'domain'   : domain,
                  'platform' : platform,
                  'data'     : events}
            
    with open(outfile,'w') as f:
        json.dump(submission, f)

    return
#end reddit_sim_to_json


#given a filename of cascade seed test data, load all seed posts
#lucky us, the input data is multiple json objects in the same file - not a list of them
#at least they're one per line!
def load_reddit_seeds(filename):
    data = []
        
    with open(filename,'r') as f:
        for line in f:
            d = json.loads(line)
            data.append(d)

    print("Read", len(data), "events")

    return data
#end load_reddit_seeds


#main execution begins here
print("")

#verify command line args
if len(sys.argv) != 4:
	print("Incorrect command line arguments\nUsage: python3 predict_cascades.py <domain> <seed filename> <output filename>")
	exit(0)

#extract arguments
domain = sys.argv[1]
infile = sys.argv[2]
outfile = sys.argv[3]

#load post seeds
raw_post_seeds = load_reddit_seeds(infile)

#convert to dictionary of subreddit->list of post objects
post_seeds = defaultdict(list)
for post in raw_post_seeds:
	post_seeds[post['subreddit']].append(post)
print({key : len(post_seeds[key]) for key in post_seeds}, "\n")

#process each subreddit
for subreddit, seeds in post_seeds.items():
	#TESTING ONLY!!!!
	if subreddit != "Bitcoin":
		continue

	print("Processing", subreddit)

	#load subreddit posts (don't need the comments!)
	sub_posts = cascade_manip.load_filtered_posts(domain, subreddit)
	#load subreddit parameters
	sub_params = cascade_manip.load_cascade_params(domain, subreddit+"100")

	#filter posts - TESTING ONLY!!!!!!!!
	sub_posts = {post_id : post for post_id, post in sub_posts.items() if post_id in sub_params}
	print("Filtered to", len(sub_posts), "posts with fitted parameters")

	#verify loads
	if sub_posts == False or sub_params == False:
		print("Load failed - exiting")
		exit(0)

	#build graph of these posts/params
	sub_graph = ParamGraph()
	sub_graph.build_graph(sub_posts, sub_params)

	#add all seed posts from this subreddit to graph
	for post in seeds:
		sub_graph.add_post(post)
	print("Updated graph has", sub_graph.graph.number_of_nodes(), "nodes and", sub_graph.graph.size(), "edges")

	#run node2vec to get embeddings
	sub_graph.run_node2vec()

	#for each post, infer parameters and simulate
	for post in seeds:
		inferred_params = sub_graph.infer_params(post, 'weighted')
		print(inferred_params)
