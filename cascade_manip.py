#handles manipulation of reddit cascades (helper functions)
#for cascade creation and analysis, see cascade_analysis.py

import data_utils
import file_utils
import plot_utils
import os
import glob
import load_model_data
from collections import defaultdict


#given cascades and comments, remove any cascades containing missing elements (posts or comments)
def remove_missing(code, cascades = False, comments = False):
	#load data if missing
	if cascades == False or comments == False:
		print("loading data - build cascades")
		cascades, comments, missing_posts, missing_comments = build_cascades(code)

	print("\nStarting with", len(cascades), "cascades and", len(comments), "comments")

	#loop all posts, only keep ones that are both real and have no missing comments
	cascades = {key:value for (key,value) in cascades.items() if value['placeholder'] == False and value['missing_comments'] == False}

	#loop all comments, only keep the ones that are real and that we kept the parent post for
	comments = {key:value for (key,value) in comments.items() if value['placeholder'] == False and value['link_id_h'][3:] in cascades}

	print("Filtered to", len(cascades), "complete cascades with", len(comments), "comments")

	return cascades, comments
#end remove_missing


#given the root of a cascade (top-level post for first call, comment for remainder), 
#traverse the cascade and count total number of comments
#   node = current node to traverse down from (call on top-level post to traverse entire cascade)
#   comments = dictionary of coment id -> comment object containing all comments
def traverse_cascade(node, comments):
	total_comments = 0		#total number of comments in cascade

	#loop and recurse on all replies
	for reply_id in node['replies']:
		total_comments += traverse_cascade(comments[reply_id], comments)		#add this reply's comments to total
	total_comments += len(node['replies'])		#add all direct comments to total

	return total_comments
#end traverse_cascade


#given a set of cascades and a subreddit, filter cascades to only those from that subreddit
def filter_cascades_by_subreddit(cascades, subreddit):
	filtered_cascades = {}		#dictionary for filtered cascades

	print("Filtering to posts in", subreddit, "subreddit")

	for cascade_id, cascade_post in cascades.items():
		if cascade_post['subreddit'] == subreddit:
			filtered_cascades[cascade_id] = cascade_post

	print("Found", len(filtered_cascades), "for subreddit", subreddit, "(from", len(cascades), "cascades)\n")

	return filtered_cascades
#end filter_cascades by subreddit