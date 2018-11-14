#handles manipulation of reddit cascades (helper functions)
#for cascade creation and analysis, see cascade_analysis.py

import file_utils
import cascade_analysis
import os


#given cascades and comments, remove any cascades containing missing elements (posts or comments)
def remove_missing(code, cascades = False, comments = False):
	#load data if missing
	if cascades == False or comments == False:
		print("loading data - build cascades")
		cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(code)

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

	print("Found", len(filtered_cascades), "for subreddit", subreddit, "(from", len(cascades), "cascades)")

	return filtered_cascades
#end filter_cascades by subreddit


#given a list of post objects, filter the comments to only those in those cascades
def filter_comments_by_posts(cascades, comments):
	comment_ids = set()		#build set of comment ids to include in filtered dictionary

	print("Filtering comments to match posts")

	#loop all posts, built list of comment ids
	for post_id, post in cascades.items():
		comment_ids.update(get_cascade_comment_ids(post, comments))		#add this posts's comments to overall list

	#filter comments to only those in the list
	filtered_comments = { comment_id : comments[comment_id] for comment_id in comment_ids }

	print("Filtered to", len(filtered_comments), "comments (from", len(comments), "comments)")

	return filtered_comments
#end filter_comments_by_post


#given a single post object, get set of all comment ids in the corresponding cascade
#(must pass in all comments that are a part of that post at minimum)
def get_cascade_comment_ids(post, comments):
	comment_ids = set()		#set to hold overall list of comment ids

	nodes_to_visit = [] + post['replies']	#init queue to direct post replies
	while len(nodes_to_visit) != 0:
		curr = nodes_to_visit.pop(0)	#grab current comment id
		comment_ids.add(curr)			#add this comment to set of cascade comments
		nodes_to_visit.extend(comments[curr]['replies'])	#add this comment's replies to queue

	return comment_ids
#end get_cascade_comment_ids


#save cascades (complete or filtered) to pickle
#if filtered = False, saving all cascades for this code
#if filtered is a string, indicates subreddit cascades are filtered by
def save_cascades(code, cascades, filtered = False):
	if filtered == False:
		file_utils.verify_dir("data_cache/%s_cascades" % code)
		print("Saving cascades to data_cache/%s_cascades/%s_cascade_posts.pkl" % (code, code))
		file_utils.save_pickle(cascades, "data_cache/%s_cascades/%s_cascade_posts.pkl" % (code, code))
	else:
		file_utils.verify_dir("data_cache/filtered_cascades")
		print("Saving filtered cascades to data_cache/filtered_cascades/%s_%s_cascades.pkl" % (code, filtered))
		file_utils.save_pickle(cascades, "data_cache/filtered_cascades/%s_%s_cascades.pkl" % (code, filtered))
#end save_cascades


#save cascade comments (complete or filtered) to pickle
#if filtered = False, saving all comments for this code
#if filtered is a string, indicates subreddit comments are filtered by
def save_comments(code, comments, filtered = False):
	if filtered == False:
		print("Saving comments to data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code))
		#save all comments to pickle
		file_utils.verify_dir("data_cache/%s_cascades" % code)
		#break cyber comments into separate files, because memory error
		if code == "cyber":
			temp = {}		#temporary dictionary to hold a chunk of comments
			count = 0
			for comment_id, comment in comments.items():
				temp[comment_id] = comment
				count += 1
				if count % 1000000 == 0:
					file_utils.save_pickle(temp, "data_cache/%s_cascades/%s_cascade_comments_%s.pkl" % (code, code, count//1000000))
					temp = {}
			#last save
			file_utils.save_pickle(temp, "data_cache/%s_cascades/%s_cascade_comments_%s.pkl" % (code, code, count//1000000))
		else:
			file_utils.save_pickle(comments, "data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code))
	else:
		file_utils.verify_dir("data_cache/filtered_cascades")
		print("Saving filtered comments to data_cache/filtered_cascades/%s_%s_comments.pkl" % (code, filtered))
		file_utils.save_pickle(comments, "data_cache/filtered_cascades/%s_%s_comments.pkl" % (code, filtered))
#end save_comments


#save cascade fitted parameters (complete or filtered) to pickle
#if filtered = False, saving all cascade params for this code
#if filtered is a string, indicates subreddit cascade params are filtered by
def save_cascade_params(code, cascade_params, filtered = False):
	if filtered == False:
		file_utils.verify_dir("data_cache/fitted_params/" % code)
		print("Saving cascade params to data_cache/fitted_params/%s_cascade_params.pkl" % (code, code))
		file_utils.save_pickle(cascade_params, "data_cache/fitted_params/%s_cascade_params.pkl" % (code, code))
	else:
		file_utils.verify_dir("data_cache/fitted_params")
		print("Saving filtered cascades to data_cache/fitted_params/%s_%s_cascade_params.pkl" % (code, filtered))
		file_utils.save_pickle(cascade_params, "data_cache/fitted_params/%s_%s_cascade_params.pkl" % (code, filtered))
#end save_cascade_params


#load saved cascade parameters from pickle
#if filtered = False, loading all cascade params for this code
#if filtered is a string, indicates subreddit cascade params are filtered by
def load_cascade_params(code, filtered = False):
	if filtered == False:
		filename = "data_cache/fitted_params/%s_cascade_params.pkl" % code
	else:
		filename = "data_cache/fitted_params/%s_%s_cascade_params.pkl" % (code, filtered)

	if os.path.exists(filename) == False:		
		print("No saved cascade parameters - exiting")
		exit(0)
	else:
		print("Loading cascade parameters from cache:", filename)
		params = file_utils.load_pickle(filename)

	return params
#end load_cascade_params


#load filtered posts/comments from saved pickle
def load_filtered_cascades(code, subreddit):
	#if files don't exist, quit
	if os.path.exists("data_cache/filtered_cascades/%s_%s_comments.pkl" % (code, subreddit)) == False or os.path.exists("data_cache/filtered_cascades/%s_%s_cascades.pkl" % (code, subreddit)) == False:
		print("No saved filtered cascades")
		return False, False

	print("Loading", subreddit, "posts and comments from cache")

	#load from file
	cascades = file_utils.load_pickle("data_cache/filtered_cascades/%s_%s_cascades.pkl" % (code, subreddit))
	comments = file_utils.load_pickle("data_cache/filtered_cascades/%s_%s_comments.pkl" % (code, subreddit))

	print("Loaded", len(cascades), "posts and", len(comments), "comments")

	return cascades, comments
#end load_filtered_cascades