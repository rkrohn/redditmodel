#handles creation and analysis of reddit cascades
#for cascade manipulation/helper functions, see cascade_manip.py

import data_utils
import file_utils
import plot_utils
import cascade_manip
import fit_cascade
import os
import glob
import load_model_data
from collections import defaultdict
import re

#given a list of posts and a list of comments, reconstruct the post/comment (cascade) structure
#store cascade in the following way using a dictionary
#	post id -> post object
# 	post/comment replies field -> list of direct replies
#if loading directly from cascades, just pass in the code
#otherwise (no cascades to read), pass in loaded posts and comments
#if no cascades and no loaded comments, load them first
def build_cascades(code, posts = False, comments = False):
	#if cascades already exist, read from cache
	if os.path.exists("data_cache/%s_cascades/%s_cascade_posts.pkl" % (code, code)) and (os.path.exists("data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code)) or os.path.exists("data_cache/%s_cascades/%s_cascade_comments_1.pkl" % (code, code))):
		#load from pickle
		print("Loading cascades from data_cache")
		cascades = file_utils.load_pickle("data_cache/%s_cascades/%s_cascade_posts.pkl" % (code, code))
		#comments: either a single file, or multiple files
		print("Loading comments from data_cache")
		if os.path.exists("data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code)):
			comments = file_utils.load_pickle("data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code))
		else:			
			comments = {}
			files = sorted(glob.glob('data_cache/%s_cascades/%s_cascade_comments*' % (code, code)))
			for file in files:
				print("Loading", file)
				new_comments = file_utils.load_pickle(file)
				comments.update(new_comments)
		missing_posts = file_utils.load_json("data_cache/%s_cascades/%s_cascade_missing_posts.json" % (code, code))
		missing_comments = file_utils.load_json("data_cache/%s_cascades/%s_cascade_missing_comments.json" % (code, code))
		print("   Loaded", len(cascades), "cascades with", len(comments), "comments")
		print("     ", len(missing_posts), "missing posts", len(missing_comments), "missing comments")
		return cascades, comments, missing_posts, missing_comments

	#if no cached cascades, build them from scratch

	#if no loaded posts/comments, load those up first
	if posts == False or comments == False:
		posts, comments = load_model_data.load_reddit_data(code)

	print("Extracting post/comment structure for", len(posts), "posts and", len(comments), "comments")

	#add replies field to all posts/comments, init to empty list
	data_utils.add_field(posts, "replies", [])
	data_utils.add_field(comments, "replies", [])
	#add placeholder field to all posts/comments, flag indicates if we created a dummy object
	data_utils.add_field(posts, 'placeholder', False)
	data_utils.add_field(comments, 'placeholder', False)

	#add comment_count field to all post objects as well: count total number of comments all the way down the cascade
	data_utils.add_field(posts, "comment_count_total", 0)
	#and direct replies only
	data_utils.add_field(posts, "comment_count_direct", 0)
	#and add a missing_comments field to all post objects: set True if we find any missing comments in this cascade
	data_utils.add_field(posts, "missing_comments", False)

	#grab list of fields for each type of object (used to create placeholders when items are missing)
	post_fields = list(posts[0].keys())
	comment_fields = list(comments[0].keys())

	'''
	id_h = post/commend id
	parent_id_h = direct parent
	link_id_h = post parent
	if a parent_id starts with t1_, you remove t1_ and match the rest against a comment id 
	if it starts with t3_, you remove t3_ and match the rest against a submission id.
	linked_id always starts with t3_, since it always points to a submission.
	'''

	#create dictionary of post id -> post object to store cascades
	cascades = data_utils.list_to_dict(posts, "id_h")

	#convert list of comments to dictionary, where key is comment id
	comments = data_utils.list_to_dict(comments, "id_h")

	#now that we can find posts and comments at will, let's build the dictionary!
	
	#loop all comments, assign to immediate parent and increment comment_count of post parent
	comment_count = 0
	missing_comments = set()	#missing comments
	missing_posts = set()		#missing posts
	for comment_id in list(comments.keys()):

		#get immediate parent (post or comment)
		direct_parent = comments[comment_id]['parent_id_h'][3:]
		direct_parent_type = "post" if comments[comment_id]['parent_id_h'][:2] == "t3" else "comment"
		#get post parent
		post_parent = comments[comment_id]['link_id_h'][3:]
		comment_count += 1

		#add this comment to replies list of immediate parent, and update counters on post_parent
		try:
			#if post parent missing, create placeholder
			if post_parent not in cascades:
				cascades[post_parent] = create_object(post_parent, post_fields)
				missing_posts.add(post_parent)

			#update overall post comment count for this new comment
			cascades[post_parent]['comment_count_total'] += 1

			#now handle direct parent, post or comment
			#parent is post
			if direct_parent_type == "post":
				#missing post, create placeholder to hold replies
				if direct_parent not in cascades:
					cascades[direct_parent] = create_object(direct_parent, post_fields)
					missing_posts.add(direct_parent)
				#add this comment to replies field of post (no total comment increment, done above)
				cascades[direct_parent]['replies'].append(comment_id)
				#add 1 to direct comment count field
				cascades[direct_parent]['comment_count_direct'] += 1

			#parent is comment
			else:	
				#missing comment, create placeholder to contain replies, point to parent post by default
				if direct_parent not in comments:
					comments[direct_parent] = create_object(direct_parent, comment_fields)
					#point this placeholder comment to the top-level post
					comments[direct_parent]['link_id_h'] = post_parent
					comments[direct_parent]['parent_id_h'] = post_parent
					#add manufactured comment to counters
					cascades[post_parent]['comment_count_total'] += 1
					cascades[post_parent]['comment_count_direct'] += 1	
					#and add to replies	
					cascades[post_parent]['replies'].append(direct_parent)	
					#flag this cascade as containing missing comments
					cascades[post_parent]['missing_comments'] = True	
					missing_comments.add(direct_parent)		#add comment to list of missing
				#add current comment to replies field of parent comment
				comments[direct_parent]['replies'].append(comment_id)
		except:
			print("FAIL")
			print(len(missing_posts), "posts")
			print(len(missing_comments), "comments")
			for field in comments[comment_id]:
				if field != "replies":
					print(field, comments[comment_id][field])
			exit(0)

	print("\nProcessed", comment_count,  "comments in", len(cascades), "cascades")
	print("   ", len(missing_posts), "missing posts")
	print("   ", len(missing_comments), "missing comments")
	print("   ", len([x for x in cascades if cascades[x]['missing_comments']]), "cascades with missing comments")

	#verify the above process, a couple different ways

	#count comments from parent counters across all cascades
	'''
	total_comments = 0
	for post_id, post in cascades.items():
		total_comments += post['comment_count']
	print(total_comments, "from post counters")
	'''

	#traverse each cascade and count comments, check against stored comment count
	'''
	for post_id, post in cascades.items():
		traverse_comments = traverse_cascade(post, comments)
		if traverse_comments != post['comment_count']:
			print("post counter says", post['comment_count'], "comments, but traversal says", traverse_comments)
	'''

	#save cascades for later loading
	cascade_manip.save_cascades(code, cascades)				#cascades
	cascade_manip.save_comments(code, comments)		#comments
	file_utils.save_json(list(missing_posts), "data_cache/%s_cascades/%s_cascade_missing_posts.json" % (code, code))
	file_utils.save_json(list(missing_comments), "data_cache/%s_cascades/%s_cascade_missing_comments.json" % (code, code))

	return cascades, comments, missing_posts, missing_comments
#end build_cascades

#given compiled cascades, return distribution dictionary of subreddit -> number of posts
#function will load cascades if not passed in
#if display == True, print the distribution
def get_subreddits(code, cascades = False, display = False):
	#no cascades, load them first
	if cascades == False:
		cascades, comments, missing_posts, missing_comments = build_cascades(code)

	#get distribution
	subreddit_dist = data_utils.dictionary_field_dist(cascades, 'subreddit')

	#print distribution if desired
	if display:
		for key, value in subreddit_dist.items():
			print(key, value)

	#save distribution to json file
	print("Saving subreddit distribution to results/%s_post_subreddit_dist.json" % code)
	file_utils.verify_dir("results")
	file_utils.save_json(subreddit_dist, "results/%s_post_subreddit_dist.json" % code)

	return subreddit_dist
#end get_subreddits

#given cascades and comments, determine and plot response time distribution of top-level comments only
#take initial post as t = 0, and comment as time since post
#if bin_minutes == 1 and remove_first == True, throw out any comments in the first minute (probably bots)
def top_level_comment_response_dist(code, cascades = False, comments = False, bin_minutes = 1, remove_first = True):
	#load data if missing
	if cascades == False or comments == False:
		cascades, comments, missing_posts, missing_comments = build_cascades(code)

	print("\nComputing top-level comment response time distribution")

	#response time dictionary: time in minutes -> number of responses with that delay
	response_times = defaultdict(int)

	#for each post, look at all top-level replies
	for post_id, post in cascades.items():		#loop posts
		#if this post is a dummy object, throw an error to the user and move on
		if post['placeholder']:
			print("Data contains placeholder post. Please use remove_missing to filter out incomplete cascades first.")
			exit(0)

		post_time = post['created_utc']		#grab post time to compute reply delay

		for comment_id in post['replies']:		#loop replies
			#get response time in minutes for this comment
			response_time = int((comments[comment_id]['created_utc'] - post_time) / (bin_minutes * 60.0)) * bin_minutes

			#if response time is somehow negative, throw an error message but keep running
			if response_time < 0:
				print("Warning: negative response time!")
			#add one to counter for this response time (binned by minutes)
			response_times[response_time] += 1

	#throw out first minute (bots)
	if remove_first == True:
		response_times.pop(0, None)

	#convert frequencies to probability distribution function
	total = sum(response_times.values())
	for key in response_times.keys():
		response_times[key] /= total

	#save response time distribution, but only if bin_minutes = 1
	if bin_minutes == 1:
		print("Saving top-level comment response time distribution to results/%s_top_level_comment_response_time_dist_%s_<options>.json" % (code, bin_minutes))
		file_utils.verify_dir("results")
		file_utils.save_json(response_times, "results/%s_top_level_comment_response_time_dist_%s.json" % (code, bin_minutes))

	#plot everything
	print("Plotting top-level comment response time distribution to plots/%s_top_level_comment_response_times_%s.png" % (code, bin_minutes))
	file_utils.verify_dir("plots")
	plot_utils.plot_dict_data(response_times, "reply delay time (minutes)", "number of replies", "Top-Level Comment Response Time Distribution - %s Minute Bins" % bin_minutes, filename = "plots/%s_top_level_comment_response_times_%s_log.png" % (code, bin_minutes), x_min = 0, log_scale_x = True, log_scale_y = True)
	plot_utils.plot_dict_data(response_times, "reply delay time (minutes)", "number of replies", "Top-Level Comment Response Time Distribution - %s Minute Bins" % bin_minutes, filename = "plots/%s_top_level_comment_response_times_%s_zoom_log.png" % (code, bin_minutes), x_min = 0, x_max = 60*24, log_scale_x = True, log_scale_y = True)
	plot_utils.plot_dict_data(response_times, "reply delay time (minutes)", "number of replies", "Top-Level Comment Response Time Distribution - %s Minute Bins" % bin_minutes, filename = "plots/%s_top_level_comment_response_times_%s_zoom.png" % (code, bin_minutes), x_min = 0, x_max = 60*24)

#end top_level_comment_response_dist

#given loaded cascades, plot our direct reply count against the provided num_comments to check the correlation
def check_comment_count(code, cascades = False):
	#load data if missing
	if cascades == False:
		print("loading data - build cascades")
		cascades, comments, missing_posts, missing_comments = build_cascades(code)

	#data dictionary: key is num_comments field, value is number of direct replies we found
	direct_count_dict = defaultdict(list)
	#and one for the total number of comments, since that might be more what they're giving us
	total_count_dict = defaultdict(list)

	#process each cascade
	for post_id, post in cascades.items():
		direct_count_dict[post['num_comments']].append(post['comment_count_direct'])
		total_count_dict[post['num_comments']].append( post['comment_count_total'])

	#convert lists to average
	for key in direct_count_dict.keys():
		direct_count_dict[key] = sum(direct_count_dict[key]) / len(direct_count_dict[key])
		total_count_dict[key] = sum(total_count_dict[key]) / len(total_count_dict[key])

	#plot results (no save for now)
	file_utils.verify_dir("plots")
	plot_utils.plot_mult_dict_data([direct_count_dict, total_count_dict], ['cascade direct replies', 'cascade total comments'], 'num_comments from data', 'comments from cascade', 'Number of Comments: Given vs Counted', filename = "plots/%s_comment_counts.png" % code)

#end check_comment_count

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

#create empty/placeholder object with desired fields and id
#set all fields to None, since we have no data for this missing object
def create_object(identifier, fields):
	obj = {}
	for field in fields:
		obj[field] = None
	obj['id_h'] = identifier
	obj['replies'] = []
	obj['placeholder'] = True
	if 'comment_count_total' in fields:
		obj['comment_count_total'] = 0
		obj['comment_count_direct'] = 0
	if 'missing_comments' in fields:
		obj['missing_comments'] = True
	return obj
#end create_object

#save cascade comments to pickle
def save_cascade_comments(code, comments):
	#save all comments to pickle
	if not os.path.exists("data_cache/%s_cascades" % code):
		os.makedirs("data_cache/%s_cascades" % code)
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
#end save_comments

#fit model parameters for all cascades given, loading any saved ones to get a headstart
def fit_all_cascades(code, cascades, comments, pickle_save, subreddit = False):
	#if all saved, load from that
	if file_utils.verify_file("data_cache/fitted_params/%s_cascade_params.pkl" % code):
		cascade_params = file_utils.load_pickle("data_cache/fitted_params/%s_cascade_params.pkl" % code)
		fit_fail = [post_id for post_id, post in cascades.items() if post_id not in cascade_params]
		return cascade_params, fit_fail

	#anything to load? if so, load the latest checkpoint
	if pickle_save:
		#build glob filestring - to get all matching checkpoints
		if subreddit == False:
			filename = "data_cache/fitted_params/%s*_cascade_params.pkl" % code
		else:
			filename = "data_cache/fitted_params/%s_%s*_cascade_params.pkl" % (code, subreddit)

		#extract matching filenames and their numeric values, selecting the most complete one to load
		files = glob.glob(filename)
		best_int = -1		#count of records in best file - set to "" if a complete file is found
		for file in files:
			file_int = re.search(r'\d+', file)
			#if no number in filename, have a complete file - use that
			if file_int is None:
				best_int = ""
				break
			else:
				file_int = int(file_int.group())
				if file_int > best_int:
					best_int = file_int

		#load checkpoint, if we have one
		if best_int != -1:
			if subreddit == False:
				cascade_params = cascade_manip.load_cascade_params(code, str(best_int))
			else:
				cascade_params = cascade_manip.load_cascade_params(code, subreddit + str(best_int))
			print("Loaded", len(cascade_params), "fitted cascade parameters")
		#otherwise, empty dictionary
		else:
			cascade_params = {}
	else:
		cascade_params = {}

	avg_quality = 0

	#fit any cascades that have not been fitted before, add to params dictionary: post_id -> params
	post_count = len(cascade_params)
	fit_fail = []
	print("Fitting all cascade models")
	for post_id, post in cascades.items():
		#if this cascade already fitted, and params are valid, skip
		if post_id in cascade_params and (cascade_params[post_id][0] != 20 and cascade_params[post_id][1] != 500 and cascade_params[post_id][2] != 2.3):
			continue

		#fit the current cascade (filtering comments to just this post is not required)
		#print("Fitting cascade", post_id)
		param_res = fit_cascade.fit_cascade_model(post, comments)
		#if negative comment times, skip this cascade and move to next
		if param_res == False:
			fit_fail.append(post_id)
			continue
		cascade_params[post_id] = param_res
		avg_quality += cascade_params[post_id][6]
		post_count += 1
		if post_count % 1000 == 0:
			print("Fitted", post_count, "cascades")
			if pickle_save and post_count % 10000 == 0:
				if subreddit == False:
					cascade_manip.save_cascade_params(code, cascade_params, str(post_count))
				else:
					cascade_manip.save_cascade_params(code, cascade_params, subreddit + str(post_count))

	avg_quality /= len(cascade_params)

	#dump params to file
	print("Fitted a total of", len(cascade_params), "cascades (average quality", str(avg_quality) + ")")
	if pickle_save:
		cascade_manip.save_cascade_params(code, cascade_params, subreddit)

	#return all params, loaded and newly fitted
	return cascade_params, fit_fail
#end fit_all_cascades