#functions to get socsim data (crytpo, cve, cyber) into model pipeline

import functions_gen_cascade_model
import file_utils

import pandas as pd
from copy import deepcopy

#filepaths of data and pre-processed files - keeping everything in the same spot, for sanity/simplicity

#raw posts for (domain, domain) - dictionary of post_id -> dictionary of a bunch of stuff
raw_posts_filepath = "data_cache/%s_cascades/%s_cascade_posts.pkl"
#raw comments for (domain, domain)
raw_comments_filepath = "data_cache/%s_cascades/%s_cascade_comments.pkl"  
#processed posts for (domain, domain) - dictionary of post id -> post containing title tokens, author, created utc
processed_posts_filepath = "reddit_data/%s/%s_processed_posts.pkl"
#fitted params for posts for (domain, domain) - dictionary of post id -> params tuple
fitted_params_filepath = "reddit_data/%s/%s_post_params.pkl"
#reconstructed cascades for (domain, domain) - dictionary of post id -> cascade dict, with "time", "num_comments", and "replies", where "replies" is nested list of reply objects
cascades_filepath = "reddit_data/%s/%s_cascades.pkl"
#raw root set for (domain, set_id) - parquet files from IU
raw_root_set_filepath = "reddit_data/reddit_%s_%s_roots.parquet"
#better root set for (domain, domain) - dictionary with items "train" and "test", each a list of post ids
root_set_filepath = "reddit_data/%s/%s_train_test_partition.pkl"


def define_vprint(verbose):
	#hackery: declare a special print function for verbose output
	#make it global here for all the other functions to use
	global vprint
	if verbose:
		def vprint(*args):
			# Print each argument separately so caller doesn't need to
			# stuff everything to be printed into a single string
			for arg in args:
				print(arg, end='')
			print("")
	else:   
		vprint = lambda *a: None      # do-nothing function
#end define_vprint


#given a subreddit (really a domain), load all required data
#returns separate training and testing sets
#if files don't exist, call methods to perform preprocessing
def load_data(domain):
	#get all the posts, cascades, and params
	posts, cascades, params, failed_fit_posts = load_all_data(domain)

	#partition into training and testing sets

	#load list of train/test post_ids
	train_ids, test_ids = load_set_list(domain)

	#partition posts, cascades, params, and failed_fit lists based on these id list

	#training data
	train_posts, train_cascades, train_params, train_failed_fit_posts = filter_data_by_ids(train_ids, posts, cascades, params, failed_fit_posts)
	vprint("Filtered training data to %d posts, %d cascades, and %d params" % (len(train_posts), len(train_cascades), len(train_params)))
	vprint("   %d failed fit posts" % len(train_failed_fit_posts))

	#testing data
	test_posts, test_cascades, test_params, test_failed_fit_posts = filter_data_by_ids(test_ids, posts, cascades, params, failed_fit_posts)
	vprint("Filtered testing data to %d posts, %d cascades, and %d params" % (len(test_posts), len(test_cascades), len(test_params)))
	vprint("   %d failed fit posts" % len(test_failed_fit_posts))

	return train_posts, train_cascades, train_params, train_failed_fit_posts, test_posts, test_cascades, test_params, test_failed_fit_posts
#end load_data	


#for given domain, load all posts, cascades, and fitted params
#if files don't exist, call methods to perform preprocessing
def load_all_data(domain):

	vprint("Loading %s" % domain)

	#posts
	#load posts if processed file exists
	if file_utils.verify_file(processed_posts_filepath % (domain, domain)):
		posts = file_utils.load_pickle(processed_posts_filepath % (domain, domain))
	#if posts file doesn't exist, create it - loading in the process
	else:
		vprint("   Processed posts file doesn't exist, creating now")
		posts = process_posts(domain)
	vprint("   Loaded %d posts" % len(posts))

	#cascades
	cascades = get_cascades(domain, posts)
	vprint("   Loaded %d cascades" % len(cascades))

	#throw out posts that we don't have cascades for - incomplete or some other issue
	if len(posts) != len(cascades):
		posts, deleted_count = functions_gen_cascade_model.filter_dict_by_list(posts, list(cascades.keys()), num_deleted=True)
		vprint("   Deleted %d posts without cascades" % deleted_count)
		cascades, deleted_count = functions_gen_cascade_model.filter_dict_by_list(cascades, list(posts.keys()), num_deleted=True)
		vprint("   Deleted %d cascades without posts" % deleted_count)
	
	#params
	#load if params file exists
	if file_utils.verify_file(fitted_params_filepath % (domain, domain)):
		params_data = file_utils.load_pickle(fitted_params_filepath % (domain, domain))
		vprint("   Loaded %d fitted params (%d failed fit)" % (len(params_data['params_dict']), len(params_data['failed_fit_list'])))

		#see if there are any posts that we can fit and add to the loaded set
		if len(params_data['params_dict']) + len(params_data['failed_fit_list']) != len(cascades):
			#small set of cascades to fit
			cascades_to_fit = {cascade_id: cascade_obj for cascade_id, cascade_obj in cascades.items() if cascade_id not in params_data['params_dict'] and cascade_id not in params_data['failed_fit_list']}
			vprint("Fitting %d cascades for %s" % (len(cascades_to_fit), domain))
			#fit these missing cascades
			update_params_out = functions_gen_cascade_model.fit_posts_from_cascades(cascades_to_fit)
			#add to loaded params
			params_data['params_dict'].update(update_params_out['params_dict'])
			params_data['failed_fit_list'].extend(update_params_out['failed_fit_list'])
			#save updated file
			file_utils.save_pickle(params_data, fitted_params_filepath % (domain, domain))
		#if params file doesn't exist, create it - loading in the process
	else:
		vprint("   Fitted params file doesn't exist, creating now")
		params_data = fit_posts(domain, cascades)
	#extract successfully fitted params and list of failed posts
	params = params_data['params_dict']
	failed_fit_posts = params_data['failed_fit_list']

	#throw out posts that we don't have params (or fail note) for - neg comment times or some other issue
	if len(posts) != len(params)+len(failed_fit_posts):
		posts, deleted_count = functions_gen_cascade_model.filter_dict_by_list(posts, list(params.keys())+failed_fit_posts, num_deleted=True)
		vprint("   Deleted %d posts without params" % deleted_count)
		params, deleted_count = functions_gen_cascade_model.filter_dict_by_list(params, list(posts.keys()), num_deleted=True)
		vprint("   Deleted %d params without posts" % deleted_count)

	vprint("Loaded %d posts " % len(posts))
	vprint("   %d cascades" % len(cascades))
	vprint("   %d fitted params, %d failed fit" % (len(params), len(failed_fit_posts)))

	#return results
	return posts, cascades, params, failed_fit_posts
#end load_all_data


#for a given domain, preprocess those posts - tokenize and save as pickle
def process_posts(domain):
	#make sure raw posts exist
	if file_utils.verify_file(raw_posts_filepath % (domain, domain)) == False:
		print("No raw posts to process - exiting")
		exit(0)
	#load raw posts
	vprint("Loading raw posts for %s" % domain)
	raw_posts = file_utils.load_pickle(raw_posts_filepath % (domain, domain))

	#convert to our nested dictionary structure
	posts = {}
	for post_id, post_object in raw_posts.items():
		#check for good post, fail and error if something is amiss
		if post_object['title_m'] is None or post_object['created_utc'] is None or post_object['author_h'] is None:
			print("Skipping invalid post %s" % post_id)
			continue

		#build new post dict
		post = {}
		post['tokens'] = functions_gen_cascade_model.extract_tokens(post_object['title_m'])
		if post['tokens'] == False:
			print(post_id, post_object)
			exit(0)
		post['time'] = int(post_object['created_utc'])
		post['author'] = post_object['author_h']
		post['subreddit'] = post_object['subreddit']	#extra field for just this data - in case we need/want it later

		#add to overall post dict
		if post_object['name_h'] is None:
			real_post_id = "t3_" + post_id
		else:
			real_post_id = post_object['name_h']		#post id with t3_ prefix
		posts[real_post_id] = post

	#save to pickle 
	file_utils.save_pickle(posts, processed_posts_filepath % (domain, domain))

	return posts
#end process_posts


#given domain loaded posts, load cascades associated with those posts
#if cascades don't exist, build them
#returns cascades dictionary
def get_cascades(domain, posts):
	#if reconstructed cascades already exist, load those
	if file_utils.verify_file(cascades_filepath % (domain, domain)):
		cascades = file_utils.load_pickle(cascades_filepath % (domain, domain))		
	#otherwise, reconstruct the cascades (get them back as return value)
	else:
		vprint("Reconstructing cascades for %s" % domain)
		#load comments
		comments = load_comments(domain, posts)

		#reconstruct the cascades
		cascades = build_and_save_socsim_cascades(domain, posts, comments)

	return cascades
#end get_cascades


#given domain and loaded posts, load comments associated with those posts
#(filtering out other comments we may encounter along the way)
#returns nested dictionary of comment_id -> link_id, parent_id, and time (all ids with prefixes)
def load_comments(domain, posts):
	#build set of post ids we care about
	post_ids = set(posts.keys())

	comments = {}			#all relevant comments

	#load raw comments
	raw_comments = file_utils.load_pickle(raw_comments_filepath % (domain, domain))

	#convert to our nested dictionary structure, filtering out irrelevant comments along the way
	for comment_id, comment_object in raw_comments.items():
		#if comment object link and parent fields don't have the t3_/t1_ prefix, add it now
		#post parent always a post with t3_ prefix
		if comment_object['link_id_h'][:3] != "t3_":
			comment_object['link_id_h'] = "t3_" + comment_object['link_id_h']
		#immediate parent could be a post or a comment
		if comment_object['parent_id_h'][:3] != "t3_" and comment_object['parent_id_h'][:3] != "t1_":
			#if link and parent match, immediate parent is a post
			if "t3_"+comment_object['parent_id_h'] == comment_object['link_id_h']:
				comment_object['parent_id_h'] = "t3_" + comment_object['parent_id_h']
			#otherwise, immediate parent is comment
			else:
				comment_object['parent_id_h'] = "t1_" + comment_object['parent_id_h']

		#skip comment if not for post in set
		if comment_object['link_id_h'] not in post_ids:
			print("   No post parent for this comment, skipping")
			continue

		#special-case for these two rogue crypto comments - hardcoding a time as the midpoint of the
		#parent and the child
		if comment_id == 'YlNE1IsFAufmn-RJJE25mw':
			comment_object['created_utc'] = 1431117009
		if comment_id == 'P3hMh-ZJJjG52smrScH76g':
			comment_object['created_utc'] = 1448843362

		#if time field is empty, skip this comment
		if comment_object['created_utc'] is None:
			print("   No time for this comment, skipping")
			continue

		#build new comment dict
		comment = {}
		comment['time'] = int(comment_object['created_utc'])
		comment['link_id'] = comment_object['link_id_h']
		comment['parent_id'] = comment_object['parent_id_h']
		comment['text'] = comment_object['body_m']
		comment['author'] = comment_object['author_h']

		#add to overall comment dict
		comment_id = "t1_" + comment_object['id_h']		#post id with t1_ prefix
		comments[comment_id] = comment

	vprint("Total of %d comments for %s posts (of %d scanned)" % (len(comments), domain, len(raw_comments)))
	return comments
#end load_comments

#given a domain, dict of posts and dict of relevant comments, 
#reconstruct the post/comment (cascade) structure
#heavy-lifting done in build_cascades, this just handles the load/save
def build_and_save_socsim_cascades(domain, posts, comments):
	vprint("Extracting post/comment structure for %d %s posts and %d comments" % (len(posts), domain, len(comments)))

	cascades = functions_gen_cascade_model.build_cascades(posts, comments)

	#save cascades for later loading
	file_utils.save_pickle(cascades, cascades_filepath % (domain, domain))

	return cascades
#end build_and_save_socsim_cascades


#given a domain, and loaded posts (but not comments),
#fit parameters for these posts and save results as pickle
#load cascades if they already exist, otherwise build them first
#cascades format is post_id -> nested dict of replies, time, comment_count_total and comment_count_direct
#each reply has id, time, and their own replies field
def fit_posts(domain, cascades):

	#fit parameters to each cascade
	vprint("Fitting %d cascades for %s" % (len(cascades), domain))

	params_out = functions_gen_cascade_model.fit_posts_from_cascades(cascades)

	file_utils.save_pickle(params_out, fitted_params_filepath % (domain, domain))
	
	return params_out		#return params + fail list in dict
#end fit_posts


#load set of post ids (roots) for either training or testing set
def load_set_list(domain):
	#pickle exists, load it
	if file_utils.verify_file(root_set_filepath % (domain, domain)):
		sets = file_utils.load_pickle(root_set_filepath % (domain, domain))
		train_ids = sets['train']
		test_ids = sets['test']
		#and add t3_ prefix to all
		train_ids = ["t3_"+post_id for post_id in train_ids]
		test_ids = ["t3_"+post_id for post_id in test_ids]

	#doesn't exist, load from icky parquet and convert
	else:
		#training set
		df = file_utils.load_parquet(raw_root_set_filepath % (domain, "train"))
		train_ids = list(df['root_id'])
		#testing set
		df = file_utils.load_parquet(raw_root_set_filepath % (domain, "test"))
		test_ids = list(df['root_id'])
		#dict for pickle save
		sets = {"train": train_ids, "test": test_ids}
		#save as pickle
		file_utils.save_pickle(sets, root_set_filepath % (domain, domain))

	vprint("Loaded %d training and %d testing post ids" % (len(train_ids), len(test_ids)))
	return train_ids, test_ids
#end load_set_list


#given posts, cascades, params, and failed fit list, filter to match a set of post_ids
def filter_data_by_ids(id_set, posts, cascades, params, failed_fit_posts):
	posts = functions_gen_cascade_model.filter_dict_by_list(posts, id_set)
	cascades = functions_gen_cascade_model.filter_dict_by_list(cascades, id_set)
	params = functions_gen_cascade_model.filter_dict_by_list(params, id_set)
	failed_fit_posts = [post_id for post_id in failed_fit_posts if post_id in id_set]

	return posts, cascades, params, failed_fit_posts
#end filter_data_by_ids