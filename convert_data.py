#convert some socialsim reddit data to new reddit data format, so we can do some testing
#(doesn't have to be good, just has to be there)

import file_utils
import functions_gen_cascade_model


#convert to desired nested dictionary structure
def process(in_posts, subreddit, year, month):	
	posts = {}
	for post_id, raw_post in in_posts.items():

		#check for good row, fail and error if something is amiss (probably a non-quoted body)
		if raw_post['title_m'] == None or raw_post['subreddit'] == None or raw_post['created_utc'] == None or raw_post['author_h'] == None:
			print("Invalid post, skipping")
			continue

		#build new post dict
		post = {}
		post['tokens'] = functions_gen_cascade_model.extract_tokens(raw_post['title_m'])
		if post['tokens'] == False:
			continue
		post['time'] = int(raw_post['created_utc'])
		post['author'] = raw_post['author_h']

		#add to overall post dict
		post_id = "t3_" + raw_post['id_h']		#post id with t3_ prefix
		posts[post_id] = post

	#save to pickle 
	processed_posts_filepath = "reddit_data/%s/%s_processed_posts_%d_%d.pkl"
	file_utils.save_pickle(posts, processed_posts_filepath % (subreddit, subreddit, year, month))

	print("Processed %d posts for %d-%d" % (len(posts), month, year))

	return posts
#end process

#convert comments for cascade reconstruction
def filter_comments(subreddit, post_month, post_year, posts, raw_comments):

	#loop comments, only keep the ones that match post
	comments = {}			#all relevant comments
	for comment_id, raw_comment in raw_comments.items():
		#skip comment if not for post in set
		if raw_comment['link_id_h'] not in posts:
			continue

		#build new comment dict
		comment = {}
		comment['time'] = raw_comment['created_utc']
		comment['link_id'] = raw_comment['link_id_h']
		comment['parent_id'] = raw_comment['parent_id_h']
		comment['text'] = raw_comment['body_m']
		comment['author_h'] = raw_comment['author_h']

		#add to overall comment dict
		comments["t1_"+comment_id] = comment

	print("Total of %d comments for %d-%d posts" % (len(comments), post_month, post_year))
	return comments
#end load_comments


#---MAIN BEGINS HERE---#


domain = "crypto"
subreddit = "Lisk"

#load crypto subreddit data - reconstructed cascades
posts = file_utils.load_pickle("data_cache/filtered_cascades/%s_%s_cascades.pkl" % (domain, subreddit))
comments = file_utils.load_pickle("data_cache/filtered_cascades/%s_%s_comments.pkl" % (domain, subreddit))
print("Read %d posts and %d comments" % (len(posts), len(comments)))

#artificial month partitioning - half the posts to 8/16, half to 9/16 (date doesn't matter)
august_posts = dict(list(posts.items())[int(len(posts)/2):])
september_posts = dict(list(posts.items())[:int(len(posts)/2)])
print("Split to %d posts for 8-16 and %d posts for 9-16" % (len(august_posts), len(september_posts)))

#convert both post sets
file_utils.verify_dir("reddit_data/%s" % subreddit)
august_posts = process(august_posts, subreddit, 2016, 8)
september_posts = process(september_posts, subreddit, 2016, 9)

#filter comments to match post sets
august_comments = filter_comments(subreddit, 8, 2016, august_posts, comments)
september_comments = filter_comments(subreddit, 9, 2016, september_posts, comments)

#reconstruct cascades
#hackery required here first - copy the vprint definition into build_cascades, hardcode verbose=True to get it to run
functions_gen_cascade_model.build_cascades(subreddit, 8, 2016, august_posts, august_comments)
functions_gen_cascade_model.build_cascades(subreddit, 9, 2016, september_posts, september_comments)

print("")