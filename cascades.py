import data_utils
import file_utils
import os
import glob
import load_model_data

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
		if os.path.exists("data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code)):
			comments = file_utils.load_pickle("data_cache/%s_cascades/%s_cascade_comments.pkl" % (code, code))
		else:
			comments = []
			files = sorted(glob.glob('data_cache/%s_cascades/%s_cascade_comments*' % (code, code)))
			for file in files:
				new_comments = file_utils.load_pickle(file)
				comments.extend(new_comments)
		missing_posts = file_utils.load_json("data_cache/%s_cascades/%s_cascade_missing_posts.json" % (code, code))
		missing_comments = file_utils.load_json("data_cache/%s_cascades/%s_cascade_missing_comments.json" % (code, code))
		print("   Loaded", len(cascades), "cascades with", len(comments), "comments")
		return cascades, comments, missing_posts, missing_comments

	#if no cached cascades, build them from scratch

	#if no loaded posts/comments, load those up first
	posts, comments = load_model_data.load_reddit_data(code)

	print("Extracting post/comment structure for", len(posts), "and", len(comments), "comments")

	#add replies field to all posts/comments, init to empty list
	data_utils.add_field(posts, "replies", [])
	data_utils.add_field(comments, "replies", [])
	#add placeholder field to all posts/comments, flag indicates if we created a dummy object
	data_utils.add_field(posts, 'placeholder', False)
	data_utils.add_field(comments, 'placeholder', False)

	#add comment_count field to all post objects as well: count total number of comments all the way down the cascade
	data_utils.add_field(posts, "comment_count", 0)
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
	for comment_id, comment in comments.copy().items():
		#get immediate parent (post or comment)
		direct_parent = comment['parent_id_h'][3:]
		direct_parent_type = "post" if comment['parent_id_h'][:2] == "t3" else "comment"
		#get post parent
		post_parent = comment['link_id_h'][3:]
		comment_count += 1

		#add this comment to replies list of immediate parent
		try:
			#if post parent missing, create placeholder
			if post_parent not in cascades:
				cascades[post_parent] = create_object(post_parent, post_fields)
				missing_posts.add(post_parent)
			#update overall post comment count
			cascades[post_parent]['comment_count'] += 1

			#now handle direct parent, post or comment
			#parent is post
			if direct_parent_type == "post":
				#missing post, create placeholder to hold replies
				if direct_parent not in cascades:
					cascades[direct_parent] = create_object(direct_parent, post_fields)
					missing_posts.add(direct_parent)
				#add this comment to replies field of post
				cascades[direct_parent]['replies'].append(comment_id)
			#parent is comment
			else:	
				#missing comment, create placeholder to contain replies, point to parent post by default
				if direct_parent not in comments:
					temp = create_object(direct_parent, comment_fields)
					comments[direct_parent] = temp
					#point this placeholder comment to the top-level post
					comments[direct_parent]['link_id_h'] = post_parent
					cascades[post_parent]['comment_count'] += 1		#add manufactured comment to counter
					cascades[post_parent]['replies'].append(direct_parent)	#and add to replies
					comments[direct_parent]['parent_id_h'] = post_parent
					cascades[post_parent]['missing_comments'] = True	#flag this cascade as containing missing comments
					missing_comments.add(direct_parent)		#add comment to list of missing
				#add current comment to replies field of parent comment
				comments[direct_parent]['replies'].append(comment_id)
		except:
			print("FAIL")
			print(len(missing_posts), "posts")
			print(len(missing_comments), "comments")
			for field in comment:
				if field != "replies":
					print(field, comment[field])
			exit(0)

	print("\nProcessed", comment_count,  "comments in", len(cascades), "cascades")
	print("   ", len(missing_posts), "missing posts")
	print("   ", len(missing_comments), "missing comments")
	print("   ", len([x for x in cascades if cascades[x]['missing_comments']]), "cascades with missing comments")

	#verify the above process, a couple different ways

	#count comments from parent counters across all cascades
	total_comments = 0
	for post_id, post in cascades.items():
		total_comments += post['comment_count']
	print(total_comments, "from post counters")

	#traverse each cascade and count comments, check against stored comment count
	'''
	for post_id, post in cascades.items():
		traverse_comments = traverse_cascade(post, comments)
		if traverse_comments != post['comment_count']:
			print("post counter says", post['comment_count'], "comments, but traversal says", traverse_comments)
	'''

	#save cascades for later loading
	#make sure directory exists
	if not os.path.exists("data_cache/%s_cascades" % code):
		os.makedirs("data_cache/%s_cascades" % code)
	print("Saving cascades to data_cache/%s_cascades/%s_cascade_<file contents>.pkl" % (code, code))
	file_utils.save_pickle(cascades, "data_cache/%s_cascades/%s_cascade_posts.pkl" % (code, code))
	save_cascade_comments(code, comments)		#cascade comments
	file_utils.save_json(list(missing_posts), "data_cache/%s_cascades/%s_cascade_missing_posts.json" % (code, code))
	file_utils.save_json(list(missing_comments), "data_cache/%s_cascades/%s_cascade_missing_comments.json" % (code, code))

	return cascades, comments, missing_posts, missing_comments
#end build_cascades

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

#create empty/placeholder object with desired fields and id
#set all fields to None, since we have no data for this missing object
def create_object(identifier, fields):
	obj = {}
	for field in fields:
		obj[field] = None
	obj['id_h'] = identifier
	obj['replies'] = []
	obj['placeholder'] = True
	if 'comment_count' in fields:
		obj['comment_count'] = 0
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
