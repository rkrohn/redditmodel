import data_utils
import file_utils

#given a list of posts and a list of comments, reconstruct the post/comment (cascade) structure
#store cascade in the following way using a dictionary
#	post id -> post object
# 	post/comment replies field -> list of direct replies
def build_cascades(posts, comments, code):
	print("Extracting post/comment structure for", len(posts), "and", len(comments), "comments")

	#add replies field to all posts/comments, init to empty list
	data_utils.add_field(posts, "replies", [])
	data_utils.add_field(comments, "replies", [])

	#add comment_count field to all post objects as well: count total number of comments all the way down the cascade
	data_utils.add_field(posts, "comment_count", 0)

	#grab list of fields for each type of object (used to create placeholders when items are missing)
	post_fields = list(posts[0].keys())
	comment_fields = list(comments[0].keys())

	'''
	id_h = post/commend id
	parent_id_h = direct parent
	link_id_h = post parent
	if a parent_id starts with t1_, you remove t1_ and match the rest against a comment id 
	if it starts with t3_, you remove t3_ and match the rest against a submission id.
	linked_id always starts with t3_, since it always point to a submission.
	'''

	#create dictionary of post id -> post object to store cascades
	cascades = data_utils.list_to_dict(posts, "id_h")

	#convert list of posts to dictionary, where key is post id
	posts = data_utils.list_to_dict(posts, "id_h")
	#print(posts["90fvEAzkd97SnUviXxKj3Q"])
	#and do the same with the comments
	comments = data_utils.list_to_dict(comments, "id_h")

	print("dicts contain", len(posts), "posts and", len(comments), "comments")

	#now that we can find posts and comments at will, let's build the dictionary!
	
	#loop all comments, assign to immediate parent and increment comment_count of post parent
	comment_count = 0
	missing_comments = set()	#missing comments
	missing_posts = set()		#missing posts
	for comment_id, comment in comments.copy().items():
		#get immediate parent (post or comment)
		#print("comment", comment_id)
		direct_parent = comment['parent_id_h'][3:]
		direct_parent_type = "post" if comment['parent_id_h'][:2] == "t3" else "comment"
		#print("   direct parent", direct_parent, direct_parent_type)
		#get post parent
		post_parent = comment['link_id_h'][3:]
		#print("   post parent", post_parent)
		print(comment_id, "->", direct_parent, direct_parent_type, "->", post_parent)
		comment_count += 1

		#add this comment to replies list of immediate parent
		try:
			#if post parent missing, create placeholder
			if post_parent not in cascades:
				print("post parent fail")
				cascades[post_parent] = create_object(post_parent, post_fields)
				missing_posts.add(post_parent)
			#update overall post comment count
			cascades[post_parent]['comment_count'] += 1
			print("post parent good")

			#now handle direct parent, post or comment
			#parent is post
			if direct_parent_type == "post":
				#missing post, create placeholder to hold replies
				if direct_parent not in cascades:
					print("direct parent (post) fail")
					cascades[direct_parent] = create_object(direct_parent, post_fields)
					missing_posts.add(direct_parent)
				#add this comment to replies field of post
				cascades[direct_parent]['replies'].append(comment_id)
				print("direct parent (post) good")
			#parent is comment
			else:	
				#missing comment, create placeholder to contain replies, point to parent post by default
				if direct_parent not in comments:
					print("direct parent (comment) fail")
					temp = create_object(direct_parent, comment_fields)
					print(temp)
					comments[direct_parent] = temp
					print("here1")
					#point this placeholder comment to the top-level post
					comments[direct_parent]['link_id_h'] = post_parent
					cascades[post_parent]['comment_count'] += 1		#add manufactured comment to counter
					cascades[post_parent]['replies'].append(direct_parent)	#and add to replies
					print("here2")
					comments[direct_parent]['parent_id_h'] = post_parent
					print("here3")
					missing_comments.add(direct_parent)
					print("here4")
				#add current comment to replies field of parent comment
				comments[direct_parent]['replies'].append(comment_id)
				print("direct parent (comment) good")
		except:
			print("FAIL")
			print(len(missing_posts), "posts")
			print(len(missing_comments), "comments")
			for field in comment:
				if field != "replies":
					print(field, comment[field])
			exit(0)
	print("processed", comment_count,  "comments\n")

	print(len(missing_posts), "missing posts")
	print(len(missing_comments), "missing comments")

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
	print("Saving cascades to data_cache/%s_cascade_<file contents>.pkl" % code)
	file_utils.save_pickle(cascades, "data_cache/%s_cascade_posts.pkl" % code)
	file_utils.save_pickle(comments, "data_cache/%s_cascade_comments.pkl" % code)
	file_utils.save_pickle(missing_posts, "data_cache/%s_cascade_missing_posts.pkl" % code)
	file_utils.save_pickle(missing_comments, "data_cache/%s_cascade_missing_comments.pkl" % code)

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
	if 'comment_count' in fields:
		obj['comment_count'] = 0
	print("created")
	print(obj)
	return obj
#end create_object
