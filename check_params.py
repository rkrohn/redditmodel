
import file_utils
import cascade_manip
import fit_cascade

code = "crypto"
subreddit = "Bitcoin"

#load cascades
#cascades, comments = cascade_manip.load_filtered_cascades(code, subreddit)

#check the fits - how many bad initializtion params are left?
#load the param pickle
cascade_params = file_utils.load_pickle("data_cache/fitted_params/%s_%s_cascade_params.pkl" % (code, subreddit))
print("Loaded", len(cascade_params), "fitted params")

fail_count = 0
for post_id, params in cascade_params.items():
	#print(post_id, params)
	
	if params[0] == 20 and params[1] == 500 and params[2] == 2.3:
		print("FIT FAIL", post_id, params)
		fail_count += 1

		'''
		#try to fit this cascade again, get a read on what caused the failure
		print("old params", params)
		post = cascades[post_id]
		junk, post_comments = cascade_manip.filter_comments_by_posts({post_id : post}, comments)
		new_params = fit_cascade.fit_cascade_model(post, post_comments)
		print("new params", new_params)
		'''

print("Found", fail_count, "posts with bad weibull params")


#checking all the cascades - do we find any pre-post comments?
#ran through all of Bitcoin, and none found - something else must be up in the fit process
'''
#load cascades
cascades, comments = cascade_manip.load_filtered_cascades(code, subreddit)

for post_id, post in cascades.items():

	junk, post_comments = cascade_manip.filter_comments_by_posts({post_id : post}, comments)

	fail_count = 0
	for comment_id, comment in post_comments.items():
		#print("\n" + comment['body_m'])
		#print(comment['created_utc'], post['created_utc'])
		if comment['created_utc']  < post['created_utc']:
			#print("EARLY")
			fail_count += 1

print(fail_count, "pre-post comments")
'''


#investigating Alexey's post - why the negative event times?
#appears he has the wrong root time, bug in his cascade reconstruction
'''
#load cascades
cascades, comments = cascade_manip.load_filtered_cascades(code, subreddit)

#pull Alexey's post
post_id = 'xW5V3uLiOb7kFeale-9NyA'
post = cascades[post_id]
post_time = post['created_utc']
print(post)

#and corresponding comments
junk, post_comments = cascade_manip.filter_comments_by_posts({post_id : post}, comments)

#print comment times
fail_count = 0
times = []
for comment_id, comment in post_comments.items():
	print("\n" + comment['body_m'])
	print(comment['created_utc'], post_time)
	times.append(comment['created_utc'])
	if comment['created_utc']  < post['created_utc']:
		print("EARLY")
		fail_count += 1

print(fail_count, "pre-post comments")
print(sorted(times))
'''