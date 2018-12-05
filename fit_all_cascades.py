import cascade_analysis
import cascade_manip
import fit_cascade
import file_utils

#driver for fitting all cascades


code = "crypto"			#set use case/domain: must be crypto, cyber, or cve
						#crypto for dry run
						#cyber takes forever
						#cve fastest

pickle_save = False 	#if True, save fitted parameters dictionary to pickle
						#if False, save to human-readable text file instead

print("\nProcessing", code)

cascades = None
comments = None

#load the subreddit distribution for these cascades
subreddit_dist = file_utils.load_json("results/%s_post_subreddit_dist.json" % code)


#loop all subreddits for this code
for subreddit in sorted(subreddit_dist.keys()):
	
	if subreddit != 'Lisk':
		continue
	

	print("\nProcessing", subreddit)

	#load filtered, if they exist
	filtered_cascades, filtered_comments = cascade_manip.load_filtered_cascades(code, subreddit)

	#don't exist, filter them now
	if filtered_cascades == False:

		#have we loaded the raw cascades/comments yet? if not, do it now
		#(waiting until now in case we have all the filtered versions and don't need these at all)
		if cascades == None or comments == None:
			#build/load cascades (auto-load as a result, either raw data or cached cascades)
			cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(code)
			#optional: filter out cascades with any missing elements (posts or comments)
			cascades, comments = cascade_manip.remove_missing(code, cascades, comments)

		#filter cascades by a particular subreddit
		filtered_cascades = cascade_manip.filter_cascades_by_subreddit(cascades, subreddit)
		#and filter comments to match those posts
		filtered_cascades, filtered_comments = cascade_manip.filter_comments_by_posts(filtered_cascades, comments)
		#save these filtered posts/comments for easier loading later
		cascade_manip.save_cascades(code, filtered_cascades, subreddit)
		cascade_manip.save_comments(code, filtered_comments, subreddit)

	#fit params to all of the filtered cascades, loading checkpoints if they exist
	all_params = cascade_analysis.fit_all_cascades(code, filtered_cascades, filtered_comments, pickle_save, subreddit)	

	#if not saving to pickle, save to text file now
	if pickle_save == False:
		with open("data_cache/txt_params/%s_params.txt" % subreddit, "w") as f: 
			for post_id, params in all_params.items():
				f.write(post_id + " ")
				for i in range(len(params)):
					f.write((' ' if i > 0 else '') + str(params[i]))
				f.write("\n")
		print("Saved text-readable params to data_cache/txt_params/%s_params.txt" % subreddit)
