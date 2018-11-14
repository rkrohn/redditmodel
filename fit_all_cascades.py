import cascade_analysis
import cascade_manip
import fit_cascade
import file_utils

#driver for fitting all cascades


code = "crypto"			#set use case/domain: must be crypto, cyber, or cve
						#crypto for dry run
						#cyber takes forever
						#cve fastest

print("\nProcessing", code)

#build/load cascades (auto-load as a result, either raw data or cached cascades)
cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(code)
#optional: filter out cascades with any missing elements (posts or comments)
cascades, comments = cascade_manip.remove_missing(code, cascades, comments)

#load the subreddit distribution for these cascades
subreddit_dist = file_utils.load_json("results/%s_post_subreddit_dist.json" % code)

#loop all subreddits for this code
for subreddit in sorted(subreddit_dist.keys()):
	print("\n" + subreddit)

	#load filtered, if they exist
	filtered_cascades, filtered_comments = cascade_manip.load_filtered_cascades(code, subreddit)

	#don't exist, filter them now
	if filtered_cascades == False:
		#filter cascades by a particular subreddit
		filtered_cascades = cascade_manip.filter_cascades_by_subreddit(cascades, subreddit)
		#and filter comments to match those posts
		filtered_comments = cascade_manip.filter_comments_by_posts(filtered_cascades, comments)
		#save these filtered posts/comments for easier loading later
		cascade_manip.save_cascades(code, filtered_cascades, subreddit)
		cascade_manip.save_comments(code, filtered_comments, subreddit)


	#fit params to all of the filtered cascades, loading checkpoints if they exist
	cascade_analysis.fit_all_cascades(code, filtered_cascades, filtered_comments, subreddit)		
