import load_model_data
import cascade_analysis

#driver for all the other things


code = "cve"			#set use case/domain: must be crypto, cyber, or cve
						#crypto for dry run
						#cyber takes forever
						#cve fastest

print("Processing", code)

#load data and build cascades
#posts, comments = load_reddit_data(code)
#cascades, comments, missing_posts, missing_comments = cascades.build_cascades(posts, comments, code)

#load all exogenous data for this use case/domain
#load_exogenous_data(code)

#build/load cascades (auto-load as a result, either raw data or cached cascades)
cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(code)

#optional: filter out cascades with any missing elements (posts or comments)
cascades, comments = cascade_analysis.remove_missing(code, cascades, comments)

#get subreddit distribution
#cascade_analysis.get_subreddits(code, cascades)

#get/plot top-level comment response time distribution
cascade_analysis.top_level_comment_response_dist(code, cascades, comments)
