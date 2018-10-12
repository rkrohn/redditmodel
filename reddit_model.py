import load_model_data
import cascade_analysis

#driver for all the other things


code = "crypto"			#set use case/domain: must be crypto, cyber, or cve

print("Processing", code)

#load data and build cascades
#posts, comments = load_reddit_data(code)
#cascades, comments, missing_posts, missing_comments = cascades.build_cascades(posts, comments, code)

#load all exogenous data for this use case/domain
#load_exogenous_data(code)

#build/load cascades (auto-load as a result, either raw data or cached cascades)
#cascades, comments, missing_posts, missing_comments = cascades.build_cascades(code)

#load cascades and get subreddit distribution
cascades, comments, missing_posts, missing_comments = cascade_analysis.build_cascades(code)
cascade_analysis.get_subreddits(code, cascades)
