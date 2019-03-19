#given a single reddit cascade (assumed to be complete), fit a weibull and log-normal to prepare for cascade simulation
#fit root comments to a Weibull distribution
#fit deeper comments to a log-normal distribution
#see fit_weibull.py and fit_lognormal.py for more details on fitting process

#this is a slight re-write of fit_cascade.py, for the new cascade format implemented in functions_gen_cascade_model.build_cascades


from fit_weibull import *
from fit_lognormal import *
from collections import defaultdict
import random


#given a post, return sorted list of post (root) reply times in minutes (originally stored in seconds),
#taking the post time as 0 and offsetting the comment times accordingly
def get_root_comment_times(post):
	root_comment_times = []     #build list of reply times

	root_time = post['time']     #get post time in seconds to use as offset

	#list of sorted reply times, shifted by post time
	replies = sorted([(comment['time'] - root_time) / 60.0 for comment in post['replies']])

	#check for negative comment times, if any exist return False
	if not all(reply >= 0 for reply in replies):
		return False

	return replies      #return sorted replies
#end get_root_comment_times


#get list of comment times of all comments not on post (root) in minutes, offset based on the post time
def get_other_comment_times(post):
	other_comment_times = []    #list of comment times

	root_time = post['time']    #get post time in seconds to use as offset

	#init queue to second-level replies (ie, replies to root replies)
	nodes_to_visit = []
	for comment in post['replies']:  
		nodes_to_visit.extend(comment['replies']) 
	while len(nodes_to_visit) != 0:
		curr = nodes_to_visit.pop(0)    #grab current comment
		other_comment_times.append(curr['time'])           #add this comment to set of cascade comments
		nodes_to_visit.extend(curr['replies'])    #add this comment's replies to queue

	#sort and shift comment times
	other_comment_times = sorted([(time - root_time) / 60.0 for time in other_comment_times])

	#if any < 0, return false
	if not all(reply >= 0 for reply in other_comment_times):
		return False

	return other_comment_times
#end get_other_comment_times


#estimate branching number n_b: 1 - (root degree / total comments)
#pass in number of root replies and number of comment replies
def estimate_branching_factor(root_replies, comment_replies):
	if root_replies + comment_replies != 0:
		n_b = 1 - (root_replies / (root_replies + comment_replies))  
	else:
		n_b = 0
	return n_b
#end estimate_branching_factor


#given a single cascade and associated comments, fit both the root-comment Weibull and 
#deeper-comment lognormal distributions, and estimate the branching factor
#return quality estimate for optional quality filter in graph infer step
#both weibull and lognormal fit set to return False for all params if no comments or fit fails
#if this occurs for either, set all params and quality to False
def fit_params(post):

	#fit weibull to root comment times
	root_comment_times = get_root_comment_times(post)
	if root_comment_times == False:
		return False
	a, lbd, k, weibull_quality = fit_weibull(root_comment_times, hardcode_default=False)

	#fit log-normal to all other comment times
	other_comment_times = get_other_comment_times(post)
	if other_comment_times == False:
		return False
	mu, sigma, lognorm_quality = fit_lognormal(other_comment_times, hardcode_default=False)

	#estimate branching factor
	n_b = estimate_branching_factor(len(root_comment_times), len(other_comment_times))

	#any bad params, set all as bad
	if a == False or mu == False:
		return [False, False, False, False, False, False, False]

	#combine all quality measures together into a single one
	quality = (3 * weibull_quality + 2 * lognorm_quality) / 5.0

	#return all parameters together - your job to keep the order straight ;)
	return a, lbd, k, mu, sigma, n_b, quality
#end fit_params

