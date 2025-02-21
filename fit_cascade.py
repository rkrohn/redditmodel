#given a single reddit cascade (assumed to be complete), fit a weibull and log-normal to prepare for cascade simulation
#fit root comments to a Weibull distribution
#fit deeper comments to a log-normal distribution
#see fit_weibull.py and fit_lognormal.py for more details on fitting process


from fit_weibull import *
from fit_lognormal import *
from collections import defaultdict
import random


#HARDCODED PARAMS - only used when fit/estimation fails
#see fit_lognormal and fit_weibull for those hardcoded values

DEFAULT_BRANCHING = 0.05        #default branching factor n_b if post has no comments, or post comments have no replies
                                #0.05 should allow for rare comments 
DEFAULT_PERTURB = 0.15      #max delta for random perturbation
DEFAULT_BRANCHING_QUALITY_HARDCODE = 0.4     #default quality for hardcoded branching factor
DEFAULT_BRANCHING_QUALITY_DIVIDE = 0.95      #branching quality for division estimation


#given a post, get creation time of the post in seconds (tiny helper function)
def get_root_time(post):
    return post['created_utc']
#end get_root_time


#given a post and all comments, return sorted list of post (root) reply times in minutes (originally stored in seconds),
#taking the post time as 0 and offsetting the comment times accordingly
def get_root_comment_times(post, comments):
    root_comment_times = []     #build list of reply times

    root_time = get_root_time(post)     #get post time in seconds to use as offset

    #loop replies
    for comment_id in post['replies']:
        #if comment occurs less than 60 seconds before the post (assume clock weirdness), set to 0 and move to next
        if comments[comment_id]['created_utc'] - root_time < 0 and root_time - comments[comment_id]['created_utc'] < 60:
            root_comment_times.append(0.0)
        #if comment occurs more than a minute before the post, error and quit
        elif comments[comment_id]['created_utc'] - root_time < 0:
            print("NEGATIVE COMMENT TIME - FAIL!!!!")
            print(post)
            print("root time", root_time)
            reply_times = []
            for ident in post['replies']:
                reply_times.append(comments[ident]['created_utc'])
            print("reply times", sorted(reply_times))
            return False
            
        root_comment_times.append((comments[comment_id]['created_utc'] - root_time) / 60)       #get time between post and comment in minutes

    root_comment_times.sort()   #sort by time
    return root_comment_times
#end get_root_comment_times


#get list of comment times of all comments not on post (root) in minutes, offset based on the post time
def get_other_comment_times(post, comments):
    other_comment_times = []    #list of comment times

    root_time = get_root_time(post)     #get post time in seconds to use as offset

    #get list of comment ids that are not on the root
    root_comment_ids = post['replies']      #replies to root
    #build list of all non-root replies for this post
    other_comment_ids = []
    nodes_to_visit = [] + post['replies']   #init queue to direct post replies
    while len(nodes_to_visit) != 0:
        curr = nodes_to_visit.pop(0)    #grab current comment id
        if curr not in root_comment_ids:
            other_comment_ids.append(curr)           #add this comment to set of cascade comments
        nodes_to_visit.extend(comments[curr]['replies'])    #add this comment's replies to queue

    #loop comment ids of non-root replies, extract time
    for comment_id in other_comment_ids:
        other_comment_times.append((comments[comment_id]['created_utc'] - root_time) / 60)  #offset time in minutes

    other_comment_times.sort()	#sort by time
    return other_comment_times
#end get_other_comment_times


#get size of a tree as total number of comments
def get_cascade_size(post, comments):
    return len(cascade_manip.get_cascade_comment_ids(post, comments))
#end get_cascade_size


#given a tree and root, count the number of nodes on each level
def count_nodes_per_level(post, comments):

    depth_counts = defaultdict(int)     #dictionary for depth counts
    depth_counts[0] = 1

    nodes_to_visit = [] + post['replies']    #init queue to direct post replies
    depth_queue = [1] * len(nodes_to_visit)  #and a parallel depth queue
    while len(nodes_to_visit) != 0:     #BFS
        curr = nodes_to_visit.pop(0)    #grab current comment id
        depth = depth_queue.pop(0)

        depth_counts[depth] += 1

        nodes_to_visit.extend(comments[curr]['replies'])    #add this comment's replies to queue
        depth_queue.extend([depth+1] * len(comments[curr]['replies']))

    return depth_counts
#end count_nodes_per_level


#estimate branching number n_b: 1 - (root degree / total comments)
#pass in number of root replies and number of comment replies
def estimate_branching_factor(root_replies, comment_replies):
    #hardcode or estimate, depending on number of comments
    if root_replies + comment_replies == 0:
        #hardcode (+perturb) to maybe allow for rare comment replies
        n_b = DEFAULT_BRANCHING + random.uniform(-1 * DEFAULT_PERTURB * DEFAULT_BRANCHING, DEFAULT_PERTURB * DEFAULT_BRANCHING)     
        quality = DEFAULT_BRANCHING_QUALITY_HARDCODE
    elif comment_replies > root_replies * 1.25:
        n_b = 1 - (root_replies / (root_replies + comment_replies)) + (comment_replies / root_replies / 3.5)
        #cap this one at 3
        if n_b > 3:
            n_b = 3.0 - random.uniform(0, 0.25)
        quality = DEFAULT_BRANCHING_QUALITY_DIVIDE
    else:
        n_b = 1 - (root_replies / (root_replies + comment_replies))
        quality = DEFAULT_BRANCHING_QUALITY_DIVIDE
    #if estimate is 0 (post comments have no replies), hardcode
    if n_b == 0:
        #hardcode (+perturb) to maybe allow for rare comment replies
        n_b = DEFAULT_BRANCHING + random.uniform(-1 * DEFAULT_PERTURB * DEFAULT_BRANCHING, DEFAULT_PERTURB * DEFAULT_BRANCHING)
        quality = DEFAULT_BRANCHING_QUALITY_HARDCODE      
    return n_b, quality
#end estimate_branching_factor


#given a single cascade and associated comments, fit both the root-comment Weibull and deeper-comment lognormal distributions
#also, estimate the branching factor
def fit_cascade_model(post, comments, display = False):

    #print by-level breakdown of this cascade
    if display:
        depth_counts = count_nodes_per_level(post, comments)
        print("input cascade nodes per level:")
        for depth, count in depth_counts.items():
            print(depth, ":", count)
        print("")

    #fit weibull to root comment times
    root_comment_times = get_root_comment_times(post, comments)
    if root_comment_times == False:
        print("Invalid comment times, skipping this cascade.")
        return False
    if display: 
        print("root comments", root_comment_times)
    a, lbd, k, weibull_quality = fit_weibull(root_comment_times, display)

    #fit log-normal to all other comment times
    other_comment_times = get_other_comment_times(post, comments)
    if display:
        print("other comments", other_comment_times)
    mu, sigma, lognorm_quality = fit_lognormal(other_comment_times, display)

    #estimate branching factor
    n_b, branching_quality = estimate_branching_factor(len(root_comment_times), len(other_comment_times))
    if display:
        print("branching factor :", n_b, "(quality", str(branching_quality) + ")\n")

    #combine all quality measures together into a single one
    quality = (3 * weibull_quality + 2 * lognorm_quality + branching_quality) / 6

    #return all parameters together - your job to keep the order straight ;)
    return a, lbd, k, mu, sigma, n_b, quality
#end fit_cascade_model

