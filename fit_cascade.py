#given a single reddit cascade (assumed to be complete), fit a weibull and log-normal to prepare for cascade simulation
#fit root comments to a Weibull distribution
#fit deeper comments to a log-normal distribution
#see fit_weibull.py and fit_lognormal.py for more details on fitting process


import cascade_manip
from fit_weibull import *
from fit_lognormal import *
from collections import defaultdict


#HARDCODED PARAMS - only used when fit/estimation fails
#see fit_lognormal and fit_weibull for those hardcoded values

DEFAULT_BRANCHING = 0.05        #default branching factor n_b if post has no comments, orpost comments have no replies
                                #0.05 should allow for rare comments 


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
        root_comment_times.append((comments[comment_id]['created_utc'] - root_time) / 60)  #get time between post and comment in minutes

    root_comment_times.sort()   #sort by time
    return root_comment_times
#end get_root_comment_times


#get list of comment times of all comments not on post (root) in minutes, offset based on the post time
def get_other_comment_times(post, comments):
    other_comment_times = []    #list of comment times

    root_time = get_root_time(post)     #get post time in seconds to use as offset

    #get list of comment ids that are not on the root
    all_comment_ids = cascade_manip.get_cascade_comment_ids(post, comments)     #all post comments in a set
    root_comment_ids = post['replies']      #replies to root
    other_comment_ids = [comment_id for comment_id in all_comment_ids if comment_id not in root_comment_ids]

    #loop comment ids of non-root replies
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
def estimate_branching_factor(post, comments):
    num_comments = get_cascade_size(post, comments)
    #hardcode or estimate, depending on number of comments
    if num_comments == 0:
        n_b = DEFAULT_BRANCHING     #hardcode to maybe allow for rare comment replies
    else:
        n_b = 1 - (len(post['replies']) / num_comments)
    #if estimate is 0 (post comments have no replies), hardcode
    if n_b == 0:
        n_b = DEFAULT_BRANCHING       #hardcode to maybe allow for rare comment replies
    return n_b
#end estimate_branching_factor


#given a single cascade and associated comments, fit both the root-comment Weibull and deeper-comment lognormal distributions
#also, estimate the branching factor
def fit_cascade_model(post, comments, display = False):

    #print by-level breakdown of this cascade
    if display:
        depth_counts = count_nodes_per_level(post, comments)
        print("cascade nodes per level:")
        for depth, count in depth_counts.items():
            print(depth, ":", count)
        print("")

    #fit weibull to root comment times
    root_comment_times = get_root_comment_times(post, comments)   
    a, lbd, k = fit_weibull(root_comment_times)

    #fit log-normal to all other comment times
    other_comment_times = get_other_comment_times(post, comments)
    mu, sigma = fit_lognormal(other_comment_times)

    #estimate branching factor
    n_b = estimate_branching_factor(post, comments)
    if display:
        print("branching factor :", n_b, "\n")

    #return all parameters together - your job to keep the order straight ;)
    return a, lbd, k, mu, sigma, n_b
#end fit_cascade_model
