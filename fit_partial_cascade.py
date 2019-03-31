from fit_weibull import *
from fit_lognormal import *

#given a single cascade and associated comments, and an observed time, fit both the root-comment Weibull and deeper-comment lognormal distributions for the observed comments
#also, estimate the branching factor
#does NOT return param quality, because sometimes it doesn't fit at all - and don't need it for sim from partial
#observed time given in hours
#param_guess = set of initial params to inform fit process (generally, inferred from post graph)
def fit_partial_cascade(post, cascade, observed_time, param_guess=False, verbose = False):

    #hackery: declare a special print function for verbose output
    global vprint
    if verbose:
        def vprint(*args):
            # Print each argument separately so caller doesn't need to
            # stuff everything to be printed into a single string
            for arg in args:
                print(arg, end='')
            print("")
    else:   
        vprint = lambda *a: None      # do-nothing function

    #fit weibull to root comment times
    root_comment_times = get_root_comment_times(post, cascade)
    if root_comment_times == False:
        vprint("Invalid comment times, skipping this cascade.")
        return False
    #filter to only root comments we've seen
    root_comment_times = [time for time in root_comment_times if time <= observed_time * 60.0]
    
    #fit log-normal to all other comment times
    other_comment_times = get_other_comment_times(post, cascade)
    #filter to only comments we've seen
    other_comment_times = [time for time in other_comment_times if time <= observed_time * 60.0]

    #how many comments have we observed?
    observed_count = len(root_comment_times) + len(other_comment_times)
    vprint("Fitting partial cascade, observed %d of %d comments" % (observed_count, cascade['comment_count_total']))

    #weibull fit
    #refine the weibull params by calling fit with guess as starting point and #comments (total) as max iterations
    weibull_res = fit_partial_weibull(root_comment_times, param_guess[0:3], observed_count, verbose)       #get back [a, lbd, k], or False if fit fail

    #lognorm fit
    #refine the lognormal params by calling fit with guess as starting point and #comments (total) as max iterations
    lognorm_res = fit_partial_lognormal(other_comment_times, param_guess[3:5], observed_count, verbose)    #get back [mu, sigma], or False if fit fail

    #estimate branching factor
    n_b = partial_estimate_branching_factor(len(root_comment_times), len(other_comment_times), verbose)

    #build return set of params
    params = param_guess        #start with the guess we were given
    #use any fit values that didn't fail
    if weibull_res != False:
        params[0:3] = weibull_res
    if lognorm_res != False:
        params[3:5] = lognorm_res
    if n_b != False:
        params[5] = n_b

    #return all parameters together - your job to keep the order straight ;) - [a, lbd, k, mu, sigma, n_b]
    return params
#end fit_partial_cascade


#given a post and its reconstructed cascade, return sorted list of post (root) reply times in minutes (originally stored in seconds),
#taking the post time as 0 and offsetting the comment times accordingly
def get_root_comment_times(post, cascade):
    root_comment_times = []     #build list of reply times

    root_time = post['time']     #get post time in seconds to use as offset

    #loop replies
    for comment in cascade['replies']:
        #if comment occurs less than 60 seconds before the post (assume clock weirdness), set to 0 and move to next
        if comment['time'] - root_time < 0 and root_time - comment['time'] < 60:
            root_comment_times.append(0.0)
        #if comment occurs more than a minute before the post, error and quit
        elif comment['time'] - root_time < 0:
            print("NEGATIVE COMMENT TIME - FAIL!!!!")
            print(post)
            print("root time", root_time)
            reply_times = []
            for ident in post['replies']:
                reply_times.append(comments[ident]['created_utc'])
            print("reply times", sorted(reply_times))
            return False
            
        root_comment_times.append((comment['time'] - root_time) / 60)       #get time between post and comment in minutes

    root_comment_times.sort()   #sort by time
    return root_comment_times
#end get_root_comment_times


#get list of comment times of all comments not on post (root) in minutes, offset based on the post time
def get_other_comment_times(post, cascade):
    other_comment_times = []    #list of comment times

    root_time = post['time']     #get post time in seconds to use as offset

    #get list of comment ids that are on the root, so we can exclude them
    root_comment_ids = [comment['id'] for comment in cascade['replies']]      #replies to root
    #build list of all non-root replies for this post
    other_comment_ids = []
    comments_to_visit = [] + cascade['replies']   #init queue to direct post replies
    while len(comments_to_visit) != 0:
        curr = comments_to_visit.pop(0)    #grab current comment
        if curr['id'] not in root_comment_ids:
            other_comment_times.append((curr['time'] - root_time) / 60)     #add comment time to list
        comments_to_visit.extend(curr['replies'])    #add this comment's replies to queue

    other_comment_times.sort()  #sort by time
    return other_comment_times
#end get_other_comment_times


#estimate branching number n_b for partial cascade: 1 - (root degree / total comments)
#pass in number of root replies and number of comment replies
#if no root replies or comment replies yet, just return False as signal to use inferred param
def partial_estimate_branching_factor(root_replies, comment_replies, display=False):
    #not enough data, return false
    if root_replies == 0 or comment_replies == 0:
        vprint("Not enough comments to estimate branching factor, returning\n")
        return False

    #otherwise, divide and return
    n_b = 1 - root_replies / (root_replies + comment_replies)
    return n_b
#end estimate_branching_factor