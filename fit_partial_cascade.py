from fit_weibull import *
from fit_lognormal import *

#given a single cascade and associated comments, and an observed time, fit both the root-comment Weibull and deeper-comment lognormal distributions for the observed comments
#also, estimate the branching factor
#does NOT return param quality, because sometimes it doesn't fit at all - and don't need it for sim from partial
#observed time given in hours, observed comments is just an integer count
#param_guess = set of initial params to inform fit process (generally, inferred from post graph)
def fit_partial_cascade(observed_tree, param_guess=False, verbose = False):

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

    #fitting weibull to root comment times, so get list of those
    root_comment_times = [reply['time'] for reply in observed_tree['replies']]
    if root_comment_times == False:
        vprint("Invalid comment times, skipping this cascade.")
        return False, False
    
    #fitting log-normal to all other comment times, so get list of those
    other_comment_times = get_other_comment_times(observed_tree)

    #how many comments have we observed?
    observed_count = len(root_comment_times) + len(other_comment_times)
    vprint("Fitting partial cascade, observed %dcomments" % observed_count)

    #weibull fit
    #refine the weibull params by calling fit with guess as starting point and #comments (total) as max iterations
    weibull_res = fit_partial_weibull(root_comment_times, param_guess[0:3], observed_count, verbose)       #get back [a, lbd, k], or False if fit fail

    #lognorm fit
    #refine the lognormal params by calling fit with guess as starting point and #comments (total) as max iterations
    lognorm_res = fit_partial_lognormal(other_comment_times, param_guess[3:5], observed_count, verbose)    #get back [mu, sigma], or False if fit fail

    #estimate branching factor
    n_b = partial_estimate_branching_factor(len(root_comment_times), len(other_comment_times), verbose)

    #build return set of params
    params = param_guess[:]        #start with the guess we were given
    #use any fit values that didn't fail
    if weibull_res != False:
        params[0:3] = weibull_res
    if lognorm_res != False:
        params[3:5] = lognorm_res
    if n_b != False:
        params[5] = n_b

    #return all parameters together - your job to keep the order straight ;) - [a, lbd, k, mu, sigma, n_b]
    return params, observed_count
#end fit_partial_cascade


#get list of comment times of all comments not on post (root) in minutes, 
#offset based on the time of the parent comment
#input is a cascade object with comment times shifted relative to root (not parent), and already in minutes
def get_other_comment_times(post):
    other_comment_times = []    #list of comment times

    #init queue to root replies, will process children as nodes are removed from queue
    nodes_to_visit = []
    for comment in post['replies']:  
        nodes_to_visit.append(comment)
    while len(nodes_to_visit) != 0:
        parent = nodes_to_visit.pop(0)    #grab current comment
        parent_time = parent['time']    #grab time for parent comment, use to offset children
        #add all reply times to set of cascade comments, and add child nodes to queue
        for comment in parent['replies']:
            other_comment_times.append(comment['time'] - parent_time)   #offset by parent time
            nodes_to_visit.append(comment)    #add reply to processing queue

    #sort comment times (already in minutes from parent comment)
    other_comment_times = sorted([time for time in other_comment_times])

    #if any < 0, return false
    if not all(reply >= 0 for reply in other_comment_times):
        return False

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