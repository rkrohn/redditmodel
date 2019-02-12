from fit_weibull import *
from fit_lognormal import *
from fit_cascade import *

#given a single cascade and associated comments, and an observed time, fit both the root-comment Weibull and deeper-comment lognormal distributions for the observed comments
#also, estimate the branching factor
#does NOT return param quality, because sometimes it doesn't fit at all - and don't need it for sim from partial
#observed time given in hours
#param_guess = set of initial params to inform fit process (generally, inferred from post graph)
def fit_partial_cascade(post, comments, observed_time, param_guess=False, display = False):

    #print by-level breakdown of this cascade
    if display:
        depth_counts = count_nodes_per_level(post, comments)
        print("input cascade nodes per level:")
        for depth, count in depth_counts.items():
            print(depth, ":", count)
        print("")

    #fit weibull to root comment times
    root_comment_times = get_root_comment_times(post, comments)
    #filter to only root comments we've seen
    root_comment_times = [time for time in root_comment_times if time <= observed_time * 60.0]
    if root_comment_times == False:
        print("Invalid comment times, skipping this cascade.")
        return False
    if display: 
        print("root comments", root_comment_times)
    weibull_res = fit_partial_weibull(root_comment_times, param_guess[0:3], display)       #get back [a, lbd, k], or False if fit fail

    #fit log-normal to all other comment times
    other_comment_times = get_other_comment_times(post, comments)
    #filter to only comments we've seen
    other_comment_times = [time for time in other_comment_times if time <= observed_time * 60.0]
    if display:
        print("other comments", other_comment_times)
    lognorm_res = fit_partial_lognormal(other_comment_times, param_guess[3:5], display)    #get back [mu, sigma], or False if fit fail

    #estimate branching factor
    n_b = partial_estimate_branching_factor(len(root_comment_times), len(other_comment_times), display)
    if display:
        print("branching factor :", n_b, "\n")

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

#estimate branching number n_b for partial cascade: 1 - (root degree / total comments)
#pass in number of root replies and number of comment replies
#if no root replies or comment replies yet, just return False as signal to use inferred param
def partial_estimate_branching_factor(root_replies, comment_replies, display=False):

    #not enough data, return false
    if root_replies == 0 or comment_replies == 0:
        if display:
            print("Not enough comments to estimate branching factor, returning\n")
        return False

    #otherwise, divide and return
    n_b = 1 - root_replies / (root_replies + comment_replies)
    return n_b
#end estimate_branching_factor