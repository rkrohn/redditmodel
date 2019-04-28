from fit_weibull import *
from fit_lognormal import *
from fit_cascade_gen_model import *
import functions_gen_cascade_model

#given a single cascade and associated comments, and an observed time, fit both the root-comment Weibull and deeper-comment lognormal distributions for the observed comments
#also, estimate the branching factor
#does NOT return param quality, because sometimes it doesn't fit at all - and don't need it for sim from partial
#observed time given in hours, observed comments is just an integer count
#param_guess = set of initial params to inform fit process (generally, inferred from post graph)
def fit_partial_cascade(cascade, observed, observing_time, param_guess=False, verbose = False):

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

    #filter cascade to observed time/comments, but leave timestamps unchanged (UTC seconds)
    if observing_time:
    	filtered_cascade, observed_count = functions_gen_cascade_model.filter_comment_tree(cascade, observed*60, convert_times=False)
    else:
    	filtered_cascade, observed_count, observing_time = functions_gen_cascade_model.filter_comment_tree_by_num_comments(cascade, observed, convert_times=False)
    #use this filtered cascade to fit from

    #fitting weibull to root comment times, so get list of those
    root_comment_times = get_root_comment_times(filtered_cascade)
    if root_comment_times == False:
        vprint("Invalid comment times, skipping this cascade.")
        return False
    
    #fitting log-normal to all other comment times, so get list of those
    other_comment_times = get_other_comment_times(filtered_cascade)

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