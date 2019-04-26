from fit_weibull import *
from fit_lognormal import *
from fit_cascade_gen_model import *

#given a single cascade and associated comments, and an observed time, fit both the root-comment Weibull and deeper-comment lognormal distributions for the observed comments
#also, estimate the branching factor
#does NOT return param quality, because sometimes it doesn't fit at all - and don't need it for sim from partial
#observed time given in hours
#param_guess = set of initial params to inform fit process (generally, inferred from post graph)
def fit_partial_cascade(post, cascade, observed, observing_time, param_guess=False, verbose = False):

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
    root_comment_times = get_root_comment_times(cascade)
    if root_comment_times == False:
        vprint("Invalid comment times, skipping this cascade.")
        return False
    
    #fitting log-normal to all other comment times, so get list of those
    other_comment_times = get_other_comment_times(cascade)

    #filter comment lists to only those we've observed for fit process
    #if observing based on time, filter each list separately
    if observing_time:
        root_comment_times = [time for time in root_comment_times if time <= observed * 60.0]
        other_comment_times = [time for time in other_comment_times if time <= observed * 60.0]
    #if observing only the first N comments, need to consider root and other comments together
    else:
        #get list of ALL comments and root/other identifier
        all_comment_times = [(time, "root") for time in root_comment_times]
        all_comment_times += [(time, "other") for time in other_comment_times]

        #sort this list by time
        all_comment_times = sorted(all_comment_times, key=lambda x: x[0])
        #pull just the observed comments
        observed_comments = all_comment_times[:observed]

        #filter root/other lists to contain only those first n comments
        root_comment_times = root_comment_times[:sum(1 for comment in observed_comments if comment[1] == "root")]
        other_comment_times = other_comment_times[:sum(1 for comment in observed_comments if comment[1] == "other")]

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