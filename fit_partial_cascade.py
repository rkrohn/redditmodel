from fit_weibull import *
from fit_lognormal import *
from fit_cascade import *

#given a single cascade and associated comments, and an observed time, fit both the root-comment Weibull and deeper-comment lognormal distributions for the observed comments
#also, estimate the branching factor
#observed time given in hours
def fit_partial_cascade(post, comments, observed_time, display = False):

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
    a, lbd, k, weibull_quality = fit_weibull(root_comment_times, display)

    #fit log-normal to all other comment times
    other_comment_times = get_other_comment_times(post, comments)
    #filter to only comments we've seen
    other_comment_times = [time for time in other_comment_times if time <= observed_time * 60.0]
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
#end fit_partial_cascade