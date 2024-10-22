#!/usr/bin/env python
# coding: utf-8


#simulate a reddit cascade
#root comments based on a Weibull distribution
#deeper comments follow a log-normal distribution
#see fit_weibull.py and fit_lognormal.py for more details on fitting process

#based on the code from https://arxiv.org/pdf/1801.10082.pdf, with modifications


from fit_weibull import *
from fit_lognormal import *
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import warnings
warnings.filterwarnings("ignore")   #supress some matplotlib warnings
import itertools
from collections import defaultdict


#HARDCODED VALUES
INF_INTENSITY = 2.25    #value used for intensity if function returns infinity (occurs when t = 0 and k < 1)


#given current time t, max inter-event arrival time T, max number of events N_max, 
#and acceptance flag (True if t was accepted as an event time), and list of generated
#event times, determine if simulation has reached stopping condition
#returns True if simulation should stop, False otherwise
def stop_sim(t, T, N_max, accept, new_event_times, display = False):
    #quit if generated too many events
    if len(new_event_times) > N_max:
        if display:
            print("Stopping simulation: max total events reached")
        return True

    #quit if reached inter-arrival threshold
    #three conditions cause this to trigger: 
    #   no events and current time too high
    #   t was added as an event and occurred >T units after the previous event
    #   t was not accepted, and is >T units after the previous event
    if (len(new_event_times) == 0 and t > T) or (accept and len(new_event_times) > 1 and t - new_event_times[-2] > T) or (accept == False and len(new_event_times) > 0 and t - new_event_times[-1] > T):
        if display:
            print("Stopping simulation: max inter-event arrival time reached")
        return True

    #if no conditions tripped, not time to stop, return False
    return False
#end stop_sim


#evaluate weibull at time t, return result - with special check for infinite values
#all paramaters must be > 0
def weib_intensity(t, a, lbd, k):
    f = weib_func(t, a, lbd, k)     #get value of weibull function at time t
    #special-case to prevent infinte returns
    if math.isinf(f):
        f = INF_INTENSITY
    return f


#given weibull parameters (see fit_weibull.py for param details), simulate new event times following this distribution
#
#parameters:
#   a           scalar multiplier (directly from the paper)
#   lambda      scale parameter (b in the source paper)
#   k           shape parameter (alpha in the source paper)
#
#essentially a homogenous poisson process, but with a variable base intensity mu defined by the weibull kernel
#bounded by inter-event time of 1800 minutes = ~30 hours or max number of events 2000 
def generate_weibull_times(params, start_time = 0, T = 1800, N_max = 2000):   #params = [a, lbd, k]
    (a, lbd, k) = params    #unpack parameters

    new_event_times = []    #build list of new event times
    t = start_time          #init t

    #set base intensity for current time
    #intensity = lambda* = thinning rate = upper bound of mu(t) for current time interval = rate at which to generate top-level replies

    #shape param > 1, bell-shape curve
    if k > 1:
        if t > lbd * ((k-1) / k)**(1/k):
            intensity = weib_intensity(t, a, lbd, k)
        else:
            intensity = (a * k/lbd) * ((k-1) / k)**(1 - 1/k) * np.exp(-((k-1) / k))
    #shape param between 0 and 1, decreasing curve
    elif k > 0:
        intensity = weib_intensity(t,a,lbd,k)		#call method to get weibull at time t


    #thinning algorithm to generate times - see page 10 of Hawkes chapter for more details
    while True:
        accept = False      #boolean flag used in stoppig condition logic

        e = np.random.uniform(low=0, high=1)		#sample between 0 and 1 uniform distribution
        t += -np.log(e)/intensity 					#compute waiting time = -ln(e)/lambda, add to current time t

        #accept or reject t as event time?
        s = np.random.uniform(low=0, high=1)		#draw another value
        #if random value less than ratio of true event rate to thinning rate, accept
        if s < weib_intensity(t,a,lbd, k) / intensity:
            new_event_times.append(t)		#save this event time
            accept = True

            #update thinning rate if necessary for next event generation
            if (k > 1 and t > lbd*((k-1)/k)**(1/k)) or (k > 0 and k < 1):
                intensity = weib_intensity(t,a,lbd, k)

        #quit if reached inter-arrival threshold
        if stop_sim(t, T, N_max, accept, new_event_times):
            break

    return new_event_times
#end generate_weibull_times


#evaluate log-normal distribution at time t, including branching factor scaling to get intensity
#requires fitted params mu, sigma, and n_b
def lognorm_intensity(t, mu, sigma, n_b):
    return n_b * lognorm_func(t, mu, sigma)
#end lognorm_intensity


#given log-normal parameters (see fit_lognormal.py for param details), simulate new event times following this distribution
#
#parameters:
#   mu      mean
#   sigma   variance
#
#essentially a homogenous poisson process, but with a variable base intensity mu defined by the log-normal distribution
#bounded by inter-event time of 1800 minutes = ~30 hours or max number of events 2000
#also utilizes the estimated branching factor 
def generate_lognorm_times(params, n_b, start_time = 0, T = 1800, N_max = 2000): # Log-normal kernel
    (mu, sigma) = params    #unpack the parameters

    new_event_times = []    #build list of new event times
    t = start_time          #init time

    #set base intensity for current time
    #get intensity = lambda* = thinning rate = event rate = upper bound of intensity
    if t > np.exp(mu - (sigma**2)):  
        intensity = lognorm_intensity(t, mu, sigma, n_b)
    else:
        intensity = n_b * (np.exp(sigma**2 - mu) / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(sigma**2) / 2)

    if intensity > 500:
        intensity = 500

    #thinning process to generate events - see page 10 of Hawkes chapter for more details
    while True:
        accept = False          #boolean flag used in stopping logic

        e = np.random.uniform(low=0, high=1)	#sample from uniform distribution
        t += -np.log(e)/intensity 				#compute inter-event time and add to current time t

        #accept or reject current time as event time?
        s = np.random.uniform(low=0, high=1)	#sample again
        #if sample less than ratio of true event rate to thinning rate, accept
        if s < lognorm_intensity(t, mu, sigma, n_b) / intensity:
            new_event_times.append(t)		#accept this event time
            accept = True
            #update lambda* for next generation cycle, if required
            if t > np.exp(mu - (sigma**2)):
                intensity = lognorm_intensity(t, mu, sigma, n_b)
                if intensity > 500:
                    intensity = 500

        #quit if reached inter-arrival threshold
        if stop_sim(t, T, N_max, accept, new_event_times):
            break

    return new_event_times
#end generate_lognorm_times


#overall simulation function, work from root down, generating as we go
#
#if time_observed and observed_comments are false, simulate a new tree from just the root
#otherwise, simulate from the partially observed tree
#
#returns a tree stored using a dictionary structure (for easy field additions later) and a sorted list of all comment times - note comment times are in MINUTES from original post, not seconds
#current fields: time, id (arbitrarily assigned), replies (list)
#generate root comments based on fitted weibull, comment replies based on fitted log-normal
#requires fitted weibull and log-normal distribution to define hawkes process
#model params = [a, lbd, k, mu, sigma, n_b] (first 3 weibull, next 2 lognorm, last branching factor)
def simulate_comment_tree(model_params, time_observed = False, observed_tree = False, display = False):
    weibull_params, lognorm_params, n_b = unpack_params(model_params)   #unfold parameters

    #what is the root of the tree? observed tree if we have it, otherwise new root
    #simulate from partial tree, including any observed comments
    if time_observed != False:
        root = observed_tree
    #simulate from empty starting tree (just the root, no observed comments)
    else:
        root = {'time' : 0, 'id' : 0, 'replies' : list()}       #root at time 0, id is 0
        time_observed = 0       #set time_observed = 0 for later checks against comment times

    #get list of existing comments to root (root replies)
    #this is a list of comment objects, not times
    observed_root_replies = [comment for comment in root['replies']]
    #print(len(observed_root_replies), 'observed root replies', [reply['time'] for reply in observed_root_replies])

    #get list of existing comment replies (ie, not root replies)
    observed_comment_replies = []
    #init queue to second-level replies (ie, replies to root replies)
    nodes_to_visit = []
    for comment in root['replies']:  
        nodes_to_visit.extend(comment['replies']) 
    while len(nodes_to_visit) != 0:
        curr = nodes_to_visit.pop(0)    #grab current comment
        observed_comment_replies.append(curr)           #add this comment to comment list
        nodes_to_visit.extend(curr['replies']) #add this comment's replies to queue
    #print(len(observed_comment_replies), 'observed comment replies', [reply['time'] for reply in observed_comment_replies])

    #init list of all comment times to observed comments
    all_replies = [comment['time'] for comment in (observed_root_replies + observed_comment_replies)]  

    node_id = itertools.count(start = 1)      #iterator counter for new node ids, start at 1

    #generate further replies for existing comment replies
    for comment in observed_comment_replies:
        comment_time = comment['time']      #time of this comment

        #generate comment replies for this comment, where start time is how long we've observed this comment (ie, no new replies within observed window)
        reply_times = generate_lognorm_times(lognorm_params, n_b, start_time = time_observed-comment_time)
        #add the current comment (parent) time to the generated times
        reply_times = [reply_time + comment_time for reply_time in reply_times]
        #convert to list of reply objects
        reply_comments = [{'time' : reply_time, 'replies' : list(), 'id' : next(node_id)} for reply_time in reply_times]

        #add child objects to parent for all these generated replies
        comment['replies'].extend(reply_comments)  

        needs_replies = []      				#clear processing list
        needs_replies.extend(reply_comments)    #add all new replies to list to be processed

        #add all new comment times to overall time list
        all_replies.extend([reply_time for reply_time in reply_times])    

        #we have generated one level of replies to the observed comment
        #now, generate replies to these new replies all the way down (hence the while)
        while len(needs_replies) != 0:
            curr = needs_replies.pop(0)  #get current object, delete from list
            curr_time = curr['time']        #time of this comment

            #generate replies to this generated comment
            #start_time = 0 here, since we want all possible replies to this generated comment 
            #(since generated comment will always occur after the observed window)
            reply_times = generate_lognorm_times(lognorm_params, n_b, start_time = 0)
            #shift by parent time to get root-relative time
            reply_times = [reply_time + curr_time for reply_time in reply_times]
            #convert to list of reply objects
            reply_comments = [{'time' : reply_time, 'replies' : list(), 'id' : next(node_id)} for reply_time in reply_times]

            #add all new comment times to overall time list
            all_replies.extend([reply_time for reply_time in reply_times])  

            #add child objects for all these generated replies
            curr['replies'].extend(reply_comments)  
            needs_replies.extend(reply_comments)       #add all new replies to list to be processed
    #done generating children for observed comment replies

    #generate NEW top-level comments (root replies)
    new_root_comment_times = generate_weibull_times(weibull_params, start_time = time_observed)

    #add all new replies to tree (node objects)
    root['replies'].extend([{'time' : child_time, 'replies' : list(), 'id' : next(node_id)} for child_time in new_root_comment_times])

    #add these new root comments to overall time list
    all_replies.extend([reply_time for reply_time in new_root_comment_times])

    #process each root comment (new or observed), generating children all the way down
    for comment in root['replies']:
        comment_time = comment['time']      #time of root reply to get children for

        #generate comment replies to this comment (second level of tree)

        #if generating replies for an observed root comment, generate children with start time equal to how long we've observed (observed time - comment time, so no new comments within observed window)
        if comment_time <= time_observed:
            reply_times = generate_lognorm_times(lognorm_params, n_b, start_time = time_observed - comment_time)
        #if generating replies for a simulated root comment, generate children with start time of 0 to get all possible children
        else:
            reply_times = generate_lognorm_times(lognorm_params, n_b, start_time = 0)

        #shift replies by parent comment time to get root-relative time
        reply_times = [reply_time + comment_time for reply_time in reply_times]
        #convert to list of reply objects
        reply_comments = [{'time' : reply_time, 'replies' : list(), 'id' : next(node_id)} for reply_time in reply_times]

        #add child objects for all generated replies to parent object as children
        comment['replies'].extend(reply_comments)  

        needs_replies = []      					#clear processing list
        needs_replies.extend(reply_comments)        #add all new replies to list to be processed

        all_replies.extend([reply_time for reply_time in reply_times])  	#add to overall time list

        #we have generated one level of replies to the observed root comment
        #now, generate replies to these new replies all the way down (hence the while)
        while len(needs_replies) != 0:
            curr = needs_replies.pop(0)  	#get current object, delete from list
            curr_time = curr['time']        #time of this comment

            #generate replies to this generated comment
            #start_time = 0 here, since we want all possible replies to this generated comment
            reply_times = generate_lognorm_times(lognorm_params, n_b, start_time = 0)
            #shift by parent time to get root-relative time
            reply_times = [reply_time + curr_time for reply_time in reply_times]
            #convert to list of reply objects
            reply_comments = [{'time' : reply_time, 'replies' : list(), 'id' : next(node_id)} for reply_time in reply_times]

            #add all new comment times to overall time list
            all_replies.extend([reply_time for reply_time in reply_times])

            #add child objects for all these generated replies
            curr['replies'].extend(reply_comments)  
            needs_replies.extend(reply_comments)       #add all new replies to list to be processed

            #try to catch the infinite sim (might need tuning)
            if len(needs_replies) > 500:
                return False, False
    #finished processing root comments

    if display:
        print_tree(root)
        print("Simulated cascade has", len(root['replies']), "root replies and", len(all_replies), "total comments")

    return root, sorted(all_replies)  #return root of tree AND list of sorted reply times
#end simulate_comment_tree


#helper function to unpack parameters as given in tensor to separate items
def unpack_params(params):
    weibull_params = params[:3]
    lognorm_params = params[3:5]
    n_b = params[5]

    return weibull_params, lognorm_params, n_b
#end unpack_params


#given observed and simulated comment times, plot comparison histogram
def plot_two_comparison(sim, actual, filename):
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(8, 6))

    max_time = max(sim + actual) + 5    #last time, +5 for plot buffer

    #plot side-by-side histogram of entire simulated tree against actual tree
    bins = np.linspace(0, max_time, 30)
    ax1.hist([actual, sim], bins, stacked=False, normed=False, color=["blue", "green"], label=["actual", "simulated"])  
    ax1.set_xlabel('time (minutes)')
    ax1.set_ylabel('comment frequency')
    ax1.legend()
    plt.savefig(filename)
#end plot_two_comparison

#given observed, simulated, and simulated from inferred comment times, plot comparison histogram
def plot_three_comparison(sim, sim_infer, actual, filename):
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(8, 6))

    max_time = max(sim + actual + sim_infer) + 5    #last time, +5 for plot buffer

    #plot side-by-side histogram of entire simulated tree against actual tree
    bins = np.linspace(0, max_time, 30)
    ax1.hist([actual, sim, sim_infer], bins, stacked=False, normed=False, color=["blue", "green", "purple"], label=["actual", "simulated", "inferred+sim"])  
    ax1.set_xlabel('time (minutes)')
    ax1.set_ylabel('comment frequency')
    ax1.legend()
    plt.savefig(filename)
#end plot_three_comparison


#given observed and simulated root comment times, and fitted Weibull params, plot all the things
def plot_root_comments(sim, actual, filename, params = None):
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(8, 6))

    max_time = max(sim + actual) + 5    #last time, +5 for plot buffer

    #first: plot side-by-side histogram of entire simulated tree against actual tree
    
    #plot histogram for simulated tree, stacking observed and simulated together
    bins = np.linspace(0, max_time, 30)
    ax1.hist([actual, sim], bins, stacked=False, normed=False, color=["blue", "green"], label=["actual", "simulated"])  
    ax1.set_xlabel('time (minutes)')
    ax1.set_ylabel('comment frequency')

    #plot fitted weibull curve
    if params != None:
        ax2 = ax1.twinx()
        x = np.arange(0, max(sim), 1)
        y = [weib_func(t, params[0], params[1], params[2]) for t in x]
        ax2.plot(x, y, 'r-', label="Weibull fit")
        ax2.set_ylabel('Weibull fit', color='r')

    ax1.legend()
    plt.savefig(filename)
#end plot_root_comments


#given the root of a simulated tree (dictionary structure), visualize the cascade tree
def viz_tree(root, filename):
    #build networkx graph to use for viz
    G = nx.DiGraph()

    #get list of edges as returned by BFS
    edges = []
    nodes = [(root, None)]  #queue contains tree node and parent
    while len(nodes) != 0:
        curr, parent = nodes.pop(0)
        #if node has parent, add edge
        if parent != None:
            edges.append((parent, curr['id']))
        nodes.extend([(child, curr['id']) for child in curr['children']])

    G.add_edges_from(edges)     #add all edges to graph

    #plot the tree!
    plt.clf()
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=False, node_size=15)
    plt.savefig(filename)
#end viz_tree


#uses DFS to print the tree structure (just the comment times, but easily modified)
def print_tree(root, level=0):
    visited = set()    #set of visited nodes
    stack = [(root, 0)]     #node processing stack

    while len(stack) != 0:
        curr, level = stack.pop()  #get last node added to stack
        print("    " * level + "%.3f" % curr['time'] if level != 0 else "root = 0")   #print this comment time
        if curr['id'] not in visited:
            visited.add(curr['id'])
            #append replies in reverse time order so final output is sorted
            stack.extend([(child, level+1) for child in curr['replies']][::-1])    
#end print_tree


#given a simulated tree root, count the number of nodes on each level
def sim_count_nodes_per_level(root):

    depth_counts = defaultdict(int)     #dictionary for depth counts
    depth_counts[0] = 1

    nodes_to_visit = [] + root['replies']    #init queue to direct post replies
    depth_queue = [1] * len(nodes_to_visit)  #and a parallel depth queue
    while len(nodes_to_visit) != 0:     #BFS
        curr = nodes_to_visit.pop(0)    #grab current comment id
        depth = depth_queue.pop(0)

        depth_counts[depth] += 1

        nodes_to_visit.extend(curr['replies'])    #add this comment's replies to queue
        depth_queue.extend([depth+1] * len(curr['replies']))

    return depth_counts
#end count_nodes_per_level