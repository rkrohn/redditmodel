# redditmodel

Code for modelling reddit cascades.

**analyze_results.py** combine and convert metric result output files - from separate json to combined csv

**assess_sim.py** stripped-down version of hybrid_model, only simulates cascades for which we have fitted parameters and does not save output; used to examine simulated cascade against actual cascade by printing cascade data and structure

**cascade_analysis.py** given loaded posts and comments for a particular domain (cyber, crypto, or cve), reconstruct the post cascade structure (this includes the creation of "dummy" objects for any posts/comments that are referred to by links, but are not present in the data); various cascade analysis tasks (entire set of cascades, not an individual one); method for fitting all cascades

**cascade_manip.py** helper methods for working with cascades - filtering, sorting, etc

**check_params.py** sandbox for checking/verifying cascade fits

**checking_things.py** looking at param graph sizes, notes on how big I managed to run

**combine_sim_results.py** combine results from multiple post-cascade simulations together into a single file (used when simulation for a test set is divided across multiple processes)

**convert_cascades.py** given cascades in scenario2 output format (baseline2-sample_train_crypto.json and baseline1-sample_train_cryptoreddit.json), convert them to simulated cascade tree format (reddit_CVE_simulated_cascades.json); from separate objects to nested children objects

**correct_sim_output.py** correct existing simulation output files to include the t3_ prefix on all ids to match ground truth

**data_utils.py** functions for data manipulation

**dump_params.py** for a particular subreddit, convert fitted params from .pkl source to node2vec format by printing (pipe output if want a file)

**file_utils.py** functions for loading and saving of json and pickle files

**filter_cyber_posts.py** for cyber domain, filter posts/comments by subreddit and save each to separate file (easily modified for other domains)

**fit_all_cascades.py** fit all cascades of a particular domain, broken down by subreddit; saves fit params to pkl files

**fit_cascade.py** given a single reddit cascade (assumed to be complete), fit a weibull and log-normal to prepare for cascade simulation

**fit_lognormal.py** given a sequence of event times, fit a log-normal to the distribution

**fit_weibull.py** given a sequence of event times, fit a Weibull pdf to the distribution

**functions_hybrid_model.py** collection of methods used by hybrid_model.py in simulation process

**functions_prebuild_model.py** collection of methods used by prebuild_model.py in model setup process

**hybrid_model.py** given a set of seed posts, predict them all and create submission output file; requires two command line args: input filename of seed posts, and output filename for simulation json results; offloads node2vec to c++, because speed

**load_model_data.py** load reddit and exogenous data for chosen situation (cyber, crypto, or cve)

**ParamGraph.py** class for constructing a bipartite-ish-project of a user-post graph, and inferring parameters; start with a bipartite graph of users and words, with edges labelled with parameters; perform a bipartite projection (ish) to get a graph of parameter nodes; a new post introduces new nodes to this graph, but with undefined parameters; use node2vec to get these missing parameters; ABANDONED in favor of PostParamGraph or hybrid_model.py

**plot_utils.py** functions for generating and saving plots

**post_predict_cascades.py** given a set of seed posts, predict them all using PostParamGraph and python node2vec, and create submission output file; requires three command line args: domain, input filename of seed posts, and output filename for simulation json results

**PostParamGraph.py** class for constructing a post graph for parameter inference; one node per post, connect posts by the same user with an edge of weight = 1, also connect posts with words in common by a single edge of weight = (# shared) / (# in shortest title); a new post introduces a new node to this graph, but with undefined parameters; use Python node2vec to get these missing parameters

**prebuild_model.py** create and save all the necessary files for the new c++/python hybrid model; save everything to model_files for easy server loading later

**prebuild_model_cve.py** create hybrid model files for cve domain, since it's not handled on a subreddit basis

**prebuild_model_pcmasterrace.py** prebuild for just pcmasterrace, chunk up cascades for parallel fitting

**predict_cascades.py** given seed posts as json, infer parameters and simulate each cascade; save all results to json; uses the ABANDONED ParamGraph; replaced by hybrid_model.py

**reddit_model.py** driver for other things: data load an analysis, cascade fitting, tree simulation, etc

**sim_example.py** small freestanding example for parameter fitting and tree simulation (no inference)

**sim_tree.py** given model parameters (Weibull, lognormal, and branching factor), simulate a new comment cascade by generating comment times via a Hawkes process


### Directories

code_tests : trying various libraries and techniques
  * hawkes_tree : old code for generating test Hawkes trees and visualizing them (replaced by new sim_tree.py and associated); contains the following files: *MHP.py*: lightweight multivariate Hawkes process simulation code from https://github.com/stmorse/hawkes and https://stmorse.github.io/journal/Hawkes-python.html, slightly modified for our purposes; *Node.py*: lightweight node structure for cascade trees, used by hawkes_tree.py and CascadeTree.py; *CascadeTree.py*: structure to visualize a cascade tree, given the root as a Node object; *hawkes_tree.py*: eventually, should generate a comment cascade (tree) given parameters - work in progress
  * reddit_prediction_code : code from the reddit paper (https://arxiv.org/pdf/1801.10082.pdf) in it's mostly original form, plus some added functionality
  * seismic : original and modified SEISMIC code; Python version of SEISMIC model code (http://snap.stanford.edu/seismic/) based on associated Twitter paper (https://arxiv.org/pdf/1506.02594.pdf); predicts the final retweet count of a particular tweet based on the observed retweet history
   * tensor_testing : trying out some tensor libraries/methods, and the abandoned ParamTensor
  
plots : output plots from various analyses

results : results files from analyses




