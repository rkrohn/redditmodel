# redditmodel

Code for modelling reddit cascades.

**reddit_model.py** driver for other things: data load an analysis, cascade fitting, tree simulation, etc

**load_model_data.py** load reddit and exogenous data for chosen situation (cyber, crypto, or cve)

**cascade_analysis.py** given loaded posts and comments for a particular domain (cyber, crypto, or cve), reconstruct the post cascade structure (this includes the creation of "dummy" objects for any posts/comments that are referred to by links, but are not present in the data); various cascade analysis tasks (entire set of cascades, not an individual one)

**cascade_manip.py** helper methods for working with cascades - filtering, sorting, etc

**fit_cascade.py** given a single reddit cascade (assumed to be complete), fit a weibull and log-normal to prepare for cascade simulation

**data_utils.py** functions for data manipulation

**file_utils.py** functions for loading and saving of json and pickle files

**plot_utils.py** functions for generating and saving plots

**fit_weibull.py** given a sequence of event times, fit a Weibull pdf to the distribution

**fit_lognormal.py** given a sequence of event times, fit a log-normal to the distribution

**sim_tree.py** given model parameters (Weibull, lognormal, and branching factor), simulate a new comment cascade by generating comment times via a Hawkes process

**ParamGraph.py** class for constructing a post graph, and inferring parameters - work in progress

**fit_all_cascades.py** fit all cascades of a particular domain, broken down by subreddit; saves fit params to pkl files

**predict_cascades.py** given seed posts as json, infer parameters and simulate each cascade; save all results to json

**sim_example.py** small freestanding example for parameter fitting and tree simulation (no inference)


### Directories

code_tests : trying various libraries and techniques
  * hawkes_tree : old code for generating test Hawkes trees and visualizing them (replaced by new sim_tree.py and associated); contains the following files: *MHP.py*: lightweight multivariate Hawkes process simulation code from https://github.com/stmorse/hawkes and https://stmorse.github.io/journal/Hawkes-python.html, slightly modified for our purposes; *Node.py*: lightweight node structure for cascade trees, used by hawkes_tree.py and CascadeTree.py; *CascadeTree.py*: structure to visualize a cascade tree, given the root as a Node object; *hawkes_tree.py*: eventually, should generate a comment cascade (tree) given parameters - work in progress
  * reddit_prediction_code : code from the reddit paper (https://arxiv.org/pdf/1801.10082.pdf) in it's mostly original form, plus some added functionality
  * seismic : original and modified SEISMIC code; Python version of SEISMIC model code (http://snap.stanford.edu/seismic/) based on associated Twitter paper (https://arxiv.org/pdf/1506.02594.pdf); predicts the final retweet count of a particular tweet based on the observed retweet history
   * tensor_testing : trying out some tensor libraries/methods, and the abandoned ParamTensor
  
plots : output plots from various analyses

results : results files from analyses




