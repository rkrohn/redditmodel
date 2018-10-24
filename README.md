# redditmodel

Code for modelling reddit cascades.

**reddit_model.py** driver for data loading and analysis

**load_model_data.py** load reddit and exogenous data for chosen situation (cyber, crypto, or cve)

**cascade_analysis.py** given loaded posts and comments for a particular domain (cyber, crypto, or cve), reconstruct the post cascade structure (this includes the creation of "dummy" objects for any posts/comments that are referred to by links, but are not present in the data); various cascade analysis tasks (entire set of cascades, not an individual one)

**cascade_manip.py** helper methods for working with cascades - filtering, sorting, etc

**fit_cascade.py** given a single reddit cascade (assumed to be complete), fit a weibull and log-normal to prepare for cascade simulation

**data_utils.py** functions for data manipulation

**file_utils.py** functions for loading and saving of json and pickle files

**plot_utils.py** functions for generating and saving plots

**tweet.csv** sample data for testing SEISMIC code, gives one retweet cascade

**MHP.py** lightweight multivariate Hawkes process simulation code from https://github.com/stmorse/hawkes and https://stmorse.github.io/journal/Hawkes-python.html; slightly modified for our purposes

**Node.py** lightweight node structure for cascade trees, used by hawkes_tree.py and CascadeTree.py

**CascadeTree.py** structure to visualize (and maybe eventually analyze) a cascade tree, given the root as a Node object

**hawkes_tree.py** eventually, should generate a comment cascade (tree) given parameters - work in progress

**fit_weibull.py** given a sequence of event times, fit a Weibull pdf to the distribution

**fit_lognormal.py** given a sequence of event times, fit a log-normal to the distribution


### Directories

fit_testing : code from testing various fit methods

plots : output plots from various analyses

reddit_prediction_code : code from the reddit paper (https://arxiv.org/pdf/1801.10082.pdf) in it's mostly original form, plus some added functionality

results : results files from analyses

seismic : original and modified SEISMIC code; Python version of SEISMIC model code (http://snap.stanford.edu/seismic/) based on associated Twitter paper (https://arxiv.org/pdf/1506.02594.pdf); predicts the final retweet count of a particular tweet based on the observed retweet history

tensor_testing : trying out some tensor libraries/methods

