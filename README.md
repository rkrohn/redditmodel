# redditmodel

Code for modelling reddit cascades.

**load_model_data.py** load reddit and exogenous data for chosen situation (cyber, crypto, or cve)

**cascades.py** given loaded posts and comments for a particular domain (cyber, crypto, or cve), reconstruct the post cascade structure
this includes the creation of "dummy" objects for any posts/comments that are referred to by links, but are not present in the data

**data_utils.py** functions for data manipulation

**file_utils.py** functions for loading and saving of json and pickle files

**seismic.py** Python version of SEISMIC model code (http://snap.stanford.edu/seismic/) based on associated Twitter paper (https://arxiv.org/pdf/1506.02594.pdf)
predicts the final retweet count of a particular tweet based on the observed retweet history

**modified_seismic.py** starting to tweak the basic SEISMIC model in hopes of generating a retweet cascade/tree, instead of ust a final count

**tweet.csv** sample data for testing SEISMIC code, gives one retweet cascade
