from MHP import MHP
import numpy as np
import os

#create directory for output plots if it does not exist
if not os.path.exists("tree_plots"):
    os.makedirs("tree_plots")

#univariate process with default parameters mu=[0.1], alpha=[[0.5]], and omega=1.0
#sequence is stored as P.data, a numpy.ndarray with 2 columns: the first column with the timestamps, the second with the stream assignment (in this case there is only one stream)

P = MHP()
P.generate_seq(60)
P.plot_events(filename="tree_plots/univariate.png")
P.plot_rates(filename="tree_plots/univariate_events_and_rates.png")


#multivariate, custom parameters

m = np.array([0.2, 0.0, 0.0])
a = np.array([[0.1, 0.0, 0.0], 
              [0.9, 0.0, 0.0],
              [0.0, 0.9, 0.0]])
w = 3.1

P = MHP(mu=m, alpha=a, omega=w)
P.generate_seq(60)
P.plot_events(filename="tree_plots/multivariate_events.png")
P.plot_rates(filename="tree_plots/multivariate_events_and_rates.png")



#build a tree?

#first, get top-level comments - ie, replies to the post
#