import math
import numpy as np
import scipy.stats 		#actually need scipy.stats.chi2
import file_utils

#GLOBAL VARS
#here is where you set the platform parameters used as defaults by the functions
#for now, default values measured from real Twitter dataset (see paper)

THETA = 0.2314843		#power law exponent determined from reaction time distribution
CUTOFF = 300			#time in seconds where density function changes from const to power law
C = 0.0006265725		#constant density when t < cutoff
MAX_WINDOW = 2 * 60 * 60	#maximum span of the locally weighted kernel (seconds)
MIN_WINDOW = 300		#minimum span of the locally weighted kernel (seconds)
MIN_COUNT = 5			#minimum number of resharings included in the window
N_STAR = 100			#average node degree in social network

#MEMORY KERNEL - implemented across memory_pdf and memory_ccdf

#returns probability density function of human reaction time at time t for memory kernel
#
#inputs:
#   t       time
#   theta   exponent of the power law (determined from reaction time dist of sample tweets)
#   cutoff  cutoff value (time) where density function changes from constant to power law
#           measured in seconds from post time 0
#   c       constant density when t < cutoff
#
#returns:   density at time t
def memory_pdf(t, theta = THETA, cutoff = CUTOFF, c = C):
	if t < cutoff:
		return c
	else:
		return c * map.pow((t / cutoff), -(1 + theta))
#end memory_pdf

#returns complementary cumulative distribution function of human reaction time 
#at time t for memory kernel (probability of greater than t)
#
#inputs:
#   t       list of times
#   theta   exponent of the power law (determined from reaction time dist of sample tweets)
#   cutoff  cutoff value (time) where density function changes from constant to power law
#           measured in seconds from post time 0
#   c       constant density when t < cutoff
#
#returns:   list of densities at time t
def memory_ccdf(t, theta = THETA, cutoff = CUTOFF, c = C):
	#make sure all t are non-negative
	t = [x if x >=0 else 0 for x in t]

	#compute and return value for each t
	result = [1 - c * x if x <= cutoff else c * pow(cutoff, 1 + theta) / theta * (pow(x, -theta)) for x in t]
	return result
	'''
	if t <= cutoff:
		return 1 - c * t
	else:
		return c * pow(cutoff, 1 + theta) / theta * (pow(t, -theta))
	'''
#end memory_ccdf


#INTEGRAL KERNEL - integration with respect to locally weighted kernel
#across functions linear_kernel, power_kernel, and integral_memory_kernel

#returns the integral from vector t1 to vector t2 of c*[slope(t - ptime) + 1]
#
#inputs:
#	t1		vector of integral lower limit
#	t2		vector of integral upper limit
#	ptime	time at which to estimate infectiousness and predict popularity
#	slope	slope of the linear kernel
#	c 		constant density when t < cutoff
def linear_kernel(t1, t2, ptime, slope, c = C):
	#indefinite integral is c*(t-ptime*slope*t+(slope*t^2)/2)
	return (c * (t2 - (ptime * slope * t2) + (slope * pow(t2, 2) / 2))) - (c * (t1 - (ptime * slope * t1) + (slope * pow(t1,2) / 2)))
#end linear_kernel

#returns the integral from vector t1 to vector t2 of c*((t-share_time)/cutoff)^(-(1+theta))[slope(t-ptime) + 1]
#
#inputs:
#	t1			vector of integral lower limit
#	t2			vector of integral upper limit
#	ptime		time at which to estimate infectiousness and predict popularity
#	share_time 	sorted observed resharing times, share_time[0] = 0
#	slope		slope of the linear kernel
#   theta   	exponent of the power law (determined from reaction time dist of sample tweets)
#   cutoff  	cutoff value (time) where density function changes from constant to power law
#           	measured in seconds from post time 0
#	c 			constant density when t < cutoff
def power_kernel(t1, t2, ptime, share_time, slope, theta = THETA, cutoff = CUTOFF, c = C):
	return (c*pow(cutoff,(1+theta))*pow((t2-share_time),(-theta))*(share_time*slope-theta+(theta-1)*ptime*slope-theta*slope*t2+1)/((theta-1)*theta) - c*pow(cutoff,(1+theta))*pow((t1-share_time),(-theta))*(share_time*slope-theta+(theta-1)*ptime*slope-theta*slope*t1+1)/((theta-1)*theta))
#end power_kernel

#returns the vector with ith entry being integral(-inf to inf) of ptime[i] * kernel(t - ptime)
#
#inputs:
#	ptime		time at which to estimate infectiousness and predict popularity
#	share_time 	list of sorted observed resharing times, share_time[0] = 0
#	slope		slope of the linear kernel
#	window		size of the linear kernel
#   theta   	exponent of the power law (determined from reaction time dist of sample tweets)
#   cutoff  	cutoff value (time) where density function changes from constant to power law
#           	measured in seconds from post time 0
#	c 			constant density when t < cutoff
def integral_memory_kernel(ptime, share_time, slope, window, theta = THETA, cutoff = CUTOFF, c = C):
	#build list/vector of integral values, one per share_time value
	integral = [0] * len(share_time)

	for i in range(len(share_time)):
		val = share_time[i]

		if ptime <= val:
			integral[i] = 0
		elif ptime > val and ptime <= val + cutoff:
			integral[i] = linear_kernel(val, ptime, ptime, slope)
		elif ptime > val + cutoff and ptime <= val + window:
			integral[i] = linear_kernel(val, val + cutoff, ptime, slope) + power_kernel(val + cutoff, ptime, ptime, val, slope)
		elif ptime > val + window and ptime <= val + window + cutoff:
			integral[i] = linear_kernel(ptime - window, val + cutoff, ptime, slope) + power_kernel(val + cutoff, ptime, ptime, val, slope)
		elif ptime > val + window + cutoff:
			integral[i] = power_kernel(ptime - window, ptime, ptime, val, slope)

	return integral
#end integral_memory_kernel


#INFECTIOUSNESS - estimate the infectiousness of an information cascade
#uses a triangular kernel with shape changing over time
#at time ptime, use a triangluer kernel with slope = min(max(1/(p.time/2), 1/min_window), max_window)
#
#inputs:
#	share_times 	list of sorted observed resharing times, share_times[0] = 0
#	degree 		observed node degrees, list same length as share_times
#	ptimes		equally spaced list of times to estimate the infectiousness, ptimes[0] = 0
#	max_window 	maximum span of the locally weight kernel 
#	min_window 	minimum span of the locally weight kernel
#	min_count 	minimum number of resharings included in the window
#
#returns:	a list of three vectors
#		infectiousness 	the estimated infectiousness
#		p_up			the upper 95% appox confidence interval
#		p_low			the lower 95% approx confidence interval

def get_infectiousness(share_times, degree, ptimes, max_window = MAX_WINDOW, min_window = MIN_WINDOW, min_count = MIN_COUNT):

	#make sure share times are sorted
	share_times.sort()

	#build list of slopes and windows corresponding to prediction times
	#slope = 1 / (ptime / 2)
	slopes = [1.0 / (x / 2) if x != 0 else float("inf") for x in ptimes]
	#force slopes to range (1/max_window, 1/min_window)
	slopes = [1.0 / max_window if x < 1.0 / max_window else x for x in slopes]
	slopes = [1.0 / min_window if x > 1.0 / min_window else x for x in slopes]

	#window = ptime / 2
	windows = [x / 2 for x in ptimes]
	#force windows to range (min_window, max_window)
	windows = [max_window if x > max_window else x for x in windows]
	windows = [min_window if x < min_window else x for x in windows]

	#for each ptime, verify the slopes and windows to ensure at least min_count shares are included in the window
	for j in range(len(ptimes)):
		#get indexes of share_times that fall within this prediction time's window
		ind = [i for i, x in enumerate(share_times) if x >= ptimes[j] - windows[j] and x < ptimes[j]]
		#if too few shares within defined window, adjust window to contain at least min_count
		if len(ind) < min_count:
			#get indices of share times ocurring before current prediction time
			ind2 = [i for i, x in enumerate(share_times) if x < ptimes[j]]
			ind = ind2[max((len(ind2)-min_count), 1):]	#set new window indices
			if len(ind) >= 1:
				slopes[j] = 1 / (ptimes[j] - share_times[ind[0]])
				windows[j] = ptimes[j] - share_times[ind[0]]
			else:
				slopes[j] = 0
				windows[j] = None

	#matrix for integral of the	memory kernel, stored as numpy array
	#len(share_times) rows, len(ptimes) columns
	#build, then convert and transpose
	int_mem_kernel = []
	for i in range(len(ptimes)):
		res = integral_memory_kernel(ptimes[i], share_times, slopes[i], windows[i])
		int_mem_kernel.append([degree[j] * res[j] for j in range(len(degree))])
	#convert and transpose
	int_mem_kernel = np.transpose(np.array(int_mem_kernel))

	#make infectiousness predictions!
	infectiousness_seq = [0] * len(ptimes)
	p_low = [None] * len(ptimes)
	p_up = [None] * len(ptimes)
	share_times = share_times[1:]	#remove original post (first item)

	for i in range(len(ptimes)):
		share_time_tri = [x for x in share_times if windows[i] != None and x >= ptimes[i] - windows[i] and x < ptimes[i] ]		#get share times within window for this prediction
		
		rt_count_weighted = sum([slopes[i]*(x - ptimes[i]) + 1 for x in share_time_tri])

		integral_sum = np.sum(int_mem_kernel, axis=0)[i]	#sum the column

		rt_num = len(share_time_tri)

		if rt_count_weighted == 0:
			continue
		else:
			infectiousness_seq[i] = rt_count_weighted / integral_sum
			#use scipy.stats.chi2.ppf in place of R's qchisq (https://stackoverflow.com/questions/18070299/is-there-a-python-equivalent-of-rs-qchisq-function)
			p_low[i] = infectiousness_seq[i] * scipy.stats.chi2.ppf(0.05, 2 * rt_num) / (2 * rt_num)
			p_up = infectiousness_seq[i] * scipy.stats.chi2.ppf(0.95, 2 * rt_num) / (2 * rt_num)

	return infectiousness_seq, p_up, p_low
#end get_infectiousness


#PREDICTION - predict the popularity of an information cascade
#
#inputs:
#	ptimes			equally spaced list of times to estimate the infectiousness, ptimes[0] = 0
#	infectiousness 	a vector of estimated infectiousness, returned by get_infectiousness (same length as ptimes)
#	share_times 	list of sorted observed resharing times, share_times[0] = 0
#	degree 			observed node degrees (same length as share_times)
#	n_star			average node degree in the social network
#	features_return	if True, function returns a matrix of features to be used to further calibrate the prediction
#	
#returns:	a vector of predicted popularity at each time in ptim

def predict_cascade(ptimes, infectiousness, share_times, degree, n_star = N_STAR, features_return = False):

	#convert n_star to vector/list the same length as ptimes
	n_star = [n_star] * len(ptimes)

	#to train for the best n_star value, build a feature matrix
	#end up with len(ptimes) rows, 3 columns
	features = []

	#and a prediction matrix/array
	prediction = []

	#loop all prediction times
	for i in range(len(ptimes)):
		#get share times we have seen
		share_times_now = [x for x in share_times if x <= ptimes[i]]
		#and corresponding node degrees
		nf_now = [degree[j] for j, x in enumerate(share_times) if x <= ptimes[i]]

		rt0 = len(share_times_now) - 1	#scalar
		res = memory_ccdf([ptimes[i] - x for x in share_times_now])
		rt1 = sum([nf_now[j] * infectiousness[i] * res[j] for j in range(len(nf_now))])	#scalar

		#make this prediction
		prediction.append(rt0 + rt1 / (1 - infectiousness[i] * n_star[i])) 

		#feature nonsense
		features.append([rt0, rt1, infectiousness[i]])
		#each row is current # of retweets, numerator, and infectiousness

		#if infectiousness too high, set prediction to infinity
		if infectiousness[i] > 1 / n_star[i]:
			prediction[i] = float("inf")

		if features_return:
			return prediction, features
		else:
			return prediction
#end predict_cascade




#' @examples
#' data(tweet)
#' pred.time <- seq(0, 6 * 60 * 60, by = 60)
#' infectiousness <- get.infectiousness(tweet[, 1], tweet[, 2], pred.time)
#' pred <- pred.cascade(pred.time, infectiousness$infectiousness, tweet[, 1], tweet[, 2], n.star = 100)
#' plot(pred.time, pred)


#load tweet example
tweet = file_utils.read_csv_list("tweet.csv")
#convert to two lists: times and followers
relative_time_seconds = [int(x[0]) for x in tweet[1:]]
number_of_followers = [int(x[1]) for x in tweet[1:]]

#print(relative_time_seconds)
#print(number_of_followers)

pred_times = range(0, 6*60*60 + 60, 60)
#print(pred_times)

#infectiousness <- get.infectiousness(tweet[, 1], tweet[, 2], pred.time)
infectiousness, p_up, p_low = get_infectiousness(relative_time_seconds, number_of_followers, pred_times)
print(infectiousness)