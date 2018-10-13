from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

#given dictionary of form key->count, compute frequencies of different counts
def count_freq(data, local = True):
	freq = defaultdict(int)
	min = -1
	max = -1
	for key in data:
		if local and key[0] == '.':
			continue
		freq[data[key]] = freq[data[key]] + 1
		if min == -1 or data[key] < min:
			min = data[key]
		if max == -1 or data[key] > max:
			max = data[key]
	return freq, min, max
		
	
#given frequencies as dictionary, key = size, value = freq, plot them	
def plot_freq(freq, xlabel, ylabel, title, filename = "", x_max = 0, x_min = 0, log_scale = False):
	plt.clf()	
	lists = sorted(freq.items())
	x,y = zip(*lists)
	fig, ax = plt.subplots()
	plt.plot(x,y)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if log_scale:
		ax.set_yscale('log')
		ax.set_xscale('log')
	if x_max != 0 and x_min != 0:
		plt.xlim(xmin=x_min, xmax=x_max)
	elif x_max != 0:
		plt.xlim(xmin=0, xmax=x_max)
	elif x_min != 0:
		plt.xlim(xmin=x_min, xmax=x_max)	
	if filename == "":
		plt.show()
	else:
		plt.savefig(filename, bbox_inches='tight')
		
#plot data given as x and y lists	
def plot_data(x, y, xlabel, ylabel, title, filename = "", x_min = 0, x_max = 0, log_scale_x = False, log_scale_y = False):
	plt.clf()	
	fig, ax = plt.subplots()

	plt.plot(x, y)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if log_scale_x:
		ax.set_xscale('log')
	if log_scale_y:
		ax.set_yscale('log')
	if x_max != 0 and x_min != 0:
		plt.xlim(xmin=x_min, xmax=x_max)
	elif x_max != 0:
		plt.xlim(xmin=0, xmax=x_max)
	elif x_min != 0:
		plt.xlim(xmin=x_min, xmax=x_max)	
	if filename == "":
		plt.show()
	else:
		plt.savefig(filename, bbox_inches='tight')


#plot dictionary data with keys on x-axis and values on y-axis
def plot_dict_data(data, xlabel, ylabel, title, filename = "", x_min = 0, x_max = 0, log_scale_x = False, log_scale_y = False):
	#break dictionary data into lists
	x = []
	y = []
	for key in sorted(data.keys()):
		#only include data within x-axis range so plot method will set y-axis range correctly
		if (x_min == 0 and x_max == 0) or (key >= x_min and key <= x_max):
			x.append(key)
			y.append(data[key])

	#call plotting method on list data
	plot_data(x, y, xlabel, ylabel, title, filename, 0, 0, log_scale_x, log_scale_y)
		
		
#plot data given as x and 2 y lists	- will have 2 y axes on plot
def plot_two_axes(x, data1, data2, xlabel, ylabel1, ylabel2, title, filename = ""):
	plt.clf()	
		
	fig, ax1 = plt.subplots()

	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(ylabel1, color='b')
	ax1.plot(x, data1, 'b-')
	ax1.tick_params(axis='y', labelcolor='b')

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	ax2.set_ylabel(ylabel2, color='r')  # we already handled the x-label with ax1
	ax2.plot(x, data2, 'r-')
	ax2.tick_params(axis='y', labelcolor='r')
	
	plt.title(title)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped	
		
	if filename == "":
		plt.show()
	else:
		plt.savefig(filename, bbox_inches='tight')		