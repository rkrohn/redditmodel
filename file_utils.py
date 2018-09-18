#utility methods for loading/reading data files and saving pickles

import json
import gzip
import tarfile
import pickle

#given a filepath to a zipped json file, load the data
def load_zipped_json(filename, display = True):
	if display:
		print ("Loading", filename)
	with gzip.open(filename, "rb") as f:
		d = json.loads(f.read().decode("utf-8"))
	return d
#end load_zipped_json

#given a .json.gz that contains multiple json objects, read data
def load_zipped_multi_json(filename, display = True):
	if display:
		print ("Loading", filename)
	with gzip.GzipFile(filename, 'r',) as f:
		d = []
		for line in f:
			d.append(json.loads(line.decode('utf-8')))
	return d
#end load_zipped_multi_json

#given a filename, load the json
def load_json(filename):
	with open(filename, "rb") as f:
		return json.loads(f.read().decode("utf-8"))
#end load_json

#given a filename, load the multiple json objects
def load_multi_json(filename):
	d = []
	curr = ""
	with open(filename) as f:
		for line in f:
			curr += line
			try:
				jobj = json.loads(curr)
				d.append(jobj)
				curr = ""
			except ValueError:
				#not yet a complete JSON value
				pass
	return d
#end load_json

#given a filepath, load pickled data
def load_pickle(filename):
	with open(filename, "rb") as f:
		data = pickle.load(f)
	return data
#end load_pickle

#save data object as pickle
def save_pickle(data, filename):
	with open(filename, "wb") as f:
		pickle.dump(data, f)
#end dump_list