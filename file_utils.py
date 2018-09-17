#utility methods for loading/reading data files and saving pickles

import json
import gzip

#given a filepath to a zipped json file, load the data
def load_zipped_json(filename):
	with gzip.open(filename, "rb") as f:
		d = json.loads(f.read().decode("utf-8"))
#end load_zipped_json

#given a filepath, load pickled data
def load_pickle(filename):
	with open(filename, "rb") as f:
		data = pickle.load(f)
	return data
#end load_pickle