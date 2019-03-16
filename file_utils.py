#utility methods for loading/reading data files and saving pickles

import json
import gzip
import tarfile
import pickle
import csv
import os
import tarfile

DISPLAY = False

#given a filepath to a zipped json file, load the data
def load_zipped_json(filename):
	if DISPLAY:
		print ("Loading", filename)
	with gzip.open(filename, "rb") as f:
		d = json.loads(f.read().decode("utf-8"))
	return d
#end load_zipped_json

#given a .json.gz that contains multiple json objects, read data
def load_zipped_multi_json(filename):
	if DISPLAY:
		print ("Loading", filename)
	with gzip.GzipFile(filename, 'r',) as f:
		d = []
		for line in f:
			d.append(json.loads(line.decode('utf-8')))
	return d
#end load_zipped_multi_json

#given a filename, load the json
def load_json(filename):
	if DISPLAY:
		print("Loading", filename)
	with open(filename, "rb") as f:
		return json.loads(f.read().decode("utf-8"))
#end load_json

#save some data structure to json file
def save_json(data, filename):
	with open(filename, 'w') as fp:
		json.dump(data, fp, indent=4, sort_keys=False)
#end save_json

#given a filename, load the multiple json objects
def load_multi_json(filename):
	if DISPLAY:
		print("Loading", filename)
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
	if DISPLAY:
		print("Loading", filename)
	with open(filename, "rb") as f:
		data = pickle.load(f)
	return data
#end load_pickle

#save data object as pickle
def save_pickle(data, filename):
	with open(filename, "wb") as f:
		pickle.dump(data, f)
#end dump_list

#read data from csv file into list of lists
def read_csv_list(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		data = list(reader)
	return data
#end read_csv_list

#loads csv data into nested dictionary structure (like what you get from json)
def load_csv(filename):
	with open(filename, 'r') as f:
		d = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
		return d
#end load_csv

#save list of dictionary objects to csv, filtering to given fields if list is given
def save_csv(data, filename, fields=False):
	#define list of fieldnames, if not given
	if fields == False:
		fields = list(data[0].keys())

	with open(filename, mode='w') as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fields)

		writer.writeheader()
		for row in data:
			writer.writerow({key: row[key] for key in fields})
#end save_csv

#unpacks gzip file to desired destination
def unzip_gz(source, dest):
	inF = gzip.GzipFile(source, 'rb')
	s = inF.read()
	inF.close()

	outF = open(dest, 'wb')
	outF.write(s)
	outF.close()
#end unzip_gz

#given a path to a directory, create it if it does not exist
def verify_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
#end verify_dir

#given a filepath, return true if the file exists
#(tiny helper function, because I can never remember this syntax!)
def verify_file(path):
	return os.path.exists(path)
#end verify_file

#list all files in tar file
def list_tar(filename):
	tar = tarfile.open(filename)	#"../2018DecCP/Reddit/Cyber/Tng_an_RC_Cyber_sent.tar"
	files = tar.getmembers()
	for file in files:
		print(file)
#end list_tar