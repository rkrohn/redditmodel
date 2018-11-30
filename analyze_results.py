#read, unpack, and play with dry run measurements

import file_utils
import glob
import csv

#glob filestring to get all results files
filestring = "dryrun/results/*-metrics.json"

#get list of all matching files
files = glob.glob(filestring)

data = {}	#nested dictionary of file/run identifier -> metric topic -> metric name -> computed metric

#process each file individually, add data to dictionary
for file in files:
	#get identifier for this file
	ident, rest = file.split('-')
	ident = ident.split('/')[-1]

	#load file data
	data[ident] = file_utils.load_json(file)

ident_column = 'sim version'

#for better csv dump-age, combine metric topic and name into a single key
#list of dictionaries, one per run/ident
dump_data = []
#loop identifiers
for ident in data:
	row = {ident_column : ident}
	#loop metric topics
	for topic in data[ident]:
		#loop metrics
		for metric in data[ident][topic]:
			#add this to updated data
			row[topic + " : " + metric] = data[ident][topic][metric]
	dump_data.append(row)

#build list of csv columns
columns = sorted(list(dump_data[0].keys()))
columns.remove(ident_column)
columns.insert(0, ident_column)

#dump to csv
with open("dryrun/results/combined_metrics.csv", 'w') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=columns)
	writer.writeheader()
	for data in dump_data:
		writer.writerow(data)
print("Combined results saved to dryrun/results/combined_metrics.csv")

