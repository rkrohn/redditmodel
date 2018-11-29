#correct existing simulation output files to include the t3_ prefix on all ids

import file_utils
import glob

#glob filestring to get all results files
filestring = "dryrun/submit/sim_res_*.json"

#get list of all matching files
files = glob.glob(filestring)

prefix = "t3_"		#prefix to prepend to all id references

#process each file individually, correcting ids along the way
for file in files:
	print("\nCorrecting", file)

	#load the josn
	data = file_utils.load_json(file)
	print("  ", len(data['data']), "events to fix")

	#correct comment/post records, where each is a dictionary of the following form:
	#{"parentID": "A4XW5Jol_qVgUAKDWeOeaw", "communityID": "t5_3i6d8", "rootID": "A4XW5Jol_qVgUAKDWeOeaw", "nodeUserID": "okcc60doiWAfkR89nAAvHQ", "nodeTime": "1501876531", "nodeID": "A4XW5Jol_qVgUAKDWeOeaw", "actionType": "post"}
	for event in data['data']:

		#fix id fields
		event['parentID'] = prefix + event['parentID']
		event['rootID'] = prefix + event['rootID']
		event['nodeID'] = prefix + event['nodeID']

	#save the updated file overtop of the old one
	file_utils.save_json(data, file)
	print("Corrected file saved to", file)

