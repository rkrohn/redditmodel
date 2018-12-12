#given cascades in scenario2 output format (baseline2-sample_train_crypto.json and baseline1-sample_train_cryptoreddit.json),
#convert them to simulated cascade tree format (reddit_CVE_simulated_cascades.json)

#desired format
'''
"t3_1qKdzDuScYNNgXmB7nNg7Q": {
	"post_freq": 1, 
	"root": {"children": [{"id": 1, "children": [], "time": 4752.69486103414}, {"id": 2, "children": [], "time": 5815.654732250838}, {"id": 3, "children": [], "time": 5928.558991673393}, {"id": 4, "children": [], "time": 6153.1699359940585}, {"id": 5, "children": [], "time": 6365.390717651069}, {"id": 6, "children": [], "time": 7128.1397465476875}, {"id": 7, "children": [], "time": 7858.0277689068325}, {"id": 8, "children": [], "time": 8631.262080868222}, {"id": 9, "children": [], "time": 9294.886750943093}, {"id": 10, "children": [], "time": 9919.834638083834}, {"id": 11, "children": [], "time": 11412.743187336453}, {"id": 12, "children": [], "time": 12343.300893828917}], "id": 0, "time": 0}, "user": "uxZcuTqkW2ugeNO8BFVCAA", "prams": [9.103714577365668, 8727.133944221368, 3.92639574805435, 0, 1.5, 0.05]}
'''
#dictionary, where post id is key
#maps to dictionary with keys:
#	post_freq = 1
#	root -> "children" list
#	time
#each child is dictionary, with their own children list and time fields

#input is a list of dictionaries, each with the following fields
#	nodeID
#	nodeUserID
#	parentID	(immmediate parent)
#	rootID		(post parent)
#	actionType	(post or comment)
#	nodeTime	(seconds)
#	communityID
'''
{"nodeID": "t3_YJsQ1UdSE4IMQQRtHZS_Kw", "nodeUserID": "qYbJWQfgn0AMeWeRFunuOw", "parentID": "t3_YJsQ1UdSE4IMQQRtHZS_Kw", "rootID": "t3_YJsQ1UdSE4IMQQRtHZS_Kw", "actionType": "post", "nodeTime": 1501725487, "communityID": "t5_2s3qj"}
'''

import sys

import file_utils


if len(sys.argv) != 3:
	print("wrong args - need input file and output file")
	exit(0)

infile = sys.argv[1]
outfile = sys.argv[2]

#read input data
data = file_utils.load_json(infile)['data']
print("Read", len(data), "events")

#pull parent posts from data, add to output
output = {}
for event in data:
	if event['actionType'] == "post":
		output[event['nodeID']] = {'post_freq': 1, "root": {'time': event['nodeTime'], 'children': list()}}
print("Found", len(output), "posts")

#filter data to remove posts we accounted for
data = [event for event in data if event['actionType'] != "post"]
print("Down to", len(data), "events")

post_count = 0

#reconstruct each cascade
for post_id, post in output.items():

	#print("Processing post", post_id)

	#root time = 0
	post_time = output[post_id]['root']['time']
	output[post_id]['root']['time'] = 0
	#print(post_time, "->", output[post_id]['root']['time'])

	#get list of events associated with this cascade
	comments = [event for event in data if event['rootID'] == post_id]
	#remove them from data
	data = [event for event in data if event['rootID'] != post_id]

	#reconstruct these comments into a tree (not quite the right format, yet)
	comment_tree = {}	#comment id -> {children -> list of child ids, time -> time}
	comment_tree = {comment['nodeID']: {'children': list(), 'children_ids': list(), 'time': (comment['nodeTime'] - post_time) / 60, 'parent': comment['parentID']} for comment in comments}

	#add all comments to correct child list
	for comment in comments:
		if comment['parentID'] == post_id:
			output[post_id]['root']['children'].append(comment_tree[comment['nodeID']])
		else:
			comment_tree[comment['parentID']]['children_ids'].append(comment['nodeID'])
			comment_tree[comment['parentID']]['children'].append(comment_tree[comment['nodeID']])

	#remove unnecessary fields: children_ids, parent
	for comment_id, comment_dict in comment_tree.items():
		comment_dict.pop('children_ids', None)
		comment_dict.pop('parent', None)

	post_count += 1
	if post_count % 100 == 0:
		print("Finished", post_count, "posts")

#save results
print("Saving results")
file_utils.save_json(output, outfile)