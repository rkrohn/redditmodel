import file_utils
import glob

#list of files to combine
files = glob.glob('output/*.json')
print(len(files), "files:", files)

count = 0
all_data = None
event_count = 0

for file in files:
	file_data = file_utils.load_json(file)

	event_count += len(file_data['data'])

	if count == 0:
		all_data = file_data
	else:
		all_data['data'].extend(file_data['data'])

	count += 1

print("Sum event count:", event_count)
print("Combined event count:", len(all_data['data']))

#save all data
file_utils.save_json(all_data, "output/all_cyber_sim_res.json")
