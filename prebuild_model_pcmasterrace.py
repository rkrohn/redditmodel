#create and save all the necessary files for the new c++/python hybrid model
#save everything to model_files for easy server loading later

import cascade_analysis
import cascade_manip
import fit_cascade
import file_utils
from functions_prebuild_model import *

from itertools import count
import sys
from itertools import islice


def chunks(data, SIZE=50000):
	it = iter(data)
	for i in range(0, len(data), SIZE):
		yield {k:data[k] for k in islice(it, SIZE)}


#command line argument to specify chunk
count = int(sys.argv[1])


#filepaths of output files
params_filepath = "pcmasterrace/params/%s_params.txt"	#text file of fitted cascade params, one file per subreddit
														#one line per cascade: cascade numeric id, params(x6), sticky factor (1-quality)


subreddit = "pcmasterrace"
domain = "cyber"

print("\nProcessing", subreddit, "in", domain, "domain")


'''
#first, chunk the cascades
cascades = None
comments = None

#load cascades
cascades, comments = load_cascades(subreddit, domain, cascades, comments)

#partition these into ~8 chunks
count = 0
for chunk_cascades in chunks(cascades):
	print("Chunked to", len(chunk_cascades), "posts")
	junk, chunk_comments = cascade_manip.filter_comments_by_posts(chunk_cascades, comments)
	file_utils.save_pickle(chunk_cascades, "pcmasterrace/chunk_cascades_%s.pkl" % count)
	file_utils.save_pickle(chunk_comments, "pcmasterrace/chunk_comments_%s.pkl" % count)
	count += 1
exit(0)
'''


#once those chunks are created - call multiple instances, one per chunk

#load chunked posts/comments
print("Loading chunk")
cascades = file_utils.load_pickle("pcmasterrace/chunk_cascades_%s.pkl" % count)
comments = file_utils.load_pickle("pcmasterrace/chunk_comments_%s.pkl" % count)
print("Loaded", len(cascades), "posts and", len(comments), "comments")

#fit params to all cascades
all_params = cascade_analysis.fit_all_cascades(domain, cascades, comments, False, subreddit)

#save to text file now
with open(params_filepath % subreddit, "w") as f: 
	for post_id, params in all_params.items():
		f.write(str(posts[post_id]['id']) + " ")		#write numeric post id
		for i in range(len(params)):
			f.write((' ' if i > 0 else '') + str(params[i]))
		f.write("\n")
print("Saved text-readable params to", params_filepath % count)

