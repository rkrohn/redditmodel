#convert fitted params from pkl to node2vec format by printing
#too lazy to do file ops now, just pipe it
import cascade_manip

domain = "crypto"
subreddit = "Lisk"

sub_params = cascade_manip.load_cascade_params(domain, subreddit, display=False)


for post_id, params in sub_params.items():
	print(post_id, end='')
	for param in params:
		print("", param, end='')
	print("")

