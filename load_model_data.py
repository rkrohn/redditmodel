
import file_utils
import data_utils
import tarfile
import os
import shutil
import glob
import cascades

#load all reddit data (comments and posts), and save as pickles
#code = {cyber, crypto, cve}, indicating reddit data to load
def load_reddit_data(code):

	#load comments, either from cached pickle or directly from data
	comments = load_cached_comments(code)
	#load posts/submissions, either from cached pickle or directly from data
	posts = load_cached_posts(code)

	if code == "cyber":		
		#load comments directly, and save pickle for later
		if comments == False:
			print("Loading comments from source")
			#extract comment files from tar if not already
			if os.path.isdir("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent") == False:
				tar = tarfile.open("../2018DecCP/Reddit/Cyber/Tng_an_RC_Cyber_sent.tar")
				tar.extractall("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent")
				tar.close()
			#load each comment file
			comments = []
			for filename in sorted(os.listdir("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent")):
				#get list of comments from this file
				new_comments = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent/" + filename)
				#add them all to a single list
				comments = data_utils.combine_lists(comments, new_comments)
			#dump comments
			save_comments(code, comments)

		#load posts directly, and save pickle for later
		if posts == False:
			print("Loading posts from source")
			#load json.gz file
			posts = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Cyber/Tng_an_RS_Cyber_sent.json.gz")
			#save to pickle
			save_posts(code, posts)

	elif code == "crypto":
		#load comments directly, and save pickle for later
		if comments == False:
			print("Loading comments from source")
			#load json.gz files
			comments1 = file_utils.load_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RC_3_coins_sent.json")
			comments2 = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RC_additional_coins_sent.json.gz")
			#combine into a single list, removing any duplicates
			comments = data_utils.combine_lists(comments1, comments2)
			#dump comments
			save_comments(code, comments)

		#load posts directly, and save pickle for later
		if posts == False:
			print("Loading posts from source")
			#load json.gz files
			posts = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RS_3_coins_sent.json.gz")
			posts.extend(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RS_additional_coins_sent.json.gz"))
			#save to pickle
			save_posts(code, posts)

	else:		#cve
		#load comments directly, and save pickle for later
		if comments == False:
			print("Loading comments from source")
			'''
			True
			* 2018DecCP/Reddit/CVE/Tng_an_RC_CVE_sent.json.gz: Reddit comments that contain a reference to CVE (e.g., Reddit Comment A-2, A-2-2)

			T/F
			* 2018DecCP/Reddit/CVE/Tng_an_RC_CVE_LINK_sent.json.gz:  All comments in submissions where above comment appeared (e.g., Reddit Comment A-1, A-1-1, A-2, A-2-1, A-2-2)

			T/F
			* 2018DecCP/Reddit/CVE/Tng_an_RC_CVE_SUB_sent.json.gz: All comments for submission mentioning CVE  (e.g.,Reddit Comment B-1, B-2)
			'''

			#load comments on any submissions, these may or may not contain a cve reference - assume false for now
			#this list will probably contain duplicates, will clean them up in the next step
			maybe_comments1 = data_utils.add_field(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/CVE/Tng_an_RC_CVE_LINK_sent.json.gz"), "cve_mention", False)
			maybe_comments2 = data_utils.add_field(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/CVE/Tng_an_RC_CVE_SUB_sent.json.gz"), "cve_mention", False)
			maybe_comments = data_utils.combine_lists(maybe_comments1, maybe_comments2) 

			#load comments with CVE mention (may be duplicates)
			comments = data_utils.add_field(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/CVE/Tng_an_RC_CVE_sent.json.gz"), "cve_mention", True)

			#combine lists together: want all of the comments with mentions, and all the unique comments without
			comments = data_utils.combine_lists(comments, maybe_comments, boolean_true="cve_mention")

			#dump comments
			save_comments(code, comments)

		'''
		True
		* 2018DecCP/Reddit/CVE/Tng_an_RS_CVE_sent.json.gz: All Reddit submissions mentioning CVE number  (e.g., Reddit Submission B)

		T/F
		* 2018DecCP/Reddit/CVE/Tng_an_RS_CVE_LINK_sent.json: Reddit submissions/posts that were referred to by a comment with a reference to CVE (Reddit Submission A)
		'''

		#load posts directly, and save pickle for later
		if posts == False:
			print("Loading posts from source")
			#load posts with CVE mention, and add correct field
			posts = data_utils.add_field(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/CVE/Tng_an_RS_CVE_sent.json.gz"), "cve_mention", True)
			#build list with these post ids
			post_ids = set(x['id_h'] for x in posts)

			#load posts that may or may not contain a CVE mention, assume they are false for now
			maybe_posts = data_utils.add_field(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/CVE/Tng_an_RS_CVE_LINK_sent.json.gz"), "cve_mention", False)

			#combine lists together: want all of the posts with mentions, and all the unique posts without
			for item in maybe_posts:
				if item['id_h'] not in post_ids:
					posts.append(item)
					post_ids.add(item['id_h'])

			#save to pickle
			save_posts(code, posts)

	print("")
	return posts, comments

#end load_reddit_data

#load all exogenous data (content depends on use case), save some to pickles
#code = {cyber, crypto, cve}, indicating exogenous data to load
def load_exogenous_data(code):

	print("Loading exogenous %s data" % code)

	if code == "cyber":

		#load major incidents
		major_incidents = file_utils.load_multi_json("../2018DecCP/Exogenous/Cyber_event/Major_Cyber_Incidents_2015-201708.json")
		print("Loaded", len(major_incidents), "incidents from Major_Cyber_Incidents_2015-201708.json")

		#load hackmageddon
		hackmageddon = file_utils.load_csv("../2018DecCP/Exogenous/Cyber_event/hackmageddon_2015-2018_sorted.csv")
		print("Loaded", len(hackmageddon), "events from hackmageddon_2015-2018_sorted.csv")

		#load hackernews (pain)
		print("Loading hackernews data")
		#extract monthly files from tar if not already and unpack the crappy file structure
		if os.path.isdir("../2018DecCP/Exogenous/hackernews/UNPACK_hackernews") == False:
			tar = tarfile.open("../2018DecCP/Exogenous/hackernews/hackernews_032015_to_032018.tar.gz", "r:gz")
			tar.extractall("../2018DecCP/Exogenous/hackernews/UNPACK_hackernews")
			tar.close()

			#move unpacked files to top level
			for f in os.listdir("../2018DecCP/Exogenous/hackernews/UNPACK_hackernews/Users/emmaprice/code/SocialSim/hackernews_files/"):
				shutil.move("../2018DecCP/Exogenous/hackernews/UNPACK_hackernews/Users/emmaprice/code/SocialSim/hackernews_files/" + f, "../2018DecCP/Exogenous/hackernews/UNPACK_hackernews/")
			shutil.rmtree("../2018DecCP/Exogenous/hackernews/UNPACK_hackernews/Users/")

			#handle any floating files - unzip and put them in the same spot
			files = glob.glob('../2018DecCP/Exogenous/hackernews/*.json*')
			for file in files:
				#extract month and year, build matching filename
				month = file[45:47]
				year = file[47:51]
				new_filename = "HNI_%s-%s" % (year, month)
				#unzip file to new destination
				file_utils.unzip_gz(file, "../2018DecCP/Exogenous/hackernews/UNPACK_hackernews/" + new_filename)

		#load in all that data
		hackernews = []
		for file in glob.glob("../2018DecCP/Exogenous/hackernews/UNPACK_hackernews/*"):
			#get list of comments from this file
			data = file_utils.load_multi_json(file)
			#add them all to a single list
			hackernews.extend(data)
		print("   Loaded", len(hackernews), "hackernews items")

	elif code == "crypto":

		#load crypto price data
		#extract coin files from tar if not already done
		if os.path.isdir("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price") == False:
			tar = tarfile.open("../2018DecCP/Exogenous/Crypto_Price/crypto_price_training.tar.gz", "r:gz")
			tar.extractall("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price")
			tar.close()

			#move unpacked files to top level
			for f in os.listdir("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price/crypto_price_training/"):
				shutil.move("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price/crypto_price_training/" + f, "../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price/")
			shutil.rmtree("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price/crypto_price_training/")

			#fix file error: first data row is on same line as headers
			for file in glob.glob("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price/*"):
				source = open(file) 
				line = source.readline()
				index = line.find('timestamp') + len('timestamp')
				output_line = line[:index] + '\n' + line[index:]
				dest = open(file, mode="w")
				dest.write(output_line)
				shutil.copyfileobj(source, dest)

		#load in all that data
		prices = {}		#dictionary of coin_id -> list of prices
		for file in os.listdir("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price/"):
			#get list of comments from this file
			data = file_utils.load_csv("../2018DecCP/Exogenous/Crypto_Price/UNPACK_crypto_price/" + file)
			#add them all to a single dictionary
			prices[file[:-11]] = data
		print("   Loaded prices for", len(prices), "coins")

	else:		#cve
		print("no data for you - weird schema?")
		'''
		#load all nvd files
		files = glob.glob('../2018DecCP/Exogenous/NVD/*.json.gz')
		data = []
		for file in files:
			print(file)
			data = file_utils.load_zipped_json(file)
			print(type(data), len(data))
			for key in data.keys():
				print(key)
				if key == "CVE_data_type":
					print(data[key])
				elif key == "CVE_Items":
					pass
				else:
					print(len(data[key]))
					print(data[key].keys())
			break
		'''

#end load_exogenous_data

#load cached comments from pickle file
def load_cached_comments(code):
	#load comments, either from cached pickle or directly from data
	if os.path.exists("data_cache/%s_comments.pkl" % code):
		#load from pickle
		print("Loading comments from data_cache")
		comments = file_utils.load_pickle("data_cache/%s_comments.pkl" % code)
		print("   Loaded", len(comments))
		return comments
	elif code == "cyber"  and os.path.exists("data_cache/cyber_comments"):
		#load from multiple pickles
		print("Loading comments from data_cache")
		comments = []
		files = sorted(glob.glob('data_cache/cyber_comments/*'))
		for file in files:
			print("   Loading", file)
			new_comments = file_utils.load_pickle(file)
			comments.extend(new_comments)
		print("   Loaded", len(comments))
		return comments
	else:
		return False
#end load_comments

#load cached posts from pickle file
def load_cached_posts(code):
	if os.path.exists("data_cache/%s_posts.pkl" % code):
		#load from pickle
		print("Loading posts from data_cache")
		posts = file_utils.load_pickle("data_cache/%s_posts.pkl" % code)
		print("   Loaded", len(posts))
		return posts
	else:
		return False
#end load_posts

#save all loaded comments to pickle
def save_comments(code, comments):
	#save all comments to pickle
	print ("Loaded", len(comments), "comments, saving to data_cache/%s_comments.pkl" % code)
	if not os.path.exists("data_cache"):
		os.makedirs("data_cache")
	#break cyber comments into 32 separate files, because memory error
	if code == "cyber":
		if not os.path.exists("data_cache/cyber_comments"):
			os.makedirs("data_cache/cyber_comments")
		for i in range(0, len(comments), 1000000):
			file_utils.save_pickle(comments[i:i+1000000], "data_cache/cyber_comments/%s_comments_%s.pkl" % (code, i//1000000))
	else:
		file_utils.save_pickle(comments, "data_cache/%s_comments.pkl" % code)
	print("   Comments saved")
#end save_comments

#save all loaded posts to pickle
def save_posts(code, posts):
	print("Loaded", len(posts), "posts, saving to data_cache/%s_posts.pkl" % code)
	if not os.path.exists("data_cache"):
		os.makedirs("data_cache")
	file_utils.save_pickle(posts, "data_cache/%s_posts.pkl" % code)
	print("   Posts saved")
#end save_posts

code = "cyber"

print("Processing", code)

posts, comments = load_reddit_data(code)
cascades.build_cascades(posts, comments, code)

#load_exogenous_data(code)