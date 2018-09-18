
import file_utils
import tarfile
import os
import shutil
import glob

#load all reddit data (comments and posts), and save as pickles
#code = {cyber, crypto, cve}, indicating reddit data to load
def load_reddit_data(code):

	if code == "cyber":
		#load comments, either from cached pickle or directly from data
		comments = load_cached_comments(code)
		#load directly, and save pickle for later
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
				data = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent/" + filename)
				#add them all to a single list
				comments.extend(data)
			#dump comments
			save_comments(code, comments)

		#load posts/submissions, either from cached pickle or directly from data
		posts = load_cached_posts(code)
		#load directly, and save pickle for later
		if posts == False:
			print("Loading posts from source")
			#load json.gz file
			posts = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Cyber/Tng_an_RS_Cyber_sent.json.gz")
			#save to pickle
			save_posts(code, posts)

	elif code == "crypto":
		#load comments, either from cached pickle or directly from data
		comments = load_cached_comments(code)
		#load directly, and save pickle for later
		if comments == False:
			print("Loading comments from source")
			#load json.gz files
			comments = file_utils.load_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RC_3_coins_sent.json")
			comments.extend(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RC_additional_coins_sent.json.gz"))
			#dump comments
			save_comments(code, comments)

		#load posts/submissions, either from cached pickle or directly from data
		posts = load_cached_posts(code)
		#load directly, and save pickle for later
		if posts == False:
			print("Loading posts from source")
			#load json.gz files
			posts = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RS_3_coins_sent.json.gz")
			posts.extend(file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Crypto/Tng_an_RS_additional_coins_sent.json.gz"))
			#save to pickle
			save_posts(code, posts)

	else:		#cve
		print("no data for you")

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
		print("no data for you")

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


#load_reddit_data("crypto")
load_exogenous_data("crypto")