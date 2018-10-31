#methods for building, factorizing, and lookups from the param tensor

import string
import numpy as np

class ParamTensor:

	def __init__(self, filename = None):
		#if given filename of cached tensor, load from disk
		if filename != None and file_utils.verify_file(filename):
			load_cached_tensor(filename)
		#given filename, but file doesn't exist - error message, and create a new empty tensor object
		elif filename != None:
			print("No cached tensor to load - creating new object")
			filename = None

		#no filename given, create new tensor object
		if filename == None:
			self.tensor = []
			self.user_dict = {}
			self.token_dict = {}
	#end __init__


	#given a set of posts and corresponding fitted parameters, build the 3-D tensor
	#index tensor by userid, then words, then paramter index
	def build_tensor(self, posts, params):
		#make sure we have same number of posts and paramters
		if len(posts) != len(params):
			print("Post and paramter sets are not the same size - skipping tensor build")
			return
		print("Building tensor for", len(posts), "posts")

		#build user_id->index dictionary
		self.__build_user_dict(posts)
		print("   Found", len(self.user_dict), "users")

		#tokenize all posts, building token->index dictionary
		self.__build_token_dict(posts)
		print("   Found", len(self.token_dict), "tokens")

		#give tensor the right shape/size
		self.tensor = np.zeros((len(self.user_dict), len(self.token_dict), 5))

		print("")

	#end build_tensor


	#compute number of users for set of posts given, and build user_id->tensor index dictionary
	def __build_user_dict(self, posts):
		self.user_dict = {}		#clear the dictionary
		next_index = 0

		for post_id, post in posts.items():
			if post['author_h'] not in self.user_dict:
				self.user_dict[post['author_h']] = next_index
				next_index += 1
	#end __build_user_dict


	#tokenize/normalize all posts, and build token->tensor index dictionary
	def __build_token_dict(self, posts):
		self.token_dict = {}	#clear the dictionary
		next_index = 0

		for post_id, post in posts.items():
			tokens = self.__extract_tokens(post)	#get tokens for this post

			for token in tokens:
				if token not in self.token_dict:
					self.token_dict[token] = next_index
					next_index += 1
	#end __build_token_dict

	#given a post, extract words by tokenizing and normalizing (no limitization for now)
	#removes all leading/trailing punctuation
	#converts list to set to remove duplicates (if we want that?)
	def __extract_tokens(self, post):
		punctuations = list(string.punctuation)		#get list of punctuation characters
		
		title = post['title_m']		#grab post title as new string		
		tokens = [word.lower() for word in title.split()]	#tokenize and normalize (to lower)		
		tokens = [word for word in tokens if word not in punctuations]		#remove punctuation-only tokens		
		tokens = [word.strip("".join(punctuations)) for word in tokens]		#strip trailing and leading punctuation

		return set(tokens)		#convert to set before returning
	#end __extract_tokens


	#load a tensor from saved pickle
	def load_cached_tensor(self):
		print("unpickle some things")
	#end load_cached_tensor


	#dump a pickle to tensor
	def save_tensor(self):
		print("pickle some things")
	#end save_tensor

#end ParamTensor