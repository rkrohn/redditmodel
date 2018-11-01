#methods for building, factorizing, and lookups from the param tensor

import string
import numpy as np

#params are indexed as follows:
#	a 		0
#	lbd 	1
#	k 		2
#	mu 		3
#	sigma	4
#	n_b		5

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
			self.tensor = None			#parameter tensor
										#index by parameter id (0-5), then user id, then token id
			self.tensor_count = None	#count of number of posts making up each entry in the tensor
			self.user_idx = {}			#maps user_id -> tensor index
			self.token_idx = {}			#maps token string -> tensor index
			self.post_tokens = {}		#maps post_id -> set of tokens
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
		print("   Found", len(self.user_idx), "users")

		#tokenize all posts, building token->index dictionary and a post_id-> list of tokens
		self.__build_token_dict(posts)
		print("   Found", len(self.token_idx), "tokens")

		#give tensor the right shape/size
		self.tensor = np.zeros((6, len(self.user_idx), len(self.token_idx)))
		#and a parallel count array, indicating how many post's paramters go into that field's average
		self.tensor_count = np.zeros((len(self.user_idx), len(self.token_idx)))

		#sum up each param in tensor, keeping count of how many are summed
		for post_id, post in posts.items():		#loop all posts
			#grab post author, convert to tensor index
			user_id = self.user_idx[post['author_h']]
			#pull post params for easy access
			post_params = params[post_id]

			#loop all tokens for this post
			for token in self.post_tokens[post_id]:
				#grab token tensor index
				token_id = self.token_idx[token]

				#loop all 6 params, add to relevant tensor fields
				for i in range(6):
					self.tensor[i][user_id][token_id] += post_params[i]
				#update count for this field
				self.tensor_count[user_id][token_id] += 1

		#divide tensor entries by count to get average
		sparse_count = 0		#count of entries that are non-zero
		for user_id in range(len(self.user_idx)):	#user index
			for token_id in range(len(self.token_idx)):		#token idx
				#if count for this token/user combo is not 0, divide to get average
				if self.tensor_count[user_id][token_id] != 0:
					sparse_count += 1
					for i in range(6):	#param index
						self.tensor[i][user_id][token_id] /= self.tensor_count[user_id][token_id]

		print("   Filled", sparse_count, "entries of", len(self.user_idx) * len(self.token_idx))

		print("")

	#end build_tensor


	#compute number of users for set of posts given, and build user_id->tensor index dictionary
	def __build_user_dict(self, posts):
		self.user_idx = {}		#clear the dictionary
		next_index = 0

		for post_id, post in posts.items():
			if post['author_h'] not in self.user_idx:
				self.user_idx[post['author_h']] = next_index
				next_index += 1
	#end __build_user_dict


	#tokenize/normalize all posts, and build token->tensor index dictionary
	def __build_token_dict(self, posts):
		self.token_idx = {}			#clear the dictionaries
		self.post_tokens = {}
		next_index = 0

		for post_id, post in posts.items():
			tokens = self.__extract_tokens(post)	#get tokens for this post
			self.post_tokens[post_id] = tokens

			for token in tokens:
				if token not in self.token_idx:
					self.token_idx[token] = next_index
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