#methods for building, factorizing, and lookups from the param tensor

import string

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
			self.test = []

	#given a post, extract words by tokenizing and normalizing (no limitization for now)
	#removes all leading/trailing punctuation
	#converts list to set to remove duplicates (if we want that?)
	def extract_tokens(self, post):
		punctuations = list(string.punctuation)		#get list of punctuation characters

		#grab post title as new string
		title = post['title_m']

		#tokenize and normalize (to lower)
		tokens = [word.lower() for word in title.split()]

		#remove punctuation-only tokens
		tokens = [word for word in tokens if word not in punctuations]

		#strip trailing and leading punctuation
		tokens = [word.strip("".join(punctuations)) for word in tokens]

		#convert to set
		tokens = set(tokens)

		print(tokens)
	#end extract_tokens

	#load a tensor from saved pickle
	def load_cached_tensor(self):
		print("unpickle some things")
	#end load_cached_tensor

	#dump a pickle to tensor
	def save_tensor(self):
		print("pickle some things")
	#end save_tensor