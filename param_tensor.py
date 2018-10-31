#methods for building, factorizing, and lookups from the param tensor

import string

#(this should probably be a class, but let's go quick and dirty for now)

#given a post, extract words by tokenizing and normalizing (no limitization for now)
#removes all leading/trailing punctuation
#converts list to set to remove duplicates (if we want that?)
def extract_tokens(post):
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