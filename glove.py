import argparse
import numpy as np
import sys
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity
from nltk.corpus import stopwords
from nltk import download

import file_utils

#path to pre-computed embedding vectors (vector dimension, one of 50, 100, 200, or 300)
vector_filepath = "reddit_data/glove_embeddings/glove.6B.%dd.txt"
#path to converted pre-computed embedding vectors (same dimensions)
#these will contain the same information as the standard glove files, but are in the gensim format
gensim_vector_filepath = "reddit_data/glove_embeddings/glove.6B.%dd.gensim.txt"

#global display flag
DISPLAY = True  


#load pre-computed embedding vectors for given dimension
def load_embeddings(dimension = 50):
    #read in pre-computed embedding vectors
    with open(vector_filepath % dimension, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]     #store as dictionary of token -> vector

    vocab_size = len(vectors)       #number of words/tokens in vocabulary
    words = list(vectors.keys())
    vocab = {w: idx for idx, w in enumerate(words)}     #map words to index
    ivocab = {idx: w for idx, w in enumerate(words)}    #map index to words

    #convert vectors to numpy array: one embedding vector per row, <dimension> columns
    W = np.zeros((vocab_size, dimension))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    #normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))        #row-wise sum of squared values, then take the square-root of that
    W_norm = (W.T / d).T                    #divide each row's values by the corresponding d value
    #if we compute d for each row now, the result will be 1 - unit variance!

    return (W_norm, vocab, ivocab)
#end load_embeddings


#verify that all text words are in vocabulary
#text given as list of lowercase tokens
#returns normalized vector representation of input text
#if some tokens are not in the vocabulary, skip them and use the defined tokens instead
#if ALL text tokens not in vocabulary, return None
def get_text_vector(text, word_embeddings, vocab):
    missing_count = 0       #count of text words not in vocabulary

    #loop text terms
    for idx, term in enumerate(text):
        #if term in vocab, add that word's representation to running list
        if term in vocab:
            if DISPLAY: print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            #save this term's vector
            if idx == 0:
                vec_result = np.copy(word_embeddings[vocab[term], :])   #start with just the first word
            else:
                vec_result += word_embeddings[vocab[term], :]       #add values of this word to running sum
        #word not in vocabulary, error and return False for now
        else:
            if DISPLAY: print('Word: %s  Out of vocabulary' % term)
            missing_count += 1

    #if all of text not in vocabulary, return None
    if missing_count == len(text)  :
        if DISPLAY: print("All of text tokens out of dictionary, returning None")
        return None
    #some number missing, print error if desired and proceed
    elif missing_count != 0:
        if DISPLAY: print('%d words out of dictionary!' % missing_count)

    #normalize the input vector representation - unit variance
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    return vec_norm
#end get_text_vector


#compute cosine distance between a word (or phrase) and all words in the vocabulary, 
#displaying the top 10 matching results from the vocabulary
#text is given as list of tokens, all lowercase
def cosine_distance(word_embeddings, vocab, ivocab, text, num_results = 10):
    #get normalized vector representation of text
    vec_norm = get_text_vector(text, word_embeddings, vocab)
    if vec_norm is None:
        return False

    #compute cosine distance between input text and all words in vocabulary (dot product)
    dist = np.dot(word_embeddings, vec_norm.T)

    #set distance to words in text (and vocab) to -inf, so they are not included in ranking
    for term in text:
        if term not in vocab: continue      #skip words outside of vocab
        index = vocab[term]
        dist[index] = -np.Inf

    #sort and get the top-n matches
    a = np.argsort(-dist)[:num_results]

    if DISPLAY:
        print("\n                               Word       Cosine distance")
        print("---------------------------------------------------------")
        for x in a:
            print("%35s\t\t%f" % (ivocab[x], dist[x]))

    #return top matches (list of indexes) and all distances
    return a, dist
#end cosine_distance


#remove stopwords from list of tokens
def remove_stopwords(tokens):
    #download stopwords list from nltk - only once per run
    try:
        remove_stopwords.downloaded += 1      #(static flag hack)
    except AttributeError:
        remove_stopwords.downloaded = 0
        download('stopwords')  
    stop_words = stopwords.words('english')

    #remove stopwords from given tokens list
    updated_tokens = [w for w in tokens if w not in stop_words]
    return updated_tokens
#end remove_stopwords


#given two sentences as lists of lowercase tokens, and an initialized model, 
#compute the word-mover distance between them
def wmd_distance(model, sentence1, sentence2):
    #remove stopwords from both sentences
    sentence1 = remove_stopwords(sentence1)
    sentence2 = remove_stopwords(sentence2)

    #compute wmd between them
    distance = model.wmdistance(sentence1, sentence2)
    if DISPLAY: print('distance = %.4f' % distance)

    return distance
#end wmd_distance


#for the specified dimension, load precomputed glove embeddings and return a gensim model
def get_glove_model(dimension = 50):
    #convert Stanford glove format to gensim word2vec format, if file doesn't exist
    if file_utils.verify_file(gensim_vector_filepath % dimension) == False:
        glove2word2vec(glove_input_file = (vector_filepath % dimension), word2vec_output_file = (gensim_vector_filepath % dimension))

    #load embeddings and init model (word2vec model, but init with glove embeddings)
    glove_model = KeyedVectors.load_word2vec_format(gensim_vector_filepath % dimension, binary=False)

    return glove_model
#end get_glove_model


#given an initialized model and a query corpus, get a similarity model
def get_similarity_model(model, corpus, top_n = 10):
    #build/index similarity model
    similarity_model = WmdSimilarity(corpus, model, num_best=top_n)
    return similarity_model
#end get_similarity_model

def get_most_similar(query, sim_model, corpus):
    #make query
    sims = similarity_model[query]

    #print
    print("\nQuery: %s" % query)
    for i in range(len(sims)):
        print('sim = %.4f' % sims[i][1])
        print(corpus[sims[i][0]])

    return sims
#end get_most_similar


if __name__ == "__main__":
    #old functionality, loading embeddings and computing distance manually
    '''
    #load pre-computed embeddings, normalize
    word_embeddings, word_to_index, index_to_word = load_embeddings()

    while True:
        input_term = input("\nEnter word or sentence (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            cosine_distance(word_embeddings, word_to_index, index_to_word, input_term.split(' '))  #get cosine distance   
    '''

    #new gensim functionality - given two sentences, compute wmd distance and similarity between them

    #get a model by reading precomputed embeddings
    glove_model = get_glove_model()

    #get two sentences
    phrase1 = 'Obama speaks to the media in Illinois'
    phrase2 = 'The president greets the press in Chicago'
    phrase3 = 'Oranges are my favorite fruit'
    #lowercase and tokenize
    token_phrase1 = phrase1.lower().split()
    token_phrase2 = phrase2.lower().split()
    token_phrase3 = phrase3.lower().split()

    #compute wmd between them
    print("Distance between \"%s\" and \"%s\": " % (phrase1, phrase2))
    distance = wmd_distance(glove_model, token_phrase1, token_phrase2)
    print("Distance between \"%s\" and \"%s\": " % (phrase1, phrase3))
    distance = wmd_distance(glove_model, token_phrase1, token_phrase3)

    #try a similarity ranking, using our precomputed vectors to init the model, and a separate corpus
    from gensim.test.utils import common_texts
    similarity_model = get_similarity_model(glove_model, common_texts, top_n = 5)
    similarity_results = get_most_similar(['person', 'time'], similarity_model, common_texts)