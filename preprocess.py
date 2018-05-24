import numpy as np
import collections
import random
import io

# Loads GLOVE vectors
def load_glove(filepath='glove.6B.100d.txt'):
	glove_vocab = []
	embedding_dict = {}
	file = io.open(filepath,'r',encoding='UTF-8')
	for line in file.readlines():
		row = line.strip().split(' ')
		vocab_word = row[0]
		glove_vocab.append(vocab_word)
		embed_vector = [float(i) for i in row[1:]] # convert to list of float
		embedding_dict[vocab_word] = embed_vector
	file.close()
	print 'Loaded GLOVE'
	return glove_vocab, embedding_dict, len(embed_vector)

# Retrieves input data for the model. Returns an array of arrays: every element
# is a single article's word tokenizations.
def get_input_data(num_articles):
	article_list = []
	for i in range(num_articles):
		filename = str(i) + ".txt"
		try:
			article = np.loadtxt(filename, dtype=np.str)
			article_list.append(article)
		except:
			continue
	return article_list

# Retrieves input scores for the model, corresponding to the .txt files retrieved in get_input_data
def get_input_labels(num_articles):
	label_list = []
	for i in range(num_articles):
		filename = str(i) + "_label.txt"
		try:
			with open(filename) as label_file:
				label = label_file.readline()
				label_list.append(float(label))
		except:
			continue
	return label_list

# Create dictionary and reverse dictionary with word ids
def build_dictionaries(words):
    count = collections.Counter(words).most_common() #creates list of word/count pairs;
    wordToID = dict()
    for word, _ in count:
        wordToID[word] = len(wordToID) #len(dictionary) increases each iteration
    iDToWord = dict(zip(wordToID.values(), wordToID.keys()))
    return wordToID, iDToWord, len(wordToID)


# Create embedding array
def create_embeddings(vocab_size, glove_vocab, wordToID, embedding_dict, embedding_dim):
	dict_as_list = sorted(wordToID.items(), key = lambda x : x[1])
	embeddings_tmp = []
	for i in range(vocab_size):
		item = dict_as_list[i][0]
		if item in glove_vocab:
			embeddings_tmp.append(embedding_dict[item])
		else:
			rand_num = np.random.uniform(low=-0.2, high=0.2,size=embedding_dim)
			embeddings_tmp.append(rand_num)
	# final embedding array corresponds to dictionary of words in the document
	embeddings = np.asarray(embeddings_tmp, dtype=np.float32)

	return embeddings
