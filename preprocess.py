######################################################################################################################################################################
###### 			Preprocess		######################################################################################################################################
######################################################################################################################################################################

import numpy as np
import collections
import random
import io
import os
import sys

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
	sys.stdout.flush()
	return glove_vocab, embedding_dict, len(embed_vector)

# Takes in folder_names_and_size, a list of tuples containing (folder_name, folder_size) pairs from which to pull data.
			# Recall folder_name is in ["train", "dev", "test"]
# Retrieves input data for the model, returning a list consisting each article's word tokenizations (a list of lists).
def get_input_data(folder_names_and_size):
	article_list = []
	for name, article_num in folder_names_and_size:
		article_file_list = os.listdir(name)
		article_file_list = sorted([ x for x in article_file_list if "label" not in x ], key=lambda x: int(x.split('.')[0]))
		# print article_file_list
		for i in range(article_num):
			filename = name + "/" + article_file_list[i]
			article = np.loadtxt(filename, dtype=np.str, ndmin=1)
			article_list.append(article)
	return article_list


def get_input_data_per_batch(batch_size, start_index, folder_name):
	article_file_list = os.listdir(folder_name)
	article_file_list = sorted([ x for x in article_file_list if "label" not in x ], key=lambda x: int(x.split('.')[0]))
	article_list = []
	# print article_file_list
	for i in range(start_index, start_index + batch_size):
		filename = folder_name + "/" + article_file_list[i]
		article = np.loadtxt(filename, dtype=np.str, ndmin=1)
		article_list.append(article)
	return article_list

def get_input_labels_per_batch(batch_size, start_index, folder_name):
	label_file_list = os.listdir(folder_name)
	label_file_list = sorted([ x for x in label_file_list if "label" in x ], key=lambda x: int(x.split('_')[0]))
	# print label_file_list
	label_list = []
	for i in range(start_index, start_index + batch_size):
		filename = folder_name + "/" + label_file_list[i]
		with open(filename) as label_file:
			label = label_file.readline()
			label_list.append(float(label))
	# print label_list
	return label_list


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

# Creates word2id and id2word dictionaries; returns these & number of words.
def build_dictionaries(words):
    count = collections.Counter(words).most_common() #creates list of (word, word_count) tuples;
    wordToID = dict()
    for word, _ in count:
        wordToID[word] = len(wordToID) #len(dictionary) increases each iteration; this is the wordID.
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


#Main function ~ calls the functions created above to fully preprocess data.
def preprocess():
	glove_vocab, glove_to_embedding, embedding_dim = load_glove()
	#TO DO: write results of load_glove, build_dictionaries, create_embeddings to file once and read in

	#These numbers were pre-calculated via command-line commands after running the script to divide the data into 70/20/10 train/dev/test folders.
	total_num_articles_train = 10 #35300
	total_num_articles_dev = 10 #10000

	# Retrieves input data for the model, returning a list consisting of each article's word tokenizations (a list of lists).
	input_data = get_input_data([("train", total_num_articles_train), ("dev", total_num_articles_dev)])
	print "Loaded input_data."
	sys.stdout.flush()

	all_articles_words = [] #list of every unique word in all articles (set)
	max_article_length = 0 #length of the maximum-length article in our input data

	# Iterate through all articles in both train & dev folders, storing the size of the longest article found and
	# builds a list of the full vocabulary across all the articles.
	for article in input_data:
		if article.size > max_article_length:
			max_article_length = article.size
		for word in article:
			if word not in all_articles_words:
				all_articles_words.append(word)
	all_articles_words.append('<PAD>') #Adds '<PAD>' word token to vocabulary of words.

	word2id, id2word, vocab_size = build_dictionaries(all_articles_words)
	print "Built dictionaries."
	sys.stdout.flush()

	embeddings = create_embeddings(vocab_size, glove_vocab, word2id, glove_to_embedding, embedding_dim)
	print "Created embeddings."
	sys.stdout.flush()

	return max_article_length, embedding_dim, vocab_size, embeddings, word2id
