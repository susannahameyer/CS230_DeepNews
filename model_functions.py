import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import collections
import random
from scipy import spatial
import io
from preprocess import load_glove, get_input_data, build_dictionaries, create_embeddings, get_input_labels

# pulls from input_data, which is a list of all tokenized articles, and returns a list of all tokenized articles
# but with word ids instead of words
def get_input_data_as_ids(word2id, input_data):
	input_data_as_ids = []
	for article in input_data:
		word_as_index_list = []
		for word in article:
			index = word2id[word]
			word_as_index_list.append(index)
		input_data_as_ids.append(word_as_index_list)
	return np.array(input_data_as_ids)

# returns the next batch from the entire set of articles
def get_batch(max_article_length, input_data_as_ids, input_scores, batch_size, num_batches, wordToID):
	for i in range(num_batches):
		masks=[]
		batch_input_data_as_ids = []
		batch_input_scores = []
		article_count = 0
		for article_num in range(i * batch_size, (i + 1) * batch_size):
			batch_input_data_as_ids.append(input_data_as_ids[article_num])
			batch_input_scores.append(input_scores[article_num])
		#max_article_length = len(max(batch_input_data_as_ids, key=len))
		padded_batch_input_data_as_ids = []
		for article_id_list in batch_input_data_as_ids:
			mask = [1] * len(article_id_list)
			while len(article_id_list) < max_article_length:
				article_id_list.append(wordToID['<PAD>'])
				mask.append(0)
			padded_batch_input_data_as_ids.append(article_id_list)
			masks.append(mask)
			#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
		#print padded_batch_input_data_as_ids.shape
		#take care of padding and mask here and also get labels
		yield max_article_length, np.asarray(padded_batch_input_data_as_ids), np.expand_dims(np.asarray(batch_input_scores), axis=1), np.asarray(masks)
		#for scores, np.expand_dims(np.asarray(batch_input_scores), axis=1)

# create model placeholders
def create_placeholders(max_article_length, batch_size, vocab_size, embedding_dim):
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
	#article_size = max(lengths(X_test))
	inputs_placeholder = tf.placeholder(tf.int32, shape=[None, max_article_length], name= "inputs_placeholder") #must edit this!!!
	masks_placeholder = tf.placeholder(tf.int32, shape=[None, max_article_length], name= "masks_placeholder") #need this to match!
	scores_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name= "scores_placeholder")
	embedding_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, embedding_dim], name= "embedding_placeholder")
	return inputs_placeholder, masks_placeholder, scores_placeholder, embedding_placeholder

# mean squared error cost function
def get_cost(predictions, true_labels):
    temp = tf.square(tf.subtract(predictions, true_labels))
    return tf.reduce_mean(temp)
