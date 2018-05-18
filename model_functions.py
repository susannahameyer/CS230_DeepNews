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
def get_input_data_as_ids(wordToID, input_data):
	input_data_as_ids = []
	for article in input_data:
		word_as_index_list = []
		for word in article:
			index = wordToID[word]
			word_as_index_list.append(index)
		input_data_as_ids.append(word_as_index_list)
	return input_data_as_ids

# returns the next batch from the entire set of articles
def get_batch(input_data_as_ids, input_scores, num_articles_per_batch, num_batches, wordToID):
	for i in range(num_batches):
		masks=[]
		batch_input_data_as_ids = []
		batch_input_scores = []
		article_count = 0
		for article_num in range(i * num_articles_per_batch, (i + 1) * num_articles_per_batch):
			batch_input_data_as_ids.append(input_data_as_ids[article_num])
			batch_input_scores.append(input_scores[article_num])
		max_article_length = len(max(batch_input_data_as_ids, key=len))
		padded_batch_input_data_as_ids = []
		for article_id_list in batch_input_data_as_ids:
			mask = [1] * len(article_id_list)
			while len(article_id_list) < max_article_length:
				article_id_list.append(wordToID['<PAD>'])
				mask.append(0)
			padded_batch_input_data_as_ids.append(article_id_list)
			masks.append(mask)
		#take care of padding and mask here and also get labels
		yield np.asarray(padded_batch_input_data_as_ids), np.expand_dims(np.asarray(batch_input_scores), axis=1), np.asarray(masks)
		#for scores, np.expand_dims(np.asarray(batch_input_scores), axis=1)

# create model placeholders
def create_placeholders(vocab_size, embedding_dim):
	inputs_placeholder = tf.placeholder(tf.int32, shape=(None, None), name= "inputs_placeholder")
	scores_placeholder = tf.placeholder(tf.float32, shape=(None, 1), name= "scores_placeholder")
	masks_placeholder = tf.placeholder(tf.int32, shape=(None, None), name= "masks_placeholder")
	embedding_placeholder = tf.placeholder(tf.int32, shape=(vocab_size, embedding_dim), name= "embedding_placeholder")
	return inputs_placeholder, scores_placeholder, masks_placeholder, embedding_placeholder

# mean squared error cost function
def get_cost(predictions, true_labels):
    temp = tf.square(tf.subtract(predictions, true_labels))
    return tf.reduce_mean(temp)