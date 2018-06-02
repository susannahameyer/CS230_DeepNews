import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import collections
import random
from scipy import spatial
import io
from preprocess import load_glove, get_input_data, build_dictionaries, create_embeddings, get_input_labels, get_input_data_per_batch, get_input_labels_per_batch

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

# pass in folder name for train/dev/test and inside this function, convert to ids
def get_batch(max_article_length, input_data_as_ids, input_scores, batch_size, num_batches, wordToID):
	for i in range(num_batches):
		masks=[]
		batch_input_data_as_ids = []
		batch_input_scores = []
		for article_num in range(i * batch_size, (i + 1) * batch_size):
			batch_input_data_as_ids.append(input_data_as_ids[article_num])
			batch_input_scores.append(input_scores[article_num])
		#max_article_length = len(max(batch_input_data_as_ids, key=len))
		padded_batch_input_data_as_ids = []
		for article_id_list in batch_input_data_as_ids:
			mask = [1.0] * len(article_id_list)
			while len(article_id_list) < max_article_length:
				article_id_list.append(wordToID['<PAD>'])
				mask.append(-100.0)
			padded_batch_input_data_as_ids.append(article_id_list)
			masks.append(mask)
		yield np.asarray(padded_batch_input_data_as_ids), np.expand_dims(np.asarray(batch_input_scores), axis=1), np.asarray(masks)
		#for scores, np.expand_dims(np.asarray(batch_input_scores), axis=1)

def get_batch_from_folder(max_article_length, folder_name, batch_size, num_batches, wordToID):
	for i in range(num_batches):
		masks=[]
		batch_input_data_as_ids = []
		batch_input_scores = []
		article_count = 0
		batch_articles = get_input_data_per_batch(batch_size, i * batch_size, folder_name)
		batch_articles_ids = get_input_data_as_ids(wordToID, batch_articles)
		batch_input_scores = get_input_labels_per_batch(batch_size, i * batch_size, folder_name)

		padded_batch_articles_ids = []
		for article_id_list in batch_articles_ids:
			mask = [1.0] * len(article_id_list)
			if len(article_id_list) > max_article_length:
				article_id_list = article_id_list[:max_article_length]
			while len(article_id_list) < max_article_length:
				article_id_list.append(wordToID['<PAD>'])
				mask.append(-100.0)
			padded_batch_articles_ids.append(article_id_list)
			masks.append(mask)
		yield np.asarray(padded_batch_articles_ids), np.expand_dims(np.asarray(batch_input_scores), axis=1), np.asarray(masks)


def run_and_eval_dev(sess, max_article_length, folder_name, dev_batch_size, num_dev_batches, wordToID, embeddings, cost, predictions):
	batches = get_batch_from_folder(max_article_length, folder_name, batch_size, num_batches, wordToID)
	all_batch_predictions = np.zeros(shape=(dev_batch_size, num_dev_batches, 1), dtype=np.float32)
	all_batch_labels = np.zeros(shape=(dev_batch_size, num_dev_batches, 1), dtype=np.float32)

	for batch in range(num_dev_batches):
		padded_batch_articles_ids, batch_labels, batch_masks = batches.next()
		all_batch_labels[:, batch] = batch_labels
		_ , batch_cost, batch_predictions = sess.run([cost, predictions], feed_dict={inputs_placeholder: padded_batch_articles_ids, masks_placeholder: batch_masks, scores_placeholder: batch_labels, embedding_placeholder: embeddings})
		all_batch_predictions[:, batch] = batch_predictions

	#Evaluate entire dev set
	similarity_threshold = 0.1
	correctly_scored_count = 0
	score_differences = abs(all_batch_labels - all_batch_predictions)
	correctly_scored_count = np.sum(score_differences < similarity_threshold)
	performance = tf.divide(correctly_scored_count, len(batch_predictions)*dev_batch_size)

	print "Dev correctly scored count: " + str(correctly_scored_count)
	print "Dev performance: " + str(performance)



# create model placeholders
def create_placeholders(max_article_length, batch_size, vocab_size, embedding_dim):
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
	#article_size = max(lengths(X_test))
	inputs_placeholder = tf.placeholder(tf.int32, shape=[batch_size, max_article_length], name= "inputs_placeholder") #must edit this!!!
	masks_placeholder = tf.placeholder(tf.float32, shape=[batch_size, max_article_length], name= "masks_placeholder") #need this to match!
	scores_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name= "scores_placeholder")
	embedding_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, embedding_dim], name= "embedding_placeholder")
	return inputs_placeholder, masks_placeholder, scores_placeholder, embedding_placeholder

# mean squared error cost function
def get_cost(predictions, true_labels):
    temp = tf.square(tf.subtract(predictions, true_labels))
    return tf.reduce_mean(temp)





# glove_vocab, glove_to_embedding, embedding_dim = load_glove()
# total_num_articles = 10
# #vocab_size = len(word2id)
# input_data = get_input_data(total_num_articles)
# input_scores = get_input_labels(total_num_articles)

# all_articles_words = [] #list of every unique word in all articles (set)
# max_article_length = 0 #length of the maximum-length article in our input data
# for article in input_data:
# 	if len(article) > max_article_length:
# 		max_article_length = len(article)
# 	for word in article:
# 		if word not in all_articles_words:
# 			all_articles_words.append(word)
# all_articles_words.append('<PAD>')

# word2id, id2word, vocab_size = build_dictionaries(all_articles_words)



# batches = get_batch_from_folder(1433, "articles", 5, 2, word2id)
# max_article_length, padded_batch_articles_ids, batch_labels, batch_masks = batches.next()
# print padded_batch_articles_ids
# print batch_labels
# print batch_masks
# print "next batch"
# max_article_length, padded_batch_articles_ids, batch_labels, batch_masks = batches.next()
# print padded_batch_articles_ids
# print batch_labels
# print batch_masks
