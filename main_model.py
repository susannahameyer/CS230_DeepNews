from model_functions import get_input_data_as_ids, get_batch, create_placeholders, get_cost, get_batch_from_folder, run_and_eval_dev
from preprocess import load_glove, get_input_data, build_dictionaries, create_embeddings, get_input_labels
import tensorflow as tf
import io
import numpy as np

def model(max_article_length, embedding_dim, vocab_size, embeddings, wordToID,
		  num_epochs = 10000, batch_size = 100, num_batches = 353, num_hidden_units = 100, learning_rate = 0.001):
# def model(max_article_length, embedding_dim, vocab_size, embeddings, wordToID,
# 		  num_epochs = 100, batch_size = 10, num_batches = 1, num_hidden_units = 100, learning_rate = 0.001):
	costs = []
	inputs_placeholder, masks_placeholder, scores_placeholder, embedding_placeholder = create_placeholders(max_article_length, batch_size, vocab_size, embedding_dim)
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
	embedded_chars = tf.nn.embedding_lookup(embedding_placeholder, inputs_placeholder)
	#Create basic RNN cell
	rnn_cell = tf.contrib.rnn.BasicRNNCell(num_hidden_units)
	#state is a tensor of shape [batch_size, cell_state_size]
	outputs, _ = tf.nn.dynamic_rnn(rnn_cell, embedded_chars, sequence_length = [max_article_length]*batch_size, dtype=tf.float32)
	#outputs shape=(10, 1433, 20) -- (batch_size, max_article_length, num_hidden_units)
	#states shape = (10, 20)
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

	masks_ = tf.reshape(masks_placeholder, shape=[batch_size, max_article_length, 1])
	# masks_ = tf.tile(masks_, [1, 1, num_hidden_units])

	padded_outputs = tf.multiply(outputs, masks_)

	output = tf.reduce_max(padded_outputs, axis=1)

	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

	predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn = tf.nn.sigmoid)
	#predictions = tf.sigmoid(predictions)

	cost = get_cost(predictions, scores_placeholder)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

	# Initialize all the variables
	init = tf.global_variables_initializer()

	# Start the session to compute the tensorflow graph
	with tf.Session() as sess:

		# Run the initialization
		sess.run(init)

		#70/20/10
		# Do the training loop
		for epoch in range(num_epochs):

			# add loop to get next batch
			epoch_cost = 0.0
			# batches = get_batch(max_article_length, input_data_as_ids, input_scores, batch_size, num_batches, wordToID)
			folder_name = "train"
			batches = get_batch_from_folder(max_article_length, folder_name, batch_size, num_batches, wordToID)

			#create generator for dev data
			#inside function where we do evaluation, iterate over entire dev set

			for batch in range(num_batches):
				#every x batches of training data, loop over dev data and calculate performance
				#for dev data, run session without optimizer
				# function for get batch, loop over batches, and evaluation for dev set
				# when ready for test set, just change folder from dev to test
				padded_batch_articles_ids, batch_labels, batch_masks = batches.next()
				#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

				_ , batch_cost, batch_predictions = sess.run([optimizer, cost, predictions], feed_dict={inputs_placeholder: padded_batch_articles_ids, masks_placeholder: batch_masks, scores_placeholder: batch_labels, embedding_placeholder: embeddings})
				#in if statement, pass sess into new function for dev/test and evaluate performance over entire set

				# compare batch_labels and batch_predictions here to get performance for train set
				similarity_threshold = 0.1
				correctly_scored_count = 0
				score_differences = abs(batch_labels - batch_predictions)
				correctly_scored_count = np.sum(score_differences < similarity_threshold)
				performance = tf.divide(correctly_scored_count,len(batch_predictions))
				if epoch % 10 == 0:
					print "Count of correct scores for batch " + str(batch) + ": " + str(correctly_scored_count)
					print "Performance for batch " + str(batch) + ": " + str(performance)


				#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

				epoch_cost += batch_cost / num_batches
			#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

			if epoch % 10 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if epoch % 5 == 0:
				costs.append(epoch_cost)
			run_and_eval_dev(sess, max_article_length, "dev", 100, 100, wordToID, embeddings, cost, predictions)

#TO DO: write results of load_glove, build_dictionaries, create_embeddings to file once and read in
glove_vocab, glove_to_embedding, embedding_dim = load_glove()
total_num_articles_train = 35300
total_num_articles_dev = 10000
# total_num_articles = 10
#vocab_size = len(word2id)
input_data = get_input_data([("train", total_num_articles_train), ("dev", total_num_articles_dev)])
print "loaded input_data"
# input_scores = get_input_labels(total_num_articles)

all_articles_words = [] #list of every unique word in all articles (set)
max_article_length = 0 #length of the maximum-length article in our input data
for article in input_data:
	if article.size > max_article_length:
		max_article_length = article.size
	for word in article:
		if word not in all_articles_words:
			all_articles_words.append(word)
all_articles_words.append('<PAD>')

word2id, id2word, vocab_size = build_dictionaries(all_articles_words)
print "built dictinoaries"

# x_train_ids = get_input_data_as_ids(word2id, input_data)
embeddings = create_embeddings(vocab_size, glove_vocab, word2id, glove_to_embedding, embedding_dim)
print "created embeddings"

#build test sets
# X_test = x_train_ids
# Y_test = input_scores

model(max_article_length, embedding_dim, vocab_size, embeddings, word2id)
