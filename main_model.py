from model_functions import get_input_data_as_ids, get_batch, create_placeholders, get_cost
from preprocess import load_glove, get_input_data, build_dictionaries, create_embeddings, get_input_labels
import tensorflow as tf
import io
import numpy as np


def model(max_article_length, input_data_as_ids, input_scores, X_test, Y_test, embedding_dim, vocab_size, embeddings, wordToID,
		  num_epochs = 100, batch_size = 10, num_batches = 1, num_hidden_units = 20, learning_rate = 0.0001):

	input_shape = input_data_as_ids.shape
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

		# move evaluation to inside epoch
		all_batch_predictions = np.zeros(shape=(batch_size, num_batches, 1), dtype=np.float32)
		all_batch_labels = np.zeros(shape=(batch_size, num_batches, 1), dtype=np.float32)
		#shape = (batch_size, num_batches) right??

		#70/20/10
		# Do the training loop
		for epoch in range(num_epochs):

			# add loop to get next batch
			epoch_cost = 0.0
			batches = get_batch(max_article_length, input_data_as_ids, input_scores, batch_size, num_batches, wordToID)

			#create generator for dev data
			#inside function where we do evaluation, iterate over entire dev set

			for batch in range(num_batches):
				#every x batches of training data, loop over dev data and calculate performance
				#for dev data, run session without optimizer
				# function for get batch, loop over batches, and evaluation for dev set
				# when ready for test set, just change folder from dev to test
				max_article_length, padded_batch_articles_ids, batch_labels, batch_masks = batches.next()

				#from IPython import embed3
				#embed()

				all_batch_labels[:, batch] = batch_labels
				#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

				_ , batch_cost, batch_prediction = sess.run([optimizer, cost, predictions], feed_dict={inputs_placeholder: padded_batch_articles_ids, masks_placeholder: batch_masks, scores_placeholder: batch_labels, embedding_placeholder: embeddings})
				
				#in if statement, pass sess into new function for dev/test and evaluate performance over entire set

				# compare batch_labels and batch_predictions here to get performance for train set

				all_batch_predictions[:, batch] = batch_prediction
				#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

				epoch_cost += batch_cost / num_batches

#				tf.assign(all_batch_predictions[:, batch], batch_prediction)

			#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

			if epoch % 10 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if epoch % 5 == 0:
				costs.append(epoch_cost)

	similarity_threshold = 0.1
	correctly_scored_count = 0
	# import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

	score_differences = abs(all_batch_labels - all_batch_predictions)
	correctly_scored_count = np.sum(score_differences < similarity_threshold)

	performance = tf.divide(correctly_scored_count,len(all_batch_predictions))

	print "Performance = " + str(performance)



#TO DO: write results of load_glove, build_dictionaries, create_embeddings to file once and read in
glove_vocab, glove_to_embedding, embedding_dim = load_glove()
total_num_articles = 10
#vocab_size = len(word2id)
input_data = get_input_data(total_num_articles)
input_scores = get_input_labels(total_num_articles)

all_articles_words = [] #list of every unique word in all articles (set)
max_article_length = 0 #length of the maximum-length article in our input data
for article in input_data:
	if len(article) > max_article_length:
		max_article_length = len(article)
	for word in article:
		if word not in all_articles_words:
			all_articles_words.append(word)
all_articles_words.append('<PAD>')

word2id, id2word, vocab_size = build_dictionaries(all_articles_words)

x_train_ids = get_input_data_as_ids(word2id, input_data)
embeddings = create_embeddings(vocab_size, glove_vocab, word2id, glove_to_embedding, embedding_dim)

#build test sets
X_test = x_train_ids
Y_test = input_scores

model(max_article_length, x_train_ids, input_scores, X_test, Y_test, embedding_dim, vocab_size, embeddings, word2id)
