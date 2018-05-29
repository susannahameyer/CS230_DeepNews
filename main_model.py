from model_functions import get_input_data_as_ids, get_batch, create_placeholders, get_cost
from preprocess import load_glove, get_input_data, build_dictionaries, create_embeddings, get_input_labels
import tensorflow as tf
import io
from vocab import get_glove
import numpy as np


def model(max_article_length, input_data_as_ids, input_scores, X_test, Y_test, embedding_dim, vocab_size, embeddings, wordToID,
		  num_epochs = 100, batch_size = 10, num_batches = 1, num_hidden_units = 20, learning_rate = 0.0001):

	input_shape = input_data_as_ids.shape
	costs = []
	inputs_placeholder, masks_placeholder, scores_placeholder, embedding_placeholder = create_placeholders(max_article_length, batch_size, vocab_size, embedding_dim)
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

	# RNN output node weights and biases
	# use tf.get_variable
	weights = { 'out': tf.Variable(tf.random_normal([num_hidden_units, embedding_dim])) }
	biases = { 'out': tf.Variable(tf.random_normal([embedding_dim])) }
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
	embedded_chars = tf.nn.embedding_lookup(embedding_placeholder, inputs_placeholder)
	#initial_state = tf.zeros([batchSize, max_article_length, embedding_dim]), dtype=tf.float32)

	#returns 10 [batch_size] tensors that are [max_article_length, embedding_dim]
	#embedded_chars_unstack = tf.unstack(embedded_chars, batch_size, axis=0)

	#Create basic RNN cell
	rnn_cell = tf.contrib.rnn.BasicRNNCell(num_hidden_units)

	# Defining initial state
	#initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

	#state is a tensor of shape [batch_size, cell_state_size]
	outputs, _ = tf.nn.dynamic_rnn(rnn_cell, embedded_chars, sequence_length = [max_article_length]*batch_size, dtype=tf.float32)
	#outputs shape=(10, 1433, 20) -- (batch_size, max_article_length, num_hidden_units)
	#states shape = (10, 20)

	#APPLY MASK so that all places where padding is cannot affect outputs
	#wherever there is padding, make those values "-inf" (yes, bc we're doing max)
	#change to -inf
	#if we're doing average over RNN, set to 0
	#convert all cells with <PAD> to -inf (if max) or 0 if avg
	# mask should have 1s where <PAD> exists if multiplying by -inf

	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT


	masks_ = tf.reshape(masks_placeholder, shape=[batch_size, max_article_length, 1])
	masks_ = tf.tile(masks_, [1, 1, num_hidden_units])

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

		all_batch_predictions = np.zeros(shape=(batch_size, num_batches, 1), dtype=np.float32)
		all_batch_labels = np.zeros(shape=(batch_size, num_batches, 1), dtype=np.float32)
		#shape = (batch_size, num_batches) right??

		# Do the training loop
		for epoch in range(num_epochs):

			# add loop to get next batch
			epoch_cost = 0.0
			# when we add more batches, loop through them here
			batches = get_batch(max_article_length, input_data_as_ids, input_scores, batch_size, num_batches, wordToID)

			for batch in range(num_batches):
				max_article_length, padded_batch_articles_ids, batch_labels, batch_masks = batches.next()
				#batch_articles_ids needs to be of size [num_articles_per_batch, max_article_length]

				#from IPython import embed3
				#embed()

				all_batch_labels[:, batch] = batch_labels
				#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

				_ , batch_cost, batch_prediction = sess.run([optimizer, cost, predictions], feed_dict={inputs_placeholder: padded_batch_articles_ids, masks_placeholder: batch_masks, scores_placeholder: batch_labels, embedding_placeholder: embeddings})
				all_batch_predictions[:, batch] = batch_prediction
				#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

				epoch_cost += batch_cost / num_batches

#				tf.assign(all_batch_predictions[:, batch], batch_prediction)

			#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

			if epoch % 10 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if epoch % 5 == 0:
				costs.append(epoch_cost)

	#mean squared error
	#L2 loss


	similarity_threshold = 0.1
	correctly_scored_count = 0
	#import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

	score_differences = abs(all_batch_labels - all_batch_predictions)
	correctly_scored_count = np.sum(score_differences < accuracy_threshold)
	#print(score_differences)
	#print(correctly_scored_count)
	#print(len(all_batch_predictions))

	performance = tf.divide(correctly_scored_count,len(all_batch_predictions))

	print "Performance = " + str(performance)
	#print ("Train Accuracy:", accuracy.eval({inputs_placeholder: X_train, scores_placeholder: Y_train}))
	#print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
	#cast back to tensor?

	# compare batch_predictions with all_batch_labels for accuracy
	# TO DO: add accuracy calculations


#TO DO: write results of load_glove, build_dictionaries, create_embeddings to file once and read in
glove_vocab, glove_to_embedding, embedding_dim = load_glove()
#emb_matrix, word2id, id2word, char_emb_matrix, char2id, id2char = get_glove(embedding_dim)

#EDIT THISSSSSS
#x_train, y_train = form_data_set('train')
#x_test, y_test = form_data_set('test')
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
