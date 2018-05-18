from model_functions import get_input_data_as_ids, get_batch, create_placeholders, get_cost
from preprocess import load_glove, get_input_data, build_dictionaries, create_embeddings, get_input_labels
import tensorflow as tf
import io

def model(input_data_as_ids, input_scores, X_test, Y_test, embedding_dim, vocab_size, embedding, wordToID,
		  num_epochs = 1000, batch_size = 10, num_batches = 1, num_hidden_units = 20, learning_rate = 0.0001):

	costs = []  
	inputs_placeholder, scores_placeholder, masks_placeholder, embedding_placeholder = create_placeholders(vocab_size, embedding_dim)

	# RNN output node weights and biases
	# use tf.get_variable
	weights = { 'out': tf.Variable(tf.random_normal([num_hidden_units, embedding_dim])) }
	biases = { 'out': tf.Variable(tf.random_normal([embedding_dim])) }

	embedded_chars = tf.nn.embedding_lookup(embedding_placeholder, inputs_placeholder)
	embedded_chars_unstack = tf.unstack(embedded_chars, axis=1)

	rnn_cell = tf.contrib.rnn.BasicRNNCell(num_hidden_units)
	outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, embedded_chars_unstack, dtype=tf.float32)

	#convert all cells with <PAD> to -inf (if max) or 0 if avg
	# mask should have 1s where <PAD> exists if multiplying by -inf
	output = tf.reduce_max(states, axis=1)

	predictions = tf.matmul(output, weights['out']) + biases['out']
	predictions = tf.sigmoid(predictions)
	
	cost = get_cost(predictions, scores_placeholder)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	# Initialize all the variables
	init = tf.global_variables_initializer()

	# Start the session to compute the tensorflow graph
	with tf.Session() as sess:

		# Run the initialization
		sess.run(init)

		batch_predictions = []
		all_batch_labels = []
		# Do the training loop
		for epoch in range(num_epochs):

			# add loop to get next batch
			epoch_cost = 0.
			# when we add more batches, loop through them here
			batches = get_batch(input_data_as_ids, input_scores, batch_size, num_batches, wordToID)
			batch_articles_ids, batch_labels, batch_mask = batches.next()
			all_batch_labels.append(batch_labels)

			_ , batch_cost, batch_prediction = sess.run([optimizer, cost, predictions], feed_dict={inputs_placeholder: batch_articles_ids, scores_placeholder: batch_labels, embedding_placeholder: embedding})
			epoch_cost += batch_cost / num_batches
			batch_predictions.append(batch_prediction)

		if print_cost == True and epoch % 100 == 0:
			print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
		if print_cost == True and epoch % 5 == 0:
			costs.append(epoch_cost)

	# compare batch_predictions with all_batch_labels for accuracy
	# TO DO: add accuracy calculations 



total_num_articles = 10
#TO DO: write results of load_glove, build_dictionaries, create_embeddings to file once and read in
glove_vocab, embedding_dict, embedding_dim = load_glove()
input_data = get_input_data(total_num_articles)
input_scores = get_input_labels(total_num_articles)

all_articles_words = []
for article in input_data:
	all_articles_words.extend(article)
all_articles_words.append('<PAD>')
wordToID, idToWord, vocab_size = build_dictionaries(all_articles_words)
input_data_as_ids = get_input_data_as_ids(wordToID, input_data)
embedding = create_embeddings(vocab_size, glove_vocab, wordToID, embedding_dict, embedding_dim)

X_test = input_data_as_ids
Y_test = input_scores

model(input_data_as_ids, input_scores, X_test, Y_test, embedding_dim, vocab_size, embedding, wordToID)


