######################################################################################################################################################################
###### 			Builds & execute the tensorflow RNN. 		##########################################################################################################
######################################################################################################################################################################


from model_functions import get_input_data_as_ids, create_placeholders, get_cost, get_batch_from_folder, run_and_eval_dev
from preprocess import preprocess, load_glove, get_input_data, build_dictionaries, create_embeddings, get_input_labels
import tensorflow as tf
import io
import numpy as np
import sys

def model(max_article_length, embedding_dim, vocab_size, embeddings, wordToID,
		  num_epochs = 10, batch_size = 10, num_batches = 1, num_hidden_units = 50, learning_rate = 0.001, folder_name="train"):

###########    Sets up the model's infrastructure.    #########################################

	#Creates Placeholders for the Tensorflow model.
	inputs_placeholder, masks_placeholder, scores_placeholder, embedding_placeholder = create_placeholders(max_article_length, batch_size, vocab_size, embedding_dim)

	#Gets the embedded characters for the inputs
	embedded_chars = tf.nn.embedding_lookup(embedding_placeholder, inputs_placeholder)

	#Layer 1: LSTM
	rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units)
	#Produces output cells of shape = (batch_size, max_article_length, num_hidden_units)
	outputs, _ = tf.nn.dynamic_rnn(rnn_cell, embedded_chars, sequence_length = [max_article_length]*batch_size, dtype=tf.float32)


	#Pads outputs to standardize each vector's length for matrix multiplication
	masks_ = tf.reshape(masks_placeholder, shape=[batch_size, max_article_length, 1])
	padded_outputs = tf.multiply(outputs, masks_)

	#Calculates predictions and cost on input data
	output = tf.reduce_max(padded_outputs, axis=1)
	predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn = tf.nn.sigmoid)
	cost = get_cost(predictions, scores_placeholder)

	#Initiates Adam Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	# Creates initializer to initialize all variables in the Tensorflow graph
	init = tf.global_variables_initializer()

###########    Starts the session; Executes the graph's computations    #########################################

	with tf.Session() as sess:

		# Runs the initializer
		sess.run(init)

		# Does the training loop
		for epoch in range(num_epochs):

			epoch_cost, dev_epoch_cost = 0.0, 0.0

			#Creates a batch generator that takes batches from folder folder_name ("train", "dev", "test") -- data has been pre-divided randomly into a 70/20/10 train/dev/test ratio
			batches = get_batch_from_folder(max_article_length, folder_name, batch_size, num_batches, wordToID)

			for batch in range(num_batches):
				padded_batch_articles_ids, batch_labels, batch_masks = batches.next()

				#Runs the session
				_ , batch_cost, batch_predictions = sess.run([optimizer, cost, predictions], feed_dict={inputs_placeholder: padded_batch_articles_ids, masks_placeholder: batch_masks, scores_placeholder: batch_labels, embedding_placeholder: embeddings})

				#Calculates performance of the batch predictions using the function:
					# 1[|batch_labels - batch_predictions| < threshold]
				similarity_threshold = 0.1
				correctly_scored_count = 0
				score_differences = abs(batch_labels - batch_predictions)
				correctly_scored_count = np.sum(score_differences < similarity_threshold)
				performance = tf.divide(correctly_scored_count,len(batch_predictions))


				#Outputs the performance of the model
				print "Count of correct scores for training batch " + str(batch) + ": " + str(correctly_scored_count)
				sys.stdout.flush()
				print "Performance for training batch " + str(batch) + ": " + str(performance)
				sys.stdout.flush()


				epoch_cost += batch_cost / num_batches

			#Evaluate the dev set
			dev_epoch_cost = run_and_eval_dev(sess, max_article_length, "test", 100, 50, wordToID, embeddings, cost, \
					predictions, inputs_placeholder, masks_placeholder, scores_placeholder, embedding_placeholder)

		    #Outputs the costs of the training and dev/test sets
			print ("Training cost after epoch %i: %f" % (epoch, epoch_cost))
			sys.stdout.flush()
			print ("Test cost after epoch %i: %f" % (epoch, dev_epoch_cost))
			sys.stdout.flush()


######################################################################################################################################################################
###### 			MAIN FUNCTION   ~~ PREPROCESSES DATA, BUILDS, AND RUNS THE MODEL		##############################################################

def run_model():
	max_article_length, embedding_dim, vocab_size, embeddings, word2id = preprocess()
	model(max_article_length, embedding_dim, vocab_size, embeddings, word2id)


if __name__ == "__main__":
	run_model()
