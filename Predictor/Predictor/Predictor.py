import tensorflow as tf
import numpy as np
import data_reader

sequence_length = 3
batch_size = 3
cellHiddenLayerSize = 512

data = data_reader.read_training_data('TrainingData.txt')
dict, reverse_dict = data_reader.build_word_dictionary(data)
vocabSize = len(dict)

# multilayer cell construction.
cell1 = tf.contrib.rnn.BasicLSTMCell(cellHiddenLayerSize)
cell2 = tf.contrib.rnn.BasicLSTMCell(cellHiddenLayerSize)
multiCell = tf.contrib.rnn.MultiRNNCell(cells=[cell1, cell2])

# simple shape. batch size is None for dynamic purposes. 
# sequence length is the max sequence length to be processed. 
# 1 is the dimensionality of this data. one dimension for now. No one-hot encoding. 
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length, 1])
# None for batch again (see above). the vocabulary size is the row-count for the one-hot vector. 
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, vocabSize])

# create the operation for handling the cells. 
x1 = tf.split(tf.reshape(x, [-1, batch_size]), sequence_length, 1)
output, states = tf.contrib.rnn.static_rnn(cell=multiCell, inputs=x1, dtype=tf.float32)

# output layer. this will squash the values to the len(dict) size. 
weights = tf.Variable(tf.truncated_normal([cellHiddenLayerSize, vocabSize], stddev=.1, dtype=tf.float32))
biases = tf.Variable(tf.constant(value=.1, shape=[vocabSize]))
finalLayer = tf.matmul(output[-1], weights) + biases

# soft max'd cross entropy cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=finalLayer, labels=y))
# reduce i. 
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as session:
    # initialize all variables
    session.run(tf.global_variables_initializer())
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    for i in range(0, 50000):

        batch = data_reader.get_word_sequence_batch(data, batch_size, sequence_length)
        batchAsNumbers = data_reader.convert_batch_to_numbers(batch, dict)

        trainingData = []
        trainingLabels = []

        for b in range(0, batch_size):
            t = np.reshape(batchAsNumbers[b][0], newshape=[sequence_length, 1])
            trainingData.append(t)

            one_hot = np.zeros([vocabSize])
            one_hot[batchAsNumbers[b][1]] = 1
            trainingLabels.append(one_hot)

        trainingData = np.reshape(trainingData, [batch_size, sequence_length, 1])
        trainingLabels = np.reshape(trainingLabels, [batch_size, vocabSize])

        _, l = session.run([optimizer, finalLayer], feed_dict={x: trainingData, y: trainingLabels})

        if i % 100 == 0:

            for b in range(0, batch_size):
                print("=================================")
                before = ' '.join([a for a in batch[b][0]]).strip()
                expected = batch[b][1]

                predictionIndex = np.argmax(l[b])
                print(before + " " + expected + "|" + reverse_dict[predictionIndex])
                print("==================================")
                print("")

            saver.save(session, "/home/brush/training/model.ckpt")

print()

