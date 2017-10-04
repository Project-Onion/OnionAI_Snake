from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import random

import numpy as np

import tensorflow as tf

tf.app.flags.DEFINE_integer('training_iteration', 1,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/student/Desktop/', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def deepnn(x, keep_prob):
    epsilon = 1e-3
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 52, 52, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])  # feature size 5x5 to have 46x46 image after convolution
        Z_conv1 = conv2d(x_image, W_conv1)
        # b_conv1 = bias_variable([32]) #32 feature maps - arbitrary - can change
        #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))  # + b_conv1) #uses max function (instead of sigmoid function)
        # h_conv1 = tf.nn.relu(conv2d(h_fc0_flat, W_conv1) + b_conv1)
        batch_mean1, batch_var1 = tf.nn.moments(Z_conv1, [0])
        scale2 = tf.Variable(tf.ones([32]))
        beta2 = tf.Variable(tf.zeros([32]))
        BN1 = tf.nn.batch_normalization(Z_conv1, batch_mean1, batch_var1, beta2, scale2, epsilon)
        h_conv1 = tf.nn.relu(BN1)  # + b_conv1) #uses max function (instead of sigmoid function)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # now size will be 23x23x32

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([4, 4, 32, 64])  # feature size 4x4 to have 20x20 image after convolution
        Z_conv2 = conv2d(h_pool1, W_conv2)
        # b_conv2 = bias_variable([64])
        # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        batch_mean2, batch_var2 = tf.nn.moments(Z_conv2, [0])
        scale2 = tf.Variable(tf.ones([64]))
        beta2 = tf.Variable(tf.zeros([64]))
        BN2 = tf.nn.batch_normalization(Z_conv2, batch_mean2, batch_var2, beta2, scale2, epsilon)
        h_conv2 = tf.nn.relu(BN2)  # + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # now size will be 10x10x64

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 10x10x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([10 * 10 * 64, 4096])  # 4096 = first power of 2 larger than 2500 (=50x50)
        b_fc1 = bias_variable([4096])

        # h_pool2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * 64])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    # with tf.name_scope('dropout'): #maybe add dropout to other layers aswell?
    #     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 4096 features to 3 classes, one for each direction
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([4096, 3])
        b_fc2 = bias_variable([3])

        # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    # regularizer = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
    # regularizer = tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)

    return y_conv, keep_prob  # , regularizer

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') #VALID = no padding


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID') #VALID = no padding

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
    startTime = time.time()

    mini_batch_amount = 123123
    amountOfMiniBatchFilesToTrain = 100
    amountOfMiniBatchFilesToValidate = 50
    amountOfMiniBatchFilesToTest = 50

    print ("amountOfMiniBatchFilesToTrain: " + str(amountOfMiniBatchFilesToTrain) + "\tamountOfMiniBatchFilesToValidate: " + str(amountOfMiniBatchFilesToValidate) + "\tamountOfMiniBatchFilesToTest: " + str(amountOfMiniBatchFilesToTest))
    fileLocation = "/home/student/Desktop/"

    # Create the model
    x = tf.placeholder(tf.float32, [None, 2704], name="x")

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3], name="y")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    # Build the graph for the deep net
    # y_conv, keep_prob, regularizer = deepnn(x) # with l2 regularization
    y_conv, keep_prob= deepnn(x, keep_prob)

    tf.argmax(y_conv, 1, output_type=tf.int32, name="result_argmax")

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    # beta = 0.01 #lambda
    cross_entropy = tf.reduce_mean(cross_entropy)# + beta*regularizer)

    with tf.name_scope('adam_optimizer'): #adam replaces gradient decent
        learning_rate = 1e-3#tf.placeholder(tf.float32, shape=[])
        # train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        print ("adam with learning rate: " + str(learning_rate))

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    mini_batch_size = 1000
    print ("mini batch size: "+str(mini_batch_size))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        numEpochs = 1
        for epoch in range(numEpochs):
            # mini_batches = [[np.stack(numpyCombinedData[0][rearrange][k:k + mini_batch_size]), np.stack(numpyCombinedData[1][rearrange][k:k + mini_batch_size])]
            #                 for k in range(0, (int)(numpyCombinedData[0].shape[0] * train_data_percent), mini_batch_size)]

            print ("epoch " + str(epoch+1) + " started training. time passed: "+ str(time.time()-startTime))
            for i in range(0,amountOfMiniBatchFilesToTrain):
                batchData = [np.load(fileLocation + "trainingData/" + 'trainingDataBoards' + str(i) + ".npy"),
                             np.load(fileLocation + "trainingData/" + 'trainingDataMoves' + str(i) + ".npy")]
                rearrange = np.array(range(len(batchData[0])))
                np.random.shuffle(rearrange)
                for batchStartIndex in range(0,len(batchData[0]), mini_batch_size):
                    batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
                    train_step.run(
                        feed_dict={x: batchData[0][rearrange][batchStartIndex:batchEndIndex],
                                   y_: batchData[1][rearrange][batchStartIndex:batchEndIndex],
                                   keep_prob: 0.5})#, learning_rate: 0.00003 }) #math.exp(-1*math.sqrt(epoch)*sumOfValidations*10)})

            print("epoch " + str(epoch+1) + " started validation. time passed: "+ str(time.time()-startTime))
            sumOfValidations = 0
            for i in range(0, amountOfMiniBatchFilesToValidate):
                batchData = [np.load(fileLocation + "validationData/" + 'validationDataBoards' + str(i) + ".npy"),
                             np.load(fileLocation + "validationData/" + 'validationDataMoves' + str(i) + ".npy")]
                amountOfValidations = 0
                for batchStartIndex in range(0, len(batchData[0]), mini_batch_size):
                    batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
                    validate_accuracy = accuracy.eval(feed_dict={
                        x: batchData[0][batchStartIndex: batchEndIndex],
                        y_: batchData[1][batchStartIndex:batchEndIndex], keep_prob: 1.0})
                    # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
                    sumOfValidations = sumOfValidations + validate_accuracy
                    amountOfValidations = amountOfValidations + 1
                    #print (validate_accuracy)

            print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t epoch "+ str(epoch+1) + "/" +str(numEpochs) +  ": "  + str(sumOfValidations/amountOfValidations))

        trainEndTime = time.time()
        print ("training and validation ended. \t time it took: " + str(trainEndTime - startTime))

        sumOfTests = 0
        for i in range(0, amountOfMiniBatchFilesToTest):
            batchData = [np.load(fileLocation + "testData/" + 'testDataBoards' + str(i) + ".npy"),
                         np.load(fileLocation + "testData/" + 'testDataMoves' + str(i) + ".npy")]
            amountOfTests = 0
            for batchStartIndex in range(0, len(batchData[0]), mini_batch_size):
                batchEndIndex = batchStartIndex + min(mini_batch_size, len(batchData[0]) - batchStartIndex)
                test_accuracy = accuracy.eval(feed_dict={
                    x: batchData[0][batchStartIndex:batchEndIndex],
                    y_: batchData[1][batchStartIndex:batchEndIndex], keep_prob: 1.0})
                # print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
                sumOfTests = sumOfTests + test_accuracy
                amountOfTests = amountOfTests + 1

        print("test accuracy: " + str(sumOfTests / amountOfTests))
        # print('test accuracy %g' % accuracy.eval(feed_dict={
        #     x: np.stack(numpyCombinedTestData[0]), y_: np.stack(numpyCombinedTestData[1]), keep_prob: 1.0}))

        testEndTime = time.time()
        print("testing ended. \t time for testing: " + str(testEndTime - trainEndTime) + "\t total time: "+ str(testEndTime - startTime))


        export_path = "/home/student/Desktop/saved_models/model" + str(FLAGS.model_version)#+".ckpt"
        #save_path = saver.save(sess, export_path)
        save_path = saver.save(sess, './models/output_snake_model')
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':

    tf.app.run(main=main)