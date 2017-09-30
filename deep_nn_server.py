from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time
import math
import os
from socket import *
import pickle

import random

import numpy as np


import datetime
import tensorflow as tf
from client_agent1 import *
from tensorflow.contrib.session_bundle.exporter import classification_signature
from tensorflow.python.saved_model import builder as saved_model_builder, tag_constants, signature_constants


# finalNN=[]
# lastEpoch=False

serverPort = 12000
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('', serverPort))
serverSocket.listen(1)

tf.app.flags.DEFINE_integer('training_iteration', 1,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/student/Desktop/', 'Working directory.')
FLAGS = tf.app.flags.FLAGS
#
# def deepnn(x):
#     with tf.name_scope('reshape'):
#         x_image = tf.reshape(x, [-1, 52, 52, 1])
#
#     # First convolutional layer - maps one grayscale image to 32 feature maps.
#     with tf.name_scope('conv1'):
#         W_conv1 = weight_variable([5, 5, 1, 32]) #feature size 5x5 to have 46x46 image after convolution
#         b_conv1 = bias_variable([32]) #32 feature maps - arbitrary - can change
#         h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #uses max function (instead of sigmoid function)
#
#     # Pooling layer - downsamples by 2X.
#     with tf.name_scope('pool1'):
#         h_pool1 = max_pool_2x2(h_conv1) #now size will be 23x23x32
#
#     # Second convolutional layer -- maps 32 feature maps to 64.
#     with tf.name_scope('conv2'):
#         W_conv2 = weight_variable([4, 4, 32, 64]) #feature size 4x4 to have 20x20 image after convolution
#         b_conv2 = bias_variable([64])
#         h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#
#     # Second pooling layer.
#     with tf.name_scope('pool2'):
#         h_pool2 = max_pool_2x2(h_conv2) #now size will be 10x10x64
#
#     # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
#     # is down to 10x10x64 feature maps -- maps this to 1024 features.
#     with tf.name_scope('fc1'):
#         W_fc1 = weight_variable([10 * 10 * 64, 4096]) #4096 = first power of 2 larger than 2500 (=50x50)
#         b_fc1 = bias_variable([4096])
#
#         h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * 64])
#         h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#     # Dropout - controls the complexity of the model, prevents co-adaptation of
#     # features.
#     # with tf.name_scope('dropout'): #maybe add dropout to other layers aswell?
#     #     keep_prob = tf.placeholder(tf.float32)
#     #     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#     # Map the 4096 features to 3 classes, one for each direction
#     with tf.name_scope('fc2'):
#         W_fc2 = weight_variable([4096, 3])
#         b_fc2 = bias_variable([3])
#
#         # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#         y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
#
#     # regularizer = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
#     # regularizer = tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
#
#     return y_conv#, keep_prob, regularizer
#
# def conv2d(x, W):
#   """conv2d returns a 2d convolution layer with full stride."""
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') #VALID = no padding
#
#
# def max_pool_2x2(x):
#   """max_pool_2x2 downsamples a feature map by 2X."""
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='VALID') #VALID = no padding
#
# def weight_variable(shape):
#   """weight_variable generates a weight variable of a given shape."""
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)
#
#
# def bias_variable(shape):
#   """bias_variable generates a bias variable of a given shape."""
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)
#
# def main(_):
#     startTime = time.time()
#
#     # Create the model
#     x = tf.placeholder(tf.float32, [None, 2704])
#     # y_ = tf.placeholder(tf.float32, [None, 3])
#
#     y = deepnn(x)
#     result = tf.argmax(y,output_type=tf.int32)

    # Add ops to save and restore all the variables.


with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver()
    #import_path = "/home/student/Desktop/saved_models/model" + str(FLAGS.model_version) + ".meta"
    #saver = tf.train.import_meta_graph(import_path)
    # saver.restore(sess, tf.train.latest_checkpoint('./'))
    #saver.restore(sess, import_path)

    saver = tf.train.import_meta_graph('output_snake_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # saver.restore(sess,"output_snake_model.data-00000-of-00001")

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    # y_ = graph.get_tensor_by_name("y_:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    result_argmax = graph.get_tensor_by_name("result_argmax:0")

    # Making connection
    print("Waiting for connection... ")
    connectSocket, addr = serverSocket.accept()
    while 1:

        print("Connection established from " + str(addr))
        mapReceived = ''
        expectedTotalData = connectSocket.recv(5)
        if expectedTotalData:
            print ("Expected total data size: " + expectedTotalData)
            expectedTotalData = int(expectedTotalData)
            expectedTotalDataLeft = int(expectedTotalData)
            while expectedTotalDataLeft > 0:
                mapPacket = connectSocket.recv(expectedTotalData)
                expectedTotalDataLeft = expectedTotalDataLeft - sys.getsizeof(mapPacket)
                if not mapPacket:
                    break
                mapReceived = mapReceived + mapPacket
        else:
            print("connection lost at: " + str(datetime.datetime.now()) + " - closing and reopening connection...")
            connectSocket.close()
            print("Waiting for connection... ")
            connectSocket, addr = serverSocket.accept()
            ##############################################################
            # After Crash if client sends data too quickly after establishing connection server will crash
            ##############################################################
            continue

        print("mapReceived before pickle loads: " + str(mapReceived))
        map = pickle.loads(mapReceived)
        print("Received map, sending it through deepnn")
        result_arr = sess.run(result_argmax, feed_dict={x: map.reshape(1,2704), keep_prob: 1.0})
        # answer = result.eval(feed_dict={x: map.reshape(1,2704)})
        # result = tf.argmax(deepnn(map),axis=0, output_type=tf.int32)
        result = str(result_arr[0] + 4)
        print("Obtained result " + result + " from deepnn. Pickling it now.")
        # returnResult = pickle.dumps(result)
        print("Sending result to client...")
        connectSocket.send(result)
        print("Done serving client, reseting for new client")

    print("Done sending. Closing Connection...")
    connectSocket.close()

    # Build the graph for the deep net

    #y_conv, _, _ = deepnn(playGame())



if __name__ == '__main__':

    tf.app.run(main=main)
