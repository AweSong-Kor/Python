# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 16:44:51 2016

@author: HyunMin-Kor
"""
import DataLoader
datasets = DataLoader.read_data_sets('/home/gnos/work/data/')

import tensorflow as tf

# Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W, strides=1):
    if strides == 1:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    elif strides == 2:
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
    else:
        print ("strides must be 1 or 2")

def conv2d_V(x, W, strides=1):
    if strides == 1:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    elif strides == 2:
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')
    else:
        print ("strides must be 1 or 2")
        
def max_pool_3x3_V(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')



# Placeholders
# x is input, y_ is true label
x_w = 29            # width of input data
x_h = 29            # height of input data
x_dim = x_w * x_h   # dimensionality of a single flattened 256 x 256 matrix data
y__dim = 2         # number of digit classes
x = tf.placeholder(tf.float32, shape=[None, x_w, x_h])
y_ = tf.placeholder(tf.float32, shape=[None, y__dim])

################################
#                              #
#             Stem             #
#                              #
################################
# 1-st Convolutional Layer
x_matrix = tf.reshape(x, [-1, x_w, x_h, 1])

Stem_W_conv1 = weight_variable([3, 3, 1, 32])
Stem_b_conv1 = bias_variable([32])
Stem_h_conv1 = tf.nn.relu(conv2d(x_matrix, Stem_W_conv1)+Stem_b_conv1)

# 2-nd Conv Layer
Stem_W_conv2 = weight_variable([3, 3, 32, 32])
Stem_b_conv2 = bias_variable([32])
Stem_h_conv2 = tf.nn.relu(conv2d_V(Stem_h_conv1, Stem_W_conv2)+Stem_b_conv2)

# Max Pooling
Stem_h_pool1 = max_pool_3x3_V(Stem_h_conv2)

# 4-th 1x1 Conv Layer
Stem_W_conv3 = weight_variable([1, 1, 32, 64])
Stem_b_conv3 = bias_variable([64])
Stem_h_conv3 = tf.nn.relu(conv2d(Stem_h_pool1, Stem_W_conv3)+Stem_b_conv3)

# 5-th Conv Layer
Stem_W_conv4 = weight_variable([3, 3, 64, 128])
Stem_b_conv4 = bias_variable([128])
Stem_h_conv4 = tf.nn.relu(conv2d(Stem_h_conv3, Stem_W_conv4)+Stem_b_conv4)

# 6-th Conv Layer
Stem_W_conv5 = weight_variable([3, 3, 128, 128])
Stem_b_conv5 = bias_variable([128])
Stem_h_conv5 = tf.nn.relu(conv2d(Stem_h_conv4, Stem_W_conv5)+Stem_b_conv5)

################################
#                              #
#      Inception-resnet-A      #
#                              #
################################
# line 1, 1-st 1x1 Conv Layer
InCResA_1_W_conv11 = weight_variable([1, 1, 128, 32])
InCResA_1_b_conv11 = bias_variable([32])
InCResA_1_h_conv11 = tf.nn.relu(conv2d(Stem_h_conv5, InCResA_1_W_conv11)+InCResA_1_b_conv11)

# line 2, 1-st 1x1 Conv Layer
InCResA_1_W_conv21 = weight_variable([1, 1, 128, 32])
InCResA_1_b_conv21 = bias_variable([32])
InCResA_1_h_conv21 = tf.nn.relu(conv2d(Stem_h_conv5, InCResA_1_W_conv21)+InCResA_1_b_conv21)

# line 2, 2-nd 3x3 Conv Layer
InCResA_1_W_conv22 = weight_variable([3, 3, 32, 32])
InCResA_1_b_conv22 = bias_variable([32])
InCResA_1_h_conv22 = tf.nn.relu(conv2d(InCResA_1_h_conv21, InCResA_1_W_conv22)+InCResA_1_b_conv22)

# line 3, 1-st 1x1 Conv Layer
InCResA_1_W_conv31 = weight_variable([1, 1, 128, 32])
InCResA_1_b_conv31 = bias_variable([32])
InCResA_1_h_conv31 = tf.nn.relu(conv2d(Stem_h_conv5, InCResA_1_W_conv31)+InCResA_1_b_conv31)

# line 3, 2-nd 3x3 Conv Layer
InCResA_1_W_conv32 = weight_variable([3, 3, 32, 32])
InCResA_1_b_conv32 = bias_variable([32])
InCResA_1_h_conv32 = tf.nn.relu(conv2d(InCResA_1_h_conv31, InCResA_1_W_conv32)+InCResA_1_b_conv32)

# line 3, 3-rd 3x3 Conv Layer
InCResA_1_W_conv33 = weight_variable([3, 3, 32, 32])
InCResA_1_b_conv33 = bias_variable([32])
InCResA_1_h_conv33 = tf.nn.relu(conv2d(InCResA_1_h_conv32, InCResA_1_W_conv33)+InCResA_1_b_conv33)

# Linear 1x1 Conv Layer without Relu
InCResA_1_W_convOut = weight_variable([1, 1, 96, 128])
InCResA_1_b_convOut = bias_variable([128])
InCResA_1_h_convOut = conv2d(tf.concat(3, [InCResA_1_h_conv11, InCResA_1_h_conv22,InCResA_1_h_conv33]), InCResA_1_W_convOut)+InCResA_1_b_convOut

# Sum
InCResA_1_out = tf.nn.relu(tf.add(Stem_h_conv5,InCResA_1_h_convOut))

################################
#                              #
#          Reduction-A         #
#                              #
################################
# line 1, 3x3 Max Pooling Layer
ReductionA_h_pool11 = max_pool_3x3_V(InCResA_1_out)

# line 2, Conv Layer
ReductionA_W_conv21 = weight_variable([3, 3, 128, 192])
ReductionA_b_conv21 = bias_variable([192])
ReductionA_h_conv21 = tf.nn.relu(conv2d_V(InCResA_1_out, ReductionA_W_conv21, strides=2)+ReductionA_b_conv21)

# line3, 1-st Conv Layer
ReductionA_W_conv31 = weight_variable([1, 1, 128, 96])
ReductionA_b_conv31 = bias_variable([96])
ReductionA_h_conv31 = tf.nn.relu(conv2d(InCResA_1_out,ReductionA_W_conv31)+ReductionA_b_conv31)

# line3, 2-nd Conv Layer
ReductionA_W_conv32 = weight_variable([3, 3, 96, 96])
ReductionA_b_conv32 = bias_variable([96])
ReductionA_h_conv32 = tf.nn.relu(conv2d(ReductionA_h_conv31,ReductionA_W_conv32)+ReductionA_b_conv32)

# line3, 3-nd Conv Layer
ReductionA_W_conv33 = weight_variable([3, 3, 96, 128])
ReductionA_b_conv33 = bias_variable([128])
ReductionA_h_conv33 = tf.nn.relu(conv2d_V(ReductionA_h_conv32,ReductionA_W_conv33,strides=2)+ReductionA_b_conv33)

ReductionA_out = tf.concat(3, [ReductionA_h_pool11, ReductionA_h_conv21, ReductionA_h_conv33])

################################
#                              #
#      Inception-resnet-B      #
#                              #
################################
# line 1, 1-st Conv Layer
InCResB_1_W_conv11 = weight_variable([1, 1, 448, 64])
InCResB_1_b_conv11 = bias_variable([64]) 
InCResB_1_h_conv11 = tf.nn.relu(conv2d(ReductionA_out, InCResB_1_W_conv11)+InCResB_1_b_conv11)

# line 2, 1-st Conv Layer
InCResB_1_W_conv21 = weight_variable([1, 1, 448, 64])
InCResB_1_b_conv21 = bias_variable([64]) 
InCResB_1_h_conv21 = tf.nn.relu(conv2d(ReductionA_out, InCResB_1_W_conv21)+InCResB_1_b_conv21)

# line 2, 2-nd Conv Layer
InCResB_1_W_conv22 = weight_variable([1, 3, 64, 64])
InCResB_1_b_conv22 = bias_variable([64]) 
InCResB_1_h_conv22 = tf.nn.relu(conv2d(InCResB_1_h_conv21, InCResB_1_W_conv22)+InCResB_1_b_conv22)

# line 2, 3-rd Conv Layer
InCResB_1_W_conv23 = weight_variable([3, 1, 64, 64])
InCResB_1_b_conv23 = bias_variable([64]) 
InCResB_1_h_conv23 = tf.nn.relu(conv2d(InCResB_1_h_conv22, InCResB_1_W_conv23)+InCResB_1_b_conv23)

# Linear 1x1 Conv Layer without Relu
InCResB_1_W_convOut = weight_variable([1, 1, 128, 448])
InCResB_1_b_convOut = bias_variable([448])
InCResB_1_h_convOut = conv2d(tf.concat(3, [InCResB_1_h_conv11, InCResB_1_h_conv23]), InCResB_1_W_convOut)+InCResB_1_b_convOut

# Sum
InCResB_1_out = tf.nn.relu(tf.add(ReductionA_out,InCResB_1_h_convOut))

################################
#                              #
#          Reduction-B         #
#                              #
################################
# line 1, 3x3 Max Pool Layer
ReductionB_h_pool1 = max_pool_3x3_V(InCResB_1_out)

# line 2, 1-st Conv Layer
ReductionB_W_conv21 = weight_variable([1, 1, 448, 128])
ReductionB_b_conv21 = bias_variable([128])
ReductionB_h_conv21 = tf.nn.relu(conv2d(InCResB_1_out,ReductionB_W_conv21)+ReductionB_b_conv21)

# line 2, 2-nd Conv Layer
ReductionB_W_conv22 = weight_variable([3, 3, 128, 192])
ReductionB_b_conv22 = bias_variable([192])
ReductionB_h_conv22 = tf.nn.relu(conv2d_V(ReductionB_h_conv21,ReductionB_W_conv22, strides=2)+ReductionB_b_conv22)

# line 3, 1-st Conv Layer
ReductionB_W_conv31 = weight_variable([1, 1, 448, 128])
ReductionB_b_conv31 = bias_variable([128])
ReductionB_h_conv31 = tf.nn.relu(conv2d(InCResB_1_out,ReductionB_W_conv31)+ReductionB_b_conv31)

# line 3, 2-nd Conv Layer
ReductionB_W_conv32 = weight_variable([3, 3, 128, 128])
ReductionB_b_conv32 = bias_variable([128])
ReductionB_h_conv32 = tf.nn.relu(conv2d_V(ReductionB_h_conv31,ReductionB_W_conv32, strides=2)+ReductionB_b_conv32)

# line 4, 1-st Conv Layer
ReductionB_W_conv41 = weight_variable([1, 1, 448, 128])
ReductionB_b_conv41 = bias_variable([128])
ReductionB_h_conv41 = tf.nn.relu(conv2d(InCResB_1_out,ReductionB_W_conv41)+ReductionB_b_conv41)

ReductionB_W_conv42 = weight_variable([3, 3, 128, 128])
ReductionB_b_conv42 = bias_variable([128])
ReductionB_h_conv42 = tf.nn.relu(conv2d(ReductionB_h_conv41,ReductionB_W_conv42)+ReductionB_b_conv42)

ReductionB_W_conv43 = weight_variable([3, 3, 128, 128])
ReductionB_b_conv43 = bias_variable([128])
ReductionB_h_conv43 = tf.nn.relu(conv2d_V(ReductionB_h_conv42,ReductionB_W_conv43, strides=2)+ReductionB_b_conv43)

# Filter Concat
ReductionB_out = tf.concat(3, [ReductionB_h_pool1, ReductionB_h_conv22, ReductionB_h_conv32, ReductionB_h_conv43])

################################
#                              #
#      Inception-resnet-C      #
#                              #
################################
# line 1, 1-st Conv Layer
InCResC_1_W_conv11 = weight_variable([1, 1, 896, 96])
InCResC_1_b_conv11 = bias_variable([96]) 
InCResC_1_h_conv11 = tf.nn.relu(conv2d(ReductionB_out, InCResC_1_W_conv11)+InCResC_1_b_conv11)

# line 2, 1-st Conv Layer
InCResC_1_W_conv21 = weight_variable([1, 1, 896, 96])
InCResC_1_b_conv21 = bias_variable([96]) 
InCResC_1_h_conv21 = tf.nn.relu(conv2d(ReductionB_out, InCResC_1_W_conv21)+InCResC_1_b_conv21)

# line 2, 2-nd Conv Layer
InCResC_1_W_conv22 = weight_variable([1, 3, 96, 96])
InCResC_1_b_conv22 = bias_variable([96]) 
InCResC_1_h_conv22 = tf.nn.relu(conv2d(InCResC_1_h_conv21, InCResC_1_W_conv22)+InCResC_1_b_conv22)

# line 2, 3-rd Conv Layer
InCResC_1_W_conv23 = weight_variable([3, 1, 96, 96])
InCResC_1_b_conv23 = bias_variable([96]) 
InCResC_1_h_conv23 = tf.nn.relu(conv2d(InCResC_1_h_conv22, InCResC_1_W_conv23)+InCResC_1_b_conv23)

# Linear 1x1 Conv Layer without Relu
InCResC_1_W_convOut = weight_variable([1, 1, 192, 896])
InCResC_1_b_convOut = bias_variable([896])
InCResC_1_h_convOut = conv2d(tf.concat(3, [InCResC_1_h_conv11, InCResC_1_h_conv23]), InCResC_1_W_convOut)+InCResC_1_b_convOut

# Sum
InCResC_1_out = tf.nn.relu(tf.add(ReductionB_out,InCResC_1_h_convOut))

################################
#                              #
#         Avg. Pooling         #
#           Drop Out           #
#           SoftMax            #
#                              #
################################

# Average Pooling
h_avg_pool = avg_pool(InCResC_1_out)
h_pool_flat = tf.reshape(h_avg_pool, [-1, 896])

# Drop Out
keep_prob = tf.placeholder(tf.float32)
h_pool_drop = tf.nn.dropout(h_pool_flat, keep_prob)

# Readout Layer
W_out = weight_variable([896, 2])
b_out = bias_variable([2])
y_out = tf.matmul(h_pool_drop, W_out) + b_out

# SoftMax
# Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out,y_))

################################
#                              #
#       Training & Test        #
#                              #
################################

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Train the Model
# load 100 training examples in each training iteration
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Luance the Model
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(50000):
        batch = datasets.train.next_batch(10)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" %(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.8})
     
    print("test accuracy %g" %accuracy.eval(feed_dict={x: datasets.test.data, y_: datasets.test.labels, keep_prob:1.0})) 
    
    # Save the variables to disk.
    save_path = saver.save(sess, "/home/gnos/work/model/model.ckpt")
    print("Model saved in file: %s" % save_path)    


