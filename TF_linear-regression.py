"""

@Author : AweSong-Kor
@Github : AweSong-Kor/Python

"""

import tensorflow as tf

# X and Y data
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Linear Regression 
# Y_ = W*x + b
Y_ = X * W + b
cost = tf.reduce_mean(tf.square(Y_ - Y))

# Gradient Descent
# Minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Make a session
sess = tf.Session()
# Initializes global variables
sess.run(tf.global_variables_initializer())

# Launch the graph
for step in range(0001):
	cost_val, W_val, b_val, _ = \
		sess.run([cost,W,b,train],feed_dict={X: [1,2,3], Y: [2.1,3.1,4.1]})
	if step % 20 == 0:
		print(step, cost_val, W_val, b_val)

