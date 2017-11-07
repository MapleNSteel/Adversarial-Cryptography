import tensorflow as tf

sess = tf.Session()
#new_saver = tf.train.import_meta_graph('XOR-100.meta')
#new_saver.restore(sess, tf.train.latest_checkpoint('./'))

x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")
Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Theta2")

Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")#

A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
        ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

tf.add_to_collection('train_step', train_step)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
        sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

	if i % 10 == 0:
		print('Epoch ', i)
		print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
		print('Theta1 ', sess.run(Theta1))
		print('Bias1 ', sess.run(Bias1))
		print('Theta2 ', sess.run(Theta2))
		print('Bias2 ', sess.run(Bias2))
		print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))

saver = tf.train.Saver()
saver.save(sess, 'XOR', 100)
