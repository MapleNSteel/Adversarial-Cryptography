import numpy as np

import tensorflow as tf

import time
import sys

wordLength=keyLength=8

key_input_var = tf.placeholder(tf.float32, shape=[None, keyLength], name="key")
speaker_input_var = tf.placeholder(tf.float32, shape=[None, wordLength], name="speakerIn")
speaker_output_var = tf.placeholder(tf.float32, shape=[None, wordLength], name="speakerOut")
listener_output_var = tf.placeholder(tf.float32, shape=[None, wordLength], name="listenerOut")

numBatches=2**8

key=np.tile(np.float32(np.array([1,-1,1,1,1,-1,-1,-1])),(numBatches,1))

def createDenseLayer(input_tensor, input_dims=1, output_dims=1, layer_name='', act=tf.sigmoid):

	W = tf.Variable(tf.random_normal([input_dims, output_dims]))
	b = tf.Variable(tf.random_normal([output_dims]))

	return act(tf.matmul(input_tensor,W) + b)

def iterate_minibatches(inputs, batchsize, shuffle=False):
    inputs=inputs.astype(np.float32)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def createNN():
	
	#Speaker Start

	enDenseLayerOutput1 = createDenseLayer(input_tensor=tf.concat([speaker_input_var, key_input_var], 1), input_dims=keyLength+wordLength, output_dims=16, layer_name='enDenseLayerOutput1', act=tf.tanh)

	enDenseLayerOutput2 = createDenseLayer(input_tensor=enDenseLayerOutput1, input_dims=keyLength+wordLength, output_dims=32, layer_name='enDenseLayerOutput2', act=tf.tanh)

	enDenseLayerOutput3 = createDenseLayer(input_tensor=enDenseLayerOutput2, input_dims=32, output_dims=32, layer_name='enDenseLayerOutput3', act=tf.tanh)

	enDenseLayerOutput4 = createDenseLayer(input_tensor=enDenseLayerOutput3, input_dims=32, output_dims=16, layer_name='enDenseLayerOutput4', act=tf.tanh)

	encryptOutput = createDenseLayer(input_tensor=enDenseLayerOutput4, input_dims=keyLength+wordLength, output_dims=8, layer_name='encryptOutput', act=tf.tanh)
	speaker_output_var = encryptOutput
	#Speaker Stop

	#Listener Start
	deDenseLayerOutput1 = createDenseLayer(input_tensor=tf.concat([speaker_output_var, key_input_var], 1), input_dims=keyLength+wordLength, output_dims=16, layer_name='deDenseLayerOutput1', act=tf.tanh)

	deDenseLayerOutput2 = createDenseLayer(input_tensor=deDenseLayerOutput1, input_dims=keyLength+wordLength, output_dims=32, layer_name='deDenseLayerOutput2', act=tf.tanh)

	deDenseLayerOutput3 = createDenseLayer(input_tensor=deDenseLayerOutput2, input_dims=32, output_dims=32, layer_name='deDenseLayerOutput3', act=tf.tanh)

	deDenseLayerOutput4 = createDenseLayer(input_tensor=enDenseLayerOutput3, input_dims=32, output_dims=16, layer_name='enDenseLayerOutput4', act=tf.tanh)

	decryptOutput = createDenseLayer(input_tensor=enDenseLayerOutput4, input_dims=keyLength+wordLength, output_dims=8, layer_name='decryptOutput', act=tf.tanh)
	#Listener Stop

	#Hacker Start
	hackerDenseLayerOutput1 = createDenseLayer(input_tensor=speaker_output_var, input_dims=wordLength, output_dims=16, layer_name='hackerDenseLayerOutput1', act=tf.tanh)

	hackerDenseLayerOutput2 = createDenseLayer(input_tensor=hackerDenseLayerOutput1, input_dims=wordLength+keyLength, output_dims=32, layer_name='hackerDenseLayerOutput2', act=tf.tanh)

	hackerDenseLayerOutput3 = createDenseLayer(input_tensor=hackerDenseLayerOutput2, input_dims=32, output_dims=32, layer_name='hackerDenseLayerOutput3', act=tf.tanh)

	hackerDenseLayerOutput4 = createDenseLayer(input_tensor=hackerDenseLayerOutput3, input_dims=32, output_dims=16, layer_name='hackerDenseLayerOutput4', act=tf.tanh)

	hackerOutput = createDenseLayer(input_tensor=hackerDenseLayerOutput4, input_dims=wordLength+keyLength, output_dims=8, layer_name='hackerOutput', act=tf.tanh)
	#Hacker Stop

	return encryptOutput, decryptOutput, hackerOutput

def vec2bin(vec):
	str=''
	for i in range(0,8):
		if(vec[i]>=0):
			str+='1'
		else:
			str+='0'
	return str

def main():

	encryptOutput,decryptOutput, hackerOutput = createNN()

	saver = tf.train.Saver()
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	speaker_loss=np.float32([])
	listener_loss=np.float32([])
	hacker_loss=np.float32([])
	
	if(len(sys.argv)>=2):		
		print('Number of Epochs: '+str(int(sys.argv[1])))
		num_epochs=int(sys.argv[1])
		num_epochs_hacker=4
	if(len(sys.argv)>=3):
		if(sys.argv[2]!='0'):
			start=int(sys.argv[2])
			saver = tf.train.Saver()
			saver.restore(sess, '/media/arjun/Arjun\'s Drive/Data/Workspace/Python/Tensor Flow/Adversarial Cryptography/Networks/AdversarialCryptography'+sys.argv[3]+'.ckpt')
		else:
			start=0
			print('Start: '+sys.argv[2])
	learningRateSpeaker=0.005
	learningRateListener=0.005
	learningRateHacker=0.0005
	if(len(sys.argv)>=4):
		learningRateSpeaker=np.float32(sys.argv[3])
	if(len(sys.argv)>=5):
		learningRateListener=np.float32(sys.argv[4])
	if(len(sys.argv)>=6):
		learningRateHacker=np.float32(sys.argv[5])
	
	print('Learning Rates: '+str(learningRateSpeaker)+','+str(learningRateListener)+','+str(learningRateHacker))

	listener_loss = tf.reduce_sum(tf.square(decryptOutput - speaker_input_var))	
	hacker_loss = tf.reduce_sum(tf.square(hackerOutput - speaker_input_var))
	speaker_loss = listener_loss + ((200/(hacker_loss)) - (hacker_loss))

	
	#listener_key_loss = T.sum(T.sum(T.sqr(T.grad(listener_loss,key_input_var)), axis=1))
	#speaker_hacker_loss = T.sum(T.sum(T.sqr(((8/2)-(hacker_prediction-speaker_input_var))/(8/2)), axis=1))
	
	temp = set(tf.global_variables())
	train_step_listener = tf.train.AdamOptimizer(learning_rate=learningRateListener, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(listener_loss)
	train_step_hacker = tf.train.AdamOptimizer(learning_rate=learningRateHacker, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(hacker_loss)
	train_step_speaker = tf.train.AdamOptimizer(learning_rate=learningRateSpeaker, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(speaker_loss)
	sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))

	epoch = 0
	
	X=np.float32(np.zeros((2**8,8)))
	get_bin = lambda x, n: format(x, 'b').zfill(n)

	for i in range(0,2**8):
		for j in range(0,8):
			if(get_bin(i,8)[j]=='1'):
				X[i,j]= 1
			if(get_bin(i,8)[j]=='0'):
				X[i,j]= -1

	keys=np.float32(np.zeros((2**8,numBatches,8)))
	for i in range(0,2**8):
		keys[i]=np.tile(np.float32(X[i,0:]),(numBatches,1))
	
	speaker_err = 0
	speaker_batches = 0

	listener_err = 0
	listener_batches = 0
	
	hacker_err = 0
	hacker_batches = 0

	while (epoch<num_epochs):

		print("Training model...")

		start_time = time.time()
		# In each database, we do a full pass over the training data:
		for i in range(0,2**8):
			key=keys[i]
		#	print(key)
			for batch in iterate_minibatches(X, numBatches, shuffle=True):

				sess.run(train_step_speaker, feed_dict={key_input_var: key, speaker_input_var: batch})
				sess.run(train_step_listener, feed_dict={key_input_var: key, speaker_input_var: batch})
				if (epoch%num_epochs_hacker==0):
					sess.run(train_step_hacker, feed_dict={key_input_var: key, speaker_input_var: batch})
					hacker_err+=hacker_loss.eval(feed_dict={key_input_var: key, speaker_input_var: batch})/num_epochs_hacker
				
				speaker_batches+=1
				listener_batches+=1
				hacker_batches+=1
				
				speaker_err+=speaker_loss.eval(feed_dict={key_input_var: key, speaker_input_var: batch})
				listener_err+=listener_loss.eval(feed_dict={key_input_var: key, speaker_input_var: batch})
		
				speaker_batches+=1
				listener_batches+=1
				hacker_batches+=1
	
		print("Epoch {} took {:.3f}s".format(start+epoch+1, time.time() - start_time))
		print("  speaking loss:\t\t{:.12f}".format(speaker_err/speaker_batches))
		print("  listening loss:\t\t{:.12f}".format(listener_err/listener_batches))
		print("  hacking loss:\t\t{:.12f}".format(hacker_err/hacker_batches))
			
		epoch=epoch+1
        saver.save(sess, "/media/arjun/Arjun\'s Drive/Data/Workspace/Python/Tensor Flow/Adversarial Cryptography/Networks/AdversarialCryptography"+str(start+epoch)+".ckpt")

if __name__ == '__main__':
	main()
