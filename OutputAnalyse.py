import matplotlib.pyplot as plt
import plotly.plotly as py
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
		if(sys.argv[1]!='0'):
			start=int(sys.argv[1])
			saver = tf.train.Saver()
			saver.restore(sess, '/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Workspace/Python/Tensor Flow/Adversarial Cryptography/Networks/AdversarialCryptography'+sys.argv[1]+'.ckpt')
		else:
			start=0
			print('Start: '+sys.argv[1])
	
	#listener_key_loss = T.sum(T.sum(T.sqr(T.grad(listener_loss,key_input_var)), axis=1))
	#speaker_hacker_loss = T.sum(T.sum(T.sqr(((8/2)-(hacker_prediction-speaker_input_var))/(8/2)), axis=1))
	
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

	inputInt=np.zeros((2**8))
	encryptedOutput1=np.zeros((2**8))
	decryptedOutput1=np.zeros((2**8))

	for i in range(0,2**8):
		key=keys[i]
	#	print(key)
		
		en=encryptOutput.eval(feed_dict={key_input_var: key, speaker_input_var: X})
		de=decryptOutput.eval(feed_dict={key_input_var: key, speaker_input_var: X})
		for j in range(0,2**8):
			inputInt[j]=int(vec2bin(X[j]),2)
			encryptedOutput1[j]=int(vec2bin(en[j]),2)
			decryptedOutput1[j]=int(vec2bin(de[j]),2)

		print(encryptedOutput1)
		print(decryptedOutput1)
		
		
		plt.title("Decrypt Transfer Function and Encrypt Transfer Function:")
		plt.plot(inputInt)
		plt.plot(decryptedOutput1)
		plt.plot(encryptedOutput1)
		plt.show()

if __name__ == '__main__':
	main()
