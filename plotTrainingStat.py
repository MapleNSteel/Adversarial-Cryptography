import numpy as np
import sys
import matplotlib.pyplot as plt

def main():

	outputDirectory=''

	if(len(sys.argv)>=2):
		print('Output Directory: '+str(sys.argv[1]))
		outputDirectory=sys.argv[1]
	
	speakerLoss=np.float32([])
	listenerLoss=np.float32([])


	speakerLoss = np.memmap(outputDirectory+'/speakerLoss.dat', dtype='float32', mode='r')
	listenerLoss = np.memmap(outputDirectory+'/listenerLoss.dat', dtype='float32', mode='r')
	hackerLoss = np.memmap(outputDirectory+'/hackerLoss.dat', dtype='float32', mode='r')

	#plot
	index=np.arange(1,np.size(listenerLoss)+1,1)
	
	plt.plot(index,speakerLoss,'r',index,listenerLoss,'b',index,hackerLoss,'g')
	plt.ylabel('speaker Loss, listener Loss, and hacker loss')
	plt.show()

	print((speakerLoss,listenerLoss,hackerLoss))
	print(np.size(listenerLoss))

if __name__ == '__main__':
	main()
