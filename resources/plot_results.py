import matplotlib.pyplot as plt
import numpy as np

def plot_loss_epoch(workDir,model='unet',ShowPlot=False):
	filename='log_'+model
	file = open(workDir+filename+'.csv','r')

	file_lines = file.readlines()

	epoch_arr = np.array([])
	train_loss_arr =np.array([])
	test_loss_arr =np.array([])

	for i,line in enumerate(file_lines):
		if i==0:
			epoch_index = line.split(',').index('epoch')
			train_loss_index = line.split(',').index('loss')
			test_loss_index = len(line.split(','))-1
			continue

		epoch = np.int32(line.split(',')[epoch_index]) + 1
		train_loss = np.float32(line.split(',')[train_loss_index])
		test_loss = np.float32(line.split(',')[test_loss_index].replace('\r\n',''))

		epoch_arr = np.append(epoch_arr,[epoch])
		train_loss_arr = np.append(train_loss_arr,train_loss)
		test_loss_arr = np.append(test_loss_arr,test_loss)

	plt.plot(epoch_arr,train_loss_arr)
	plt.plot(epoch_arr,test_loss_arr)
	plt.title(filename.replace('log_','')+' - (Dice) loss vs Epoch')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig(workDir+filename+'.pdf')
	plt.savefig(workDir+filename+'.png')
	if ShowPlot==True: plt.show()

	file.close()

# What to do when this file is run:
if __name__ == '__main__':
    # workDir = '/media/data/180505_v1/internal/checkpoints/'
    # workDir = '/media/data/July20-2018/'
    workDir = '/Users/rizki/Documents/Projects/ShelterSegmentation_take2/shelterdata/'
    plot_loss_epoch(workDir,'unet')
