import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data(filename, seq_length):
	data = pd.read_csv(filename, index_col = False).values
	np.random.shuffle(data)
	data = data.reshape(-1, 509, 3)

	x_train, y_train = [], []
	for row in data[:int(0.8*len(data))]:
		for i in range(0, len(row)-seq_length-1, seq_length):
			curr = row[i:i+seq_length]
			x_train.append(curr)
			y_train.append(row[i+seq_length])

	x_test, y_test = [], []
	for row in data[int(0.8*len(data)):]:
		for i in range(0, len(row)-seq_length-1, seq_length):
			curr = row[i:i+seq_length]
			x_test.append(curr)
			y_test.append(row[i+seq_length])

	return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


filename = 'lorenz_std.csv'
batch_size = 512
seq_length = 10
x_train, y_train, x_test, y_test = load_data(filename, seq_length)
print(len(x_train), "training sequences")
print(len(x_test), "test sequences")
print(x_train.shape)
print(batch_size, "batch size")

#BuildGraph
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, None, 3])
y = tf.placeholder(tf.float32, [None, 3])

rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(16)])
outputs, states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = x, dtype = tf.float32)

D1 = tf.layers.dense(outputs[:,-1], 3)

#Loss
L = tf.reduce_mean(tf.square(D1 - y))
trainer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(L)

print(x_train.shape, y_train.shape)

init = tf.global_variables_initializer()
iterations = len(x_train)//batch_size
epochs = 3000
reset = False
saver = tf.train.Saver()
model_path = "./models/lorenz_std.ckpt"

with tf.Session() as sess:
	sess.run(init)

	if "models" in os.listdir() and not reset:
		saver.restore(sess, model_path)

	start = time.time()
	print("Begun")
	for i in range(1,epochs+1):
		for j in range(iterations):
			next_batch = np.random.randint(0, len(x_train), size = batch_size)
			batch_x = x_train[next_batch]
			batch_y = y_train[next_batch]

			_, loss = sess.run([trainer, L], feed_dict = { x : batch_x, y : batch_y })

		print("Epoch", i, "L :", loss)
		if (i-1)%10 == 0:
			saver.save(sess, model_path)

	print("Done :", timedelta(seconds = time.time() - start))


	#Testerinos########################################
	TrainMSE = 0
	stepsTr = int(len(x_train)*0.01)

	for i in range(stepsTr):
		if i%10 == 0:
			print("Training points", i, "/", stepsTr)
		starter = np.array(x_train[i])							#Shape is 10, 3
		predicted = np.array(x_train[i])

		for j in range(seq_length*100-10):
			pred = sess.run(D1, feed_dict = {x : [starter]})[0]		#Shape is 3

			if len(starter) >= seq_length - 1: 
				starter = np.vstack((starter[1:], pred))
			else:
				starter = np.vstack((starter, pred))

			predicted = np.vstack((predicted, pred))

		actuals = x_train[i:i+100].reshape(seq_length*100, 3)
		TrainMSE += np.square(actuals-predicted).mean()

	TestMSE = 0  
	stepsTe = int(len(x_test)*0.01)

	for i in range(stepsTe):
		if i%10 == 0:
			print("Testing points", i, "/", stepsTe)
		starter = np.array(x_test[i])
		predicted = np.array(x_test[i])

		for j in range(seq_length*100-10):
			pred = sess.run(D1, feed_dict = {x : [starter]})[0]

			if len(starter) >= seq_length - 1: 
				starter = np.vstack((starter[1:], pred))
			else:
				starter = np.vstack((starter, pred))

			predicted = np.vstack((predicted, pred))

		actuals = x_test[i:i+100].reshape(seq_length*100, 3)
		TestMSE += np.square(actuals-predicted).mean()

	print("Train MSE Loss :", round(TrainMSE/stepsTr, 4))  		
	print("Test MSE Loss :", round(TestMSE/stepsTe, 4))

	plt.rcParams["figure.figsize"] = (8,5)


	fig = plt.figure()
	ax = plt.axes(projection = '3d')
	ax.plot([predicted[0, 0]], [predicted[0, 1]], [predicted[0, 2]], color = 'C9', markersize = 105)
	ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], alpha = 0.8, color = 'C1', label = "Predicted")
	ax.plot(actuals[:, 0], actuals[:, 1], actuals[:, 2], alpha = 0.7, color = 'C0', label = 'Actual')
	plt.title("Initial point " + str(predicted[0]))
	plt.legend()

	plt.show()