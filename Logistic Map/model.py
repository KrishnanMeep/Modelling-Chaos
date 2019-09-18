import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os
from datetime import timedelta
import matplotlib.pyplot as plt

def load_data(filename, seq_length):
	data = pd.read_csv(filename, index_col = False)
	col_length = len(data.columns)
	data = data.values.reshape(-1, col_length, 1)
	np.random.shuffle(data)

	x_train, y_train = [], []
	for row in data[:int(0.8*len(data))]:
		for i in range(0, len(row)-seq_length-1):
			curr = row[i:i+seq_length]
			x_train.append(curr)
			y_train.append(row[i+seq_length])

	x_test, y_test = [], []
	for row in data[int(0.8*len(data)):]:
		for i in range(0, len(row)-seq_length-1):
			curr = row[i:i+seq_length]
			x_test.append(curr)
			y_test.append(row[i+seq_length])

	return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), col_length


filename = 'logistic3.9.csv'
model_path = "./models/logistic3.9.ckpt"
batch_size = 512
seq_length = 10
x_train, y_train, x_test, y_test, col_length = load_data(filename, seq_length)
print(len(x_train), "training sequences")
print(len(x_test), "test sequences")
print(batch_size, "batch size")


#BuildGraph
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, None, 1])
y = tf.placeholder(tf.float32, [None, 1])

rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(32)])
outputs, states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = x, dtype = tf.float32)

D1 = tf.layers.dense(outputs[:, -1], 1)
A1 = tf.nn.sigmoid(D1)
#Loss
L = tf.reduce_mean(tf.square(A1 - y))
trainer = tf.train.AdamOptimizer(learning_rate = 0.0003).minimize(L)

print(x_train.shape, y_train.shape)

init = tf.global_variables_initializer()
iterations = (len(x_train) + len(x_test))//batch_size
epochs = 1
reset = False
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)

	if "models" in os.listdir() and not reset:
		print("Previous loaded")
		saver.restore(sess, model_path)

	start = time.time()
	print("Begun")
	for i in range(1,epochs+1):
		for j in range(iterations):
			next_batch = np.random.randint(0, len(x_train), size = batch_size)
			batch_x = x_train[next_batch]
			batch_y = y_train[next_batch]

			_, loss = sess.run([trainer, L], feed_dict = { x : batch_x, y : batch_y})

		print("Epoch", i, "L :", loss)
		if i%10 == 0:
			saver.save(sess, model_path)

	print("Done :", timedelta(seconds = time.time() - start))

	TrainMSE = 0
	stepsTr = int(len(x_train)*0.01)
	jump = col_length - seq_length - 1 

	for i in range(0, stepsTr, jump):
		print("Training points", i//jump, "/", (stepsTr//jump))

		starter = np.array(x_train[i])

		predicted = list(starter.ravel())
		beg = starter[0]

		for j in range(seq_length-1,jump):
			pred = sess.run(A1, feed_dict = {x : [starter]})[0]
			if len(starter) == seq_length : 
				starter = np.vstack((starter[1:], pred))
			else:
				starter = np.vstack((starter, pred))
			predicted.append(pred)


		predicted = np.array(predicted).ravel()
		actuals = np.array(x_train[i])
		for j in range(seq_length, jump, seq_length):
			actuals = np.vstack((actuals, x_train[i+j]))

		TrainMSE += np.square(actuals-predicted).mean()


	plt.rcParams["figure.figsize"] = (15,8)

	plt.plot(actuals[:50], 'r', alpha = 0.8, label = 'Acutal')
	plt.plot(predicted.ravel()[:50], 'b:', alpha = 0.7, label = 'Predicted')
	plt.title("Initial point " + str(beg))
	plt.legend()

	plt.show()

	TestMSE = 0  
	stepsTe = int(len(x_test)*0.01)
	for i in range(0, stepsTe, jump):
		print("Testing points", i//jump, "/", (stepsTe//jump))

		starter = np.array(x_test[i])

		predicted = list(starter.ravel())
		beg = starter[0]

		for j in range(seq_length-1,jump):
			pred = sess.run(A1, feed_dict = {x : [starter]})[0]
			if len(starter) == seq_length - 1: 
				starter = np.vstack((starter[1:], pred))
			else:
				starter = np.vstack((starter, pred))
			predicted.append(pred)

		predicted = np.array(predicted).ravel()		

		actuals = np.array(x_test[i])
		for j in range(seq_length, jump, seq_length):
			actuals = np.vstack((actuals, x_test[i+j]))

		TestMSE += np.square(actuals-predicted).mean()

	print("Train MSE Loss :", round(TrainMSE[0]/(stepsTr/jump), 4))  		
	print("Test MSE Loss :", round(TestMSE[0]/(stepsTe/jump), 4))

	plt.rcParams["figure.figsize"] = (15,8)

	plt.plot(actuals[:50], 'r', alpha = 0.8, label = 'Acutal')
	plt.plot(predicted.ravel()[:50], 'b:', alpha = 0.7, label = 'Predicted')
	plt.title("Initial point " + str(beg))
	plt.legend()

	plt.show()