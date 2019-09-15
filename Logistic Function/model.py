import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os
from datetime import timedelta
import matplotlib.pyplot as plt

def load_data(filename, seq_length):
	data = pd.read_csv('logistic3.9.csv', index_col = False).values
	print(data.shape)

	np.random.shuffle(data)

	x_train, y_train, c_train = [], [], []
	for row in data[:int(0.8*len(data))]:
		beg = row[0]
		for i in range(0, len(row)-seq_length-1, seq_length):
			curr = row[i:i+seq_length].reshape(seq_length, 1)
			c_train.append([beg])
			x_train.append(curr)
			y_train.append([row[i+seq_length]])

	x_test, y_test, c_test = [], [], []
	for row in data[int(0.8*len(data)):]:
		beg = row[0]
		for i in range(0, len(row)-seq_length-1, seq_length):
			curr = row[i:i+seq_length].reshape(seq_length, 1)
			c_test.append([beg])
			x_test.append(curr)
			y_test.append([row[i+seq_length]])

	return np.array(x_train), np.array(y_train), np.array(c_train), np.array(x_test), np.array(y_test), np.array(c_test)


filename = 'logistic3.9.csv'
batch_size = 128
seq_length = 1
x_train, y_train, c_train, x_test, y_test, c_test = load_data(filename, seq_length)
print(len(x_train), "training sequences")
print(len(x_test), "test sequences")
print(batch_size, "batch size")


#BuildGraph
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, None, 1])
y = tf.placeholder(tf.float32, [None, 1])
c = tf.placeholder(tf.float32, [None, 1])

rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(16), tf.nn.rnn_cell.BasicLSTMCell(32), 
										tf.nn.rnn_cell.BasicLSTMCell(64)])
outputs, states = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = x, dtype = tf.float32)

Cat1 = tf.concat([outputs[:, -1], c], axis = 1)
D1 = tf.layers.dense(Cat1, 1)
A1 = tf.nn.sigmoid(D1)

#Loss
L = tf.reduce_mean(tf.square(A1 - y))
trainer = tf.train.AdamOptimizer(learning_rate = 0.0003).minimize(L)

print(x_train.shape, y_train.shape)

init = tf.global_variables_initializer()
iterations = (len(x_train) + len(x_test))//batch_size
epochs = 0
reset = False
saver = tf.train.Saver()
model_path = "./models/logistic3.9.ckpt"

with tf.Session() as sess:
	sess.run(init)

	if "models" in os.listdir() and not reset:
		saver.restore(sess, model_path)

	start = time.time()
	print("Begun")
	for i in range(epochs):
		for j in range(iterations):
			next_batch = np.random.randint(0, len(x_train), size = batch_size)
			batch_x = x_train[next_batch]
			batch_y = y_train[next_batch]
			batch_c = c_train[next_batch]

			_, loss = sess.run([trainer, L], feed_dict = { x : batch_x, y : batch_y, c : batch_c })

		print("Epoch", i, "L :", loss)
		if i%10 == 0:
			saver.save(sess, model_path)

	print("Done :", timedelta(seconds = time.time() - start))

	TrainMSE = 0

	steps = len(x_train)//500-30

	for i in range(steps):
		print("Training points", i, "/", steps)

		starter = np.array([x_train[i, 0]])

		predicted = list(starter.ravel())
		beg = starter

		for _ in range(seq_length*30-1):
			pred = sess.run(A1, feed_dict = {x : [starter], c : beg})[0]
			starter = np.vstack((starter[1:], pred))
			predicted.append(pred[0])


		predicted = np.array(predicted).ravel()
		actuals = x_train[i:i+30].ravel()

		TrainMSE += np.square(actuals-predicted).mean()
		
	print("Train MSE Loss :", round(TrainMSE/steps, 4))  

	TestMSE = 0  
	steps = len(x_test)//500-30

	for i in range(steps):
		print("Testing points", i, "/", steps)

		starter = np.array([x_test[i, 0]])

		predicted = list(starter.ravel())
		beg = starter

		for _ in range(seq_length*30-1):
			pred = sess.run(A1, feed_dict = {x : [starter], c : beg})[0]
			starter = np.vstack((starter[1:], pred))
			predicted.append(pred[0])


		predicted = np.array(predicted).ravel()
		actuals = x_test[i:i+30].ravel()

		TestMSE += np.square(actuals-predicted).mean()
		
	print("Test MSE Loss :", round(TestMSE/steps, 4))

	plt.rcParams["figure.figsize"] = (25,10)


	plt.plot(seq_length, actuals[seq_length], 'gx', alpha = 0.75)
	plt.plot(actuals, 'r', alpha = 0.8, label = 'Acutal')
	plt.plot(predicted.ravel(), 'b:', alpha = 0.7, label = 'Predicted')
	plt.title("Initial point " + str(beg))
	plt.legend()

	plt.show()