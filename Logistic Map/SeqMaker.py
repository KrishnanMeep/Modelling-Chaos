import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def logistic(x, k):
	return x*(1-x)*k

k = 3.9
initital_points = np.linspace(0, 0.99, 1000)[1:]
length = len(initital_points)
data = []

print("Initial points : ", initital_points)

fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')

for i in range(length):
	x = initital_points[i]
	iterations = [0 for _ in range(9)] + [x]

	for _ in range(500):
		x = logistic(x, k)
		iterations.append(x)
	
	data.append(iterations)
	
	#pos = i
	#ax[pos//3][pos%3].plot(iterations)
	#ax[pos//3][pos%3].title.set_text(str(i/100))

#plt.show()
data = np.array(data)
print(data.shape)
df = pd.DataFrame(data)
filename = 'logistic'+str(k)+'.csv'
df.to_csv(filename, header = False, index = False)
print("Saved to", filename)