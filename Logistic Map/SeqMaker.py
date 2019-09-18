import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def logistic(x, k):
	return x*(1-x)*k

k = 3.3
num_initial_points = 1000
points_to_be_looked_at = 500
initital_points = np.linspace(0, 0.99, num_initial_points)[1:]

data = []

fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')

for i in range(num_initial_points-1):
	x = initital_points[i]
	iterations = [x]

	for _ in range(points_to_be_looked_at):
		x = logistic(x, k)
		iterations.append(x)
	
	data.append(iterations)
	
	#pos = i
	#ax[pos//3][pos%3].plot(iterations)
	#ax[pos//3][pos%3].title.set_text(str(i/100))

#plt.show()
data = np.array(data)
print(data.shape)
print(data[0])
df = pd.DataFrame(data)
filename = 'logistic'+str(k)+'.csv'
df.to_csv(filename, header = False, index = False)
print("Saved to", filename)