import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def lorenzAtt(x, y, z, sigma, rho, beta):
	xN = x + sigma*(y-x)/100				#Propotional to intensity of convection motion???
	yN = y + (x*(rho-z)-y)/100				#Proportional to temperature diff bw ascending descending currents???
	zN = z + (x*y - beta*z)/100				#Propotional to difference of vertical temperature profile from linearity????
	return xN, yN, zN

sigma = 10.0
rho = 28.9
beta = 8/3
points_to_be_saved = 500
initital_points = np.random.randn(500, 3)
length = len(initital_points)
data = []

for i in range(100):
	x, y, z = np.array(initital_points[i])
	iterations = np.array([initital_points[i]])

	for j in range(points_to_be_saved-1):
		x, y, z = lorenzAtt(x, y, z, sigma, rho, beta)
		iterations = np.vstack((iterations, [x, y, z]))
		
	data.append(iterations)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot(iterations[:, 0], iterations[:, 1], iterations[:, 2], alpha = 0.8)
ax.title.set_text(str(initital_points[i]))
plt.show()

data = np.array(data)
data = data.reshape(-1, points_to_be_saved*3)
print(data.shape)
df = pd.DataFrame(data)
df.to_csv('lorenz_std.csv', header = False, index = False)