import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

orig_stdout = sys.stdout
f = open('data.txt', 'w')
sys.stdout = f


tot_values=350
mean=0
variance=0.5
lower_limit=0
upper_limit=10
Y=[]
X=[]
X.append(lower_limit)
for i in range(tot_values-1):
	X.append(X[-1]+float(upper_limit-lower_limit)/tot_values)
for i in X:
	Y.append(math.sin(float(i))+np.random.normal(mean,variance))
#plt.scatter(X,Y)
#plt.show()

print (tot_values)
for i in range(len(X)):
	print (X[i])
	print (Y[i])

sys.stdout = orig_stdout
f.close()
