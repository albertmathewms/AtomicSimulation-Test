import numpy as np

n = 100 #Number of balls

rc = 3 #Cut-off radius

l = (n/0.1)**(1/3) #Cube length

k = 5.0 #spring constant

r0 = 2 #closest distance of approach

m = np.ones((n,1)) #Mass of the balls

dt = 0.01 #time increment

t = 500 #total running time

p = np.random.uniform(0, l, size=(n, 3))

v = np.random.uniform(-50, 50, size=(n, 3))