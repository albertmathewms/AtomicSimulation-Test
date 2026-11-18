import numpy as np

n = 10000 #Number of balls

rc = 2.5 #Cut-off radius

l = round((n*10 )**(1/3)) #Cube length

cl=rc*1.01

cno=int(l//cl)+1

k = 5.0 #spring constant

r0 = 2 #closest distance of approach

g = 0.5

m = np.ones((n,3)) #Mass of the balls

dt = 0.01 #time increment

t = 500 #total running time

p = np.random.uniform(0, l, size=(n, 3)).astype(np.float32)


cell=np.floor_divide(p, cl).astype(int)

weight=np.array([[1], [cno], [cno**2]]).astype(int)

#print(cell@weight)


neighbour=np.array([0,1,cno,(cno+1),(cno-1),(cno**2),((cno**2)+1),((cno**2)-1),((cno**2)+cno),((cno**2)-cno),((cno**2)+cno+1), ((cno**2)+cno-1),((cno**2)-cno-1), ((cno**2)-cno+1)]).astype(int)
#print(neighbour)

# Cell-linked-list neighbours: all 26 surrounding offsets plus the current cell.
neighbour=np.array([
    ix + cno*(iy+cno*iz)
    for iz in (-1,0,1)
    for iy in (-1,0,1)
    for ix in (-1,0,1)
], dtype=int)

# Flattened cell index for every ball.
hno=(cell@weight).astype(int).ravel()

# Linked-list heads and the next-particle links.
head=np.full(cno**3, -1, dtype=int)
links=np.full(n, -1, dtype=int)
for i in range(n):
    links[i]=head[hno[i]]
    head[hno[i]]=i

# Common aliases make the arrays easy to use in other simulation code.
heads=head
list=links

v = np.random.uniform(-50, 50, size=(n, 3)).astype(np.float32)
