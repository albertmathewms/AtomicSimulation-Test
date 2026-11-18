import numpy as np


def distpairs(p, rc,cell,weight):
    hno=cell@weight
    
    d = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = (np.sum((d**2), axis=2))**0.5
    neighbours = (dist<=rc) & (dist!=0)

    return(d, dist, neighbours)
    
def pair_forces_loop(p, rc, k, r0,cell,weight):
    d, dist, neighbours = distpairs(p, rc,cell,weight)
    f = np.zeros_like(p)
    magnitude = ((-k*((dist) - r0))*neighbours)/(dist + 1e-12)
    unit_vector = d/(dist[:, :, np.newaxis] + 1e-12)
    force = (magnitude[:, :, np.newaxis])*unit_vector
    f = np.sum(force, axis=1)

    # for i in range (p.shape[0]):
    #     for j in range (i+1, p.shape[0]):
    #         if neighbours[i][j]==True:
    #             magnitude = -k*(dist[i][j] - r0) 
    #             unit_vector = d[i][j]/dist[i][j]
    #             force = magnitude*unit_vector
    #             f[i] += force
    #             f[j] -= force
    return(f)

def drag_force(v, g):
    fd=np.zeros_like(v)
    fd=-g*v

    return(fd)

def walls_reflect(p, v, l):
    for dim in range(3):
        hitp = p[:, dim] < 0
        v[hitp, dim] *= -1
        p[hitp, dim] = 0.0

        hitn = p[:, dim] > l
        v[hitn, dim] *= -1
        p[hitn, dim] = l

    return(p, v)

def step_once(p, v, m, dt, l, rc, k, r0,g,cell,weight):
    f = pair_forces_loop(p, rc, k, r0,cell,weight)
    fd=drag_force(v,g)
    a = (f+fd)/m
    v += a*dt
    p += v*dt
    p, v = walls_reflect(p, v, l)

    return(p,v)

def run_simulation(p, v, m, t, dt, l, rc, k, r0,g,cell,weight):
    phistory = np.zeros((t, p.shape[0], 3))
    for step in range(t):
        p, v = step_once(p, v, m, dt, l, rc, k, r0,g,cell,weight)
        phistory[step] = p

    return(phistory)