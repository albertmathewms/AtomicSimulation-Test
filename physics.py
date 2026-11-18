import numpy as np
from variables import *
from numba import njit

@njit(cache=True)
def distpairs(p, rc):
    n = p.shape[0]
    d = np.zeros((n, n, 3))
    dist = np.zeros((n, n))
    neighbours = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            dz = p[i, 2] - p[j, 2]
            d[i, j, 0] = dx
            d[i, j, 1] = dy
            d[i, j, 2] = dz
            r = (dx * dx + dy * dy + dz * dz) ** 0.5
            dist[i, j] = r
            if 0.0 < r <= rc:
                neighbours[i, j] = True

    return(d, dist, neighbours)
    
@njit(cache=True)
def pair_forces_loop(p, rc, k, r0):
    n = p.shape[0]
    f = np.zeros_like(p)
    rc_sq = rc * rc
    for i in range(n):
        px = p[i, 0]
        py = p[i, 1]
        pz = p[i, 2]
        for j in range(i + 1, n):
            dx = px - p[j, 0]
            dy = py - p[j, 1]
            dz = pz - p[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            if 0.0 < dist_sq <= rc_sq:
                dist = np.sqrt(dist_sq)
                factor = -k * (dist - r0) / dist
                fx = factor * dx
                fy = factor * dy
                fz = factor * dz
                f[i, 0] += fx
                f[i, 1] += fy
                f[i, 2] += fz
                f[j, 0] -= fx
                f[j, 1] -= fy
                f[j, 2] -= fz
    return(f)

@njit(cache=True)
def walls_reflect(p, v, l):
    n = p.shape[0]
    for i in range(n):
        for dim in range(3):
            if p[i, dim] < 0.0:
                v[i, dim] = -v[i, dim]
                p[i, dim] = 0.0
            elif p[i, dim] > l:
                v[i, dim] = -v[i, dim]
                p[i, dim] = l

    return(p, v)

@njit(cache=True)
def step_once(p, v, m, dt, l, rc, k, r0):
    f = pair_forces_loop(p, rc, k, r0)
    n = p.shape[0]
    for i in range(n):
        inv_m = 1.0 / m[i, 0]
        for dim in range(3):
            v[i, dim] += f[i, dim] * inv_m * dt
            p[i, dim] += v[i, dim] * dt
    p, v = walls_reflect(p, v, l)

    return(p, v)

@njit(cache=True)
def run_simulation(p, v, m, t, dt, l, rc, k, r0):
    phistory = np.zeros((t, p.shape[0], 3))
    for step in range(t):
        p, v = step_once(p, v, m, dt, l, rc, k, r0)
        phistory[step] = p

    return(phistory)