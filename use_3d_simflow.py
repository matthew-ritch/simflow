#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:39:02 2022
@author: matthew-ritch
"""
import numpy as np
from simflow import *
import time

#PARAMETERS
nu = 1.8
mu = 1
g = 0
dt = .05

#Initial and boundary conditions
s = 150
flow = np.zeros([s,s,s,3])
flow[:,:,:,2]=2*(np.random.rand(s,s,s)-.8)
boundary_conditions_values = np.zeros(flow.shape)
boundary_conditions_mask = boundary_conditions_values != 0
#Boundary conditions continued: model a hollow cylinder shape inducing flow along its axis
xc=s//2; yc = s//2
r0 = 6
r1 = 8
h = s//4
offset = (s-h)//2
for x in range(s):
    for y in range(s):
        if (np.linalg.norm([x-xc,y-yc])>=r0) & (np.linalg.norm([x-xc,y-yc])<=r1):
            boundary_conditions_mask[x,y,offset:offset+h] = True
            boundary_conditions_values[x,y,offset:offset+h,:] = 0
        if (np.linalg.norm([x-xc,y-yc])<r0):
            boundary_conditions_mask[x,y,offset+1:offset+h-1,:] = True
            boundary_conditions_values[x,y,offset+1:offset+h-1,2] = 3
            v=np.array([0,0])
            if x!=xc: v[0] = -1/(x-xc)
            if y!=yc: v[1] =  1/(y-yc)
            v[0] = y-yc
            v[1] = -(x-xc)
            if any(v>0): v = 3 * v /np.linalg.norm(v)
            boundary_conditions_values[x,y,offset+1:offset+h-1,:2] = v
            flow[x,y,:,2] = 3
        if (np.linalg.norm([x-xc,y-yc])>r1):
            flow[x,y,:,2] = -1
            
#Evolve flow
def main():
    t0 = time.time()
    # evolve
    for i in range(20000):
        flow, P = evolve_flow_3d(boundary_conditions_mask, boundary_conditions_values, flow, nu, mu, g, dt)
        if i % 10 == 0:
            print(f'{(time.time() - t0)/10} seconds per step.')
            t0 = time.time()

if __name__ == '__main__':
    main()