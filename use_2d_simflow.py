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
nu = .5
mu = 1
g = 0
dt = .1

#Initial and boundary conditions
flow = np.zeros([100,100,2])
flow[:,:,1]=.2
boundary_conditions_values = np.zeros(flow.shape)
boundary_conditions_values[:,0,1]=1
boundary_conditions_mask = boundary_conditions_values != 0
cs=np.array([[21,45],[60,70],[80,25], [70,25]])
cs=[[50,50],[50,20]]
xs = np.arange(1,100)
ys = np.arange(1,100)
for c in cs:
    m=5
    for x in xs:
        for y in ys:
            v = np.array([x,y])
            if np.linalg.norm(v - c)<m:
                boundary_conditions_values[x,y,:]=0
                boundary_conditions_mask[x,y,:]=True
                
#Evolve flow
def main():
    t0 = time.time()
    for i in range(100000):
        flow, P = evolve_flow_2d(boundary_conditions_mask, boundary_conditions_values, flow, nu, mu, g, dt)
        if i % 10000 == 0:
            print(i)
            print(f'{time.time() - t0:.2f} seconds per 10,000 frames')
            t0 = time.time()

if __name__ == '__main__':
    main()