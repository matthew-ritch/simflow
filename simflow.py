#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:26:43 2022
@author: matthew-ritch

we model fluid flow using constant density incompressible navier-stokes (convective)
u_t + (u ⋅ ∇)u - (mu/rho_0) * (∇^2)u = -∇(P/rho_0) + g

evolve_flow_2d does this for 2D flow fields
evolve_flow_3d does this for 3D flow fields
"""

import numpy as np
from scipy import ndimage as nd

filt1 = [-1, 0, 1]
filt2 = [1, -2, 1]

def evolve_flow_2d(boundary_conditions_mask, boundary_conditions_values, 
                   flow, mu, rho_0, g, dt):
    '''
    Evolves a 2d flow field to its next state, given boundary conditions, fluid parameters, gravity, and a time step size.
            Parameters:
                    boundary_conditions_mask (boolean numpy array) [M x N x 2]: this mask should be True where you want to maintain boundary conditions in the flow state
                    boundary_conditions_values (float numpy array) [M x N x 2]: matrix of boundary condition flow state 
                    flow (float numpy array) [M x N x 2]: preceding state of the flow field. flow[i,j,:2] are the two components of the flow field vector at location (i,j) at a single time step.
                    mu (float scalar): dynamic viscosity
                    rho_0 (float scalar): density of the fluid
                    g (float numpy vector) [2]: 2d force of gravity applied to all fluid
                    dt (float scalar): time step duration to evolve the flow
            Returns:
                    flow (float numpy array) [M x N x 2]: resulting state of the flow field. flow[i,j,:2] are the two components of the flow field vector at location (i,j) at a single time step.
                    P (float numpy array) [M x N]: resulting pressure field. flow[i,j] is the pressure at location (i,j) at the resulting time step.
    '''
    nu = mu / rho_0
    flow[boundary_conditions_mask] = boundary_conditions_values[boundary_conditions_mask]
    ###
    u_xx = nd.convolve1d(flow, filt2, axis=0, mode='nearest')
    u_yy = nd.convolve1d(flow, filt2, axis=1, mode='nearest')
    lap = u_xx + u_yy
    ###
    u_x = nd.convolve1d(flow, filt1, axis=0, mode='nearest')
    u_y = nd.convolve1d(flow, filt1, axis=1, mode='nearest')
    grad0 = np.concatenate([u_x[:,:,0,np.newaxis], u_y[:,:,0,np.newaxis] ], axis=2)
    grad1 = np.concatenate([u_x[:,:,1,np.newaxis], u_y[:,:,1,np.newaxis] ], axis=2)
    udg0 = np.sum(flow * grad0, axis=2)
    udg1 = np.sum(flow * grad1, axis=2)
    u_dot_grad = np.concatenate([udg0[:,:,np.newaxis], udg1[:,:,np.newaxis]], axis=2)
    ###
    P = -u_x[:,:,0]-u_y[:,:,1]
    P_x = nd.convolve1d(P/rho_0, filt1, axis=0, mode='nearest')
    P_y =nd.convolve1d(P/rho_0, filt1, axis=1, mode='nearest')
    gradP = np.concatenate([P_x[:,:,np.newaxis], P_y[:,:,np.newaxis]], axis=2)
    ###
    u_t = nu * lap - u_dot_grad + g - gradP
    flow += dt * u_t
    flow[boundary_conditions_mask] = boundary_conditions_values[boundary_conditions_mask]
    ###
    return flow, P

def evolve_flow_3d(boundary_conditions_mask, boundary_conditions_values, 
                   flow, mu, rho_0, g, dt):
    '''
    Evolves a 2d flow field to its next state, given boundary conditions, fluid parameters, gravity, and a time step size.
            Parameters:
                    boundary_conditions_mask (boolean numpy array) [M x N x P x 3]: this mask should be True where you want to maintain boundary conditions in the flow state
                    boundary_conditions_values (float numpy array) [M x N x P x 3]: matrix of boundary condition flow state 
                    flow (float numpy array) [M x N x P x 3]: preceding state of the flow field. flow[i,j,k,:3] are the three components of the flow field vector at location (i,j,k) at the preceding time step.
                    mu (float scalar): dynamic viscosity
                    rho_0 (float scalar): density of the fluid
                    g (float numpy vector) [3]: 3d force of gravity applied to all fluid
                    dt (float scalar): time step duration to evolve the flow
            Returns:
                flow (float numpy array) [M x N x P x 3]: resulting state of the flow field. flow[i,j,k,:3] are the three components of the flow field vector at location (i,j,k) at the resulting time step.
                P (float numpy array) [M x N x P]: resulting pressure field. flow[i,j,k] is the pressure at location (i,j,k) at the resulting time step.
    '''
    nu = mu / rho_0
    flow[boundary_conditions_mask] = boundary_conditions_values[boundary_conditions_mask]
    ###
    u_xx = nd.convolve1d(flow, filt2, axis=0, mode='nearest')
    u_yy = nd.convolve1d(flow, filt2, axis=1, mode='nearest')
    u_zz = nd.convolve1d(flow, filt2, axis=2, mode='nearest')
    lap = u_xx + u_yy + u_zz
    ###
    u_x = nd.convolve1d(flow, filt1, axis=0, mode='nearest')
    u_y = nd.convolve1d(flow, filt1, axis=1, mode='nearest')
    u_z = nd.convolve1d(flow, filt1, axis=2, mode='nearest')
    grad0 = np.concatenate([u_x[:,:,:,0,np.newaxis], u_y[:,:,:,0,np.newaxis], u_z[:,:,:,0,np.newaxis] ], axis=3)
    grad1 = np.concatenate([u_x[:,:,:,1,np.newaxis], u_y[:,:,:,1,np.newaxis], u_z[:,:,:,1,np.newaxis] ], axis=3)
    grad2 = np.concatenate([u_x[:,:,:,2,np.newaxis], u_y[:,:,:,2,np.newaxis], u_z[:,:,:,2,np.newaxis] ], axis=3)
    udg0 = np.sum(flow * grad0, axis=3)
    udg1 = np.sum(flow * grad1, axis=3)
    udg2 = np.sum(flow * grad2, axis=3)
    u_dot_grad = np.concatenate([udg0[:,:,:,np.newaxis], udg1[:,:,:,np.newaxis], udg2[:,:,:,np.newaxis]], axis=3)
    ###
    P = -u_x[:,:,:,0]-u_y[:,:,:,1] - u_z[:,:,:,2]
    P_x = nd.convolve1d(P/rho_0, filt1, axis=0, mode='nearest')
    P_y =nd.convolve1d(P/rho_0, filt1, axis=1, mode='nearest')
    P_z=nd.convolve1d(P/rho_0, filt1, axis=2, mode='nearest')
    gradP = np.concatenate([P_x[:,:,:,np.newaxis], P_y[:,:,:,np.newaxis], P_z[:,:,:,np.newaxis]], axis=3)
    ###
    u_t = nu * lap - u_dot_grad + g - gradP
    flow += dt * u_t
    flow[boundary_conditions_mask] = boundary_conditions_values[boundary_conditions_mask]
    ###
    return flow, P