# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 23:46:33 2022

@author: Rafael Rivero de Nicolás
"""


import numpy as np
from scipy.optimize import fsolve

# %% Functions Definitions


def Euler(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * Function(U_(n),t).

    '''
    
    U_n1 = X + dt*Function(X,t)
    
    return U_n1


def Inverse__Euler(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: X = U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * Function(U_(n+1),t+dt).

    '''
    
    def Inverse__Euler__Operator(U_n1):
        
        return U_n1 - X - dt * Function(U_n1, t)
    
    
    U_n1 = fsolve(Inverse__Euler__Operator, X)
    
    return U_n1


def RK4(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6.

    '''
   
    k1 = Function( X, t )
    
    k2 = Function( X + dt * k1/2, t + dt/2 )
    
    k3 = Function( X + dt * k2/2, t + dt/2 )
    
    k4 = Function( X + dt *k3,    t + dt   )
    
    U_n1 = X + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6
    
    return U_n1


def Crank__Nicolson(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt. 
        It is also the vector that satisfies: U_(n+1) = X + dt/2 * ( Function(X,t) + Function(U_(n+1), t+dt)).

    '''
    
    def  Crank_Nicolson_Operator(U_n1):
        return  U_n1 - X - dt/2 * ( Function(X,t) + Function(U_n1,t+dt) )
    
    U_n1 = fsolve(Crank_Nicolson_Operator, X)
    
    return U_n1


def Cauchy_Problem(F, U_0, time_domain, Temporal_scheme='RK4'):
    
    print( 'Temporal Scheme used:: ' + Temporal_scheme )
    
    t = 0.; U_n1 = np.zeros(len(U_0))
    
    U = np.zeros([len(U_0), len(time_domain)])
    
    U[:,0] = U_0
    
    for i in range(0, len(time_domain)-1):
        
        dt = round(time_domain[i+1] - time_domain[i], 5)
        
        t = round(t + dt, 5)
        
        X = U[:,i]
        
        if Temporal_scheme == 'RK4':
            
            U_n1 = RK4(X, t, dt, F)
            
        elif Temporal_scheme == 'Euler':
            
            U_n1 = Euler(X, t, dt, F)
            
        elif Temporal_scheme == 'Crank-Nicolson':
            
            U_n1 = Crank__Nicolson(X, t, dt, F)
            
        elif Temporal_scheme == 'Inverse Euler':
            
            U_n1 = Inverse__Euler(X, t, dt, F)
            
        else:
            
            print('Introduce a valid Temporal scheme::\n\tEuler\n\tRK4\n\tCrank-Nicolson\n\tInverse Euler ')
            break
        
        U[:,i+1] = U_n1
            
    return U


