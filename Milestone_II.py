# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:22:48 2022

@author: Rafael Rivero de Nicolás
"""

import numpy as np
# import LB_Temporal_Schemes as ts # Users module
import LB_Math_Functions as mth # Users module

import matplotlib.pyplot as plt



from matplotlib import rc # LaTeX tipography
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text', usetex=True); plt.rc('font', family='serif')

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

# %% Functions 

def Kepler_Orbits_2N(X, t):
    '''
    This function only depends on the physics of the problem, it musts be an input argument
    
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Float
        Time instant in which F is being evaluated.

    Returns
    -------
    Array
        First derivate of the tate vector. dU/dt = F(U,t).

    '''
    F1 = X[2]
    
    F2 = X[3]
    
    F3 = -X[0]/(X[0]**2 + X[1]**2)**(3/2)
    
    F4 = -X[1]/(X[0]**2 + X[1]**2)**(3/2)

    return np.array([F1, F2, F3, F4])

# %% Initialitation

Temoral_schemes_available = {0:"Euler",
                             1:"Inverse Euler",
                             2:"RK4",
                             3:"Crank-Nicolson"}

scheme = Temoral_schemes_available[1]

r_0 = np.array([1, 0]) # Initial position   np.array([1.9, 0])

v_0 = np.array([0, 1]) # Initial velocity

U_0 = np.hstack((r_0,v_0)) # U_0 = np.array([r_0[0], r_0[1], v_0[0], v_0[1]])

print('Initial State Vector: U_0 = ', U_0, '\n\n\n')   

tf = 20 # 500

Delta_t = [0.2, 0.1, 0.01, 0.001]   # Δt for different simulations





# %%

U = {}

for dt in Delta_t:
    
    N = int( tf/dt )
    
    time_domain = np.linspace( 0, tf, N+1 )
    
    print('Temporal partition used Δt = ', str(dt))
    
    U[scheme+'__dt=' + str(dt)] = mth.Cauchy_Problem( Kepler_Orbits_2N, U_0, time_domain, Temporal_scheme = scheme )
    
    print('\n\n\n')


# %% Plotting

colours = ['blue', 'red', 'magenta', 'black', 'grey', 'cyan', 'yellow']


 # %% Schemes 

i = 0

fig, ax = plt.subplots(1,1, figsize=(8,8), constrained_layout='true')

if scheme == 'Inverse Euler' and tf==500:
    ax.set_xlim(-50,2)
    ax.set_ylim(-20,20)
elif scheme == 'Inverse Euler':
        ax.set_xlim(-1.25,1.25)
        ax.set_ylim(-1.25,1.25)


ax.set_title('Numeric Scheme: '+scheme, fontsize=20)
ax.grid()
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$y$',fontsize=20)

for key in U:
    
    x = U[key][0,:]
    y = U[key][1,:]
    
    ax.plot( x, y, c=colours[i], label=r'$\Delta t$ = '+str(Delta_t[i]))
    
    i = i+1

ax.legend(loc=0, fancybox=False, edgecolor="black", ncol = 1, fontsize=16); i = 0
fig.savefig('HITO_2'+scheme+'_tf=_'+str(tf)+'.pdf', transparent = True, bbox_inches="tight")
plt.show()