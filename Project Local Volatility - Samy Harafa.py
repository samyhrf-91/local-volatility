# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import norm
from scipy import optimize
 
#%%
plt.close('all')

prices1 = pd.read_csv(r'C:\Users\samyh\OneDrive\Documents\Financial derivatives\Project 1\New_option_prices_1.csv', header=None)
prices2 = pd.read_csv(r'C:\Users\samyh\OneDrive\Documents\Financial derivatives\Project 1\New_option_prices_2.csv', header=None)

prices1.columns = ['T','K','c']
prices2.columns = ['T','K','c']

print(prices1.head())

#plot1 = pr1.iloc[:,0:2]
#plot2 = pr2.iloc[:,0:2]

p1 = prices1.to_numpy()
p2 = prices2.to_numpy()

#%% Line plots
# =============================================================================
# fig = plt.figure()
#  
# ax = plt.axes(projection ='3d')
#  
# ax.plot3D(p1[:,0], p1[:,1], p1[:,2])
# ax.set_title('Call prices 1')
# ax.set_xlabel('Time to maturity')
# ax.set_ylabel('Strike price')
# ax.set_zlabel('Call option price')
# 
# fig2 = plt.figure()
#  
# ax2 = plt.axes(projection ='3d')
#  
# ax2.plot3D(p2[:,0], p2[:,1], p2[:,2])
# ax2.set_title('Call prices 2')
# ax2.set_xlabel('Time to maturity')
# ax2.set_ylabel('Strike price')
# ax2.set_zlabel('Call option price')
# plt.show()
# =============================================================================

#%%
 
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
ax.plot_trisurf(p1[:,0], p1[:,1], p1[:,2])
ax.set_xlabel("Maturity time")
ax.set_ylabel("Strike price")
ax.set_zlabel('Call option price')
ax.set_title("Call option prices 1")


plt.show()

fig2 = plt.figure(figsize =(14, 9))
ax2 = plt.axes(projection ='3d')
ax2.set_xlabel("Maturity time")
ax2.set_ylabel("Strike price")
ax2.set_zlabel('Call option price')
ax2.set_title("Call option prices 2")
 
ax2.plot_trisurf(p2[:,0], p2[:,1], p2[:,2])
 
plt.show()

#%%
prices = p1

def volatility(prices):
    
    # Retrieving the different option maturity times 
    time_indices = [0,21,42,63,84]
    timesteps = [prices[i+21,0]-prices[i,0] for i in time_indices[:4]]
    
    # Constant step in K for the partial derivative
    dK2 = (prices[1,1]-prices[0,1])**2 # dk = 10 here
    
    time_drv = np.zeros(105)
    K_drv = np.zeros(105)
    
    # Forward time differences
    for i in time_indices[0:4]:
        time_drv[i:i+21] = (prices[i+21:i+42,2] - prices[i:i+21,2]) / (timesteps[i//21])
    
    # Backward differences for last time point
    time_drv[84:105] = (prices[84:105,2] - prices[84-21:105-21,2]) / (timesteps[3])
    
    # Central strike price differences
    for i in time_indices:
        K_drv[i+1:i+20] = (prices[i+2:i+21,2] + prices[i:i+19,2] - 2*prices[i+1:i+20,2]) / dK2 
      
    # Special cases with skewed second order difference
    K_drv[0:85:21] = (2*prices[0:85:21,2] - 5 * prices[1:86:21,2] + 4*prices[2:87:21,2] - prices[3:88:21,2]) / dK2 
    
    # Third order difference as the second order approximation is negative
    K_drv[20:105:21] = (35*prices[20:105:21,2]/12 - 26 * prices[19:104:21,2]/3 + 19*prices[18:103:21,2]/2 - 14*prices[17:102:21,2]/3 + 11*prices[16:101:21,2]/12) / dK2 
    
    
    volatility = np.sqrt(2/(prices[:,1])**2) * np.sqrt(time_drv/K_drv)

    return volatility  

volatility1, volatility2 = volatility(p1), volatility(p2) 
 
#%% Line plots

# =============================================================================
# fig = plt.figure()
#  
# ax = plt.axes(projection ='3d')
#  
# ax.plot3D(p1[21:105,0], p1[21:105,1], volatility1[21:105])
# ax.set_title('Volatility for prices 1')
# ax.set_xlabel('Time to maturity')
# ax.set_ylabel('Strike price')
# ax.set_zlabel('Volatility')
# 
# fig2 = plt.figure()
#  
# ax2 = plt.axes(projection ='3d')
#  
# ax2.plot3D(p2[21:105,0], p2[21:105,1], volatility2[21:105])
# ax2.set_title('Volatility for prices 2')
# ax2.set_xlabel('Time to maturity')
# ax2.set_ylabel('Strike price')
# ax2.set_zlabel('Volatility')
# plt.show()         
# =============================================================================

#%% Surface plots
 
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
ax.set_title('Volatility surface for prices 1')
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Strike price')
ax.set_zlabel('Volatility')

 
ax.plot_trisurf(p2[:,0], p2[:,1], volatility1)
 
plt.show()

fig2 = plt.figure(figsize =(14, 9))
ax2 = plt.axes(projection ='3d')
ax2.set_title('Volatility surface for prices 2')
ax2.set_xlabel('Time to maturity')
ax2.set_ylabel('Strike price')
ax2.set_zlabel('Volatility')
 
ax2.plot_trisurf(p2[:,0], p2[:,1], volatility2)
 
plt.show()

#%%
### CODE FOR VOLATILITY WITH INTEREST RATE R

def volatility_bis(prices,r):
    
    prices_r=np.exp(-r*prices[:,0])*prices[:,2]
    time_indices = [0,21,42,63,84]
    timesteps = [prices[i+21,0]-prices[i,0] for i in time_indices[:4]]
    dK2 = (prices[1,1]-prices[0,1])**2 # dk = 10 here
    
    time_drv = np.zeros(105)
    K_drv = np.zeros(105)
    K_drv_fst = np.zeros(105)
    
    for i in time_indices[0:4]:
        time_drv[i:i+21] = (prices_r[i+21:i+42] - prices_r[i:i+21]) / (timesteps[i//21])
    
    #Backward differences for last time point
    time_drv[84:105] = (prices_r[84:105] - prices_r[84-21:105-21]) / (timesteps[3])
    
    for i in time_indices:
        K_drv[i+1:i+20] = (prices_r[i+2:i+21] + prices_r[i:i+19] - 2*prices_r[i+1:i+20]) / dK2 #lacks 0,20,21,41,42,62,63,83,84,104
  
    #Special cases
    #K_drv[0:85:21] = (prices_r[0:85:21] - 2 * prices_r[1:86:21] + prices_r[2:87:21]) / dK2 #right-sided FD
    #K_drv[20:105:21] = (prices_r[20:105:21] - 2 * prices_r[19:104:21] + prices_r[18:103:21]) / dK2 #left-sided FD
    
    K_drv[0:85:21] = (2*prices_r[0:85:21] - 5 * prices_r[1:86:21] + 4*prices_r[2:87:21] - prices_r[3:88:21]) / dK2 
    K_drv[20:105:21] = (35*prices_r[20:105:21]/12 - 26 * prices_r[19:104:21]/3 + 19*prices_r[18:103:21]/2 - 14*prices_r[17:102:21]/3 + 11*prices_r[16:101:21]/12) / dK2
    
    for i in time_indices:
        K_drv_fst[i+1:i+20] = (prices_r[i+2:i+21] - prices_r[i+1:i+20]) / np.sqrt(dK2)
  
    K_drv_fst[0:85:21] = (-prices_r[0:85:21] + prices_r[1:86:21]) / np.sqrt(dK2) #right-sided FD
    K_drv_fst[20:105:21] = (prices_r[20:105:21] - prices_r[19:104:21] ) / np.sqrt(dK2) #left-sided FD
    
    
    volatility = np.sqrt(2/(prices[:,1])**2) * np.sqrt((time_drv+r*prices[:,1]*K_drv_fst)/K_drv)

    return volatility

#%% Surface plots
 
volatility_bis1, volatility_bis2 = volatility_bis(p1, 0.01), volatility_bis(p2, 0.01) 

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
ax.plot_trisurf(p2[:,0], p2[:,1], volatility_bis1)
 
plt.show()

fig2 = plt.figure(figsize =(14, 9))
ax2 = plt.axes(projection ='3d')
 
ax2.plot_trisurf(p2[:,0], p2[:,1], volatility_bis2)
 
plt.show()

#%% Meanquare error between r = 0 and r used

mse1 = np.mean((volatility_bis1[volatility_bis1 > 0] - volatility1[volatility_bis1 > 0])**2)
mse2 = np.mean((volatility_bis2[volatility_bis2 > 0] - volatility2[volatility_bis2 > 0])**2)

print("Mean square error for first prices:", mse1)
print('nan values in volatility 1:', len(volatility_bis1) - len(volatility_bis1[volatility_bis1 > 0]))

print("Mean square error for second prices", mse2)
print('nan values in volatility 2:', len(volatility_bis2) - len(volatility_bis2[volatility_bis2 > 0]))

#%% Greeks
S01 = 1593
S02 = 1624
sigma2_init = 0.1*np.ones(105)

def phi2(sigma2,K,S,T,r=0):
    return np.exp(-r*T)/ (K * np.sqrt(2*np.pi*sigma2*T)) *  \
    np.exp(-(np.log(K/S) - (r-sigma2/2)*T)**2 / (2*sigma2*T))
    
def phi(sigma2,K,S,T,r=0):
    d1 = (np.log(S/K) + (r+sigma2/2)*T) / (np.sqrt(sigma2*T))
    d2 = d1 - np.sqrt(sigma2*T)
    
    return - S * d1 * norm.pdf(d1) / (K**2 * sigma2 * T) + S * norm.pdf(d1) / (K**2 * np.sqrt(sigma2 * T)) + np.exp(-r*T) * \
        norm.pdf(d2) / (np.sqrt(sigma2 * T) * K) - d2 * np.exp(-r*T) * norm.pdf(d2) / (K * sigma2 * T)
    
def theta(sigma2,K,S,T,r=0):
    d1 = (np.log(S/K) + (r+sigma2/2)*T) / (np.sqrt(sigma2*T))
    d2 = d1 - np.sqrt(sigma2*T)
    return - S* norm.pdf(d1)*d1/(2*T) + r*K*np.exp(-r*T) * norm.pdf(d2) + K * d2 * np.exp(-r*T)* norm.pdf(d2) /(2*T)

def chi(sigma2,K,S,T,r=0):
    return 0 


def newton_function(sigma2,K,S,T,r=0):
    return (2* (theta(sigma2,K,S,T,r) + r*K*chi(sigma2,K,S,T,r)) / (K**2)*phi(sigma2,K,S,T,r)) - sigma2


Kvec = p1[:,1]
Tvec = p1[:,0]
volatility_greek = optimize.newton(newton_function, sigma2_init, args=(Kvec,S01,Tvec,0,), maxiter=200)

#%%

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
ax.plot_trisurf(p2[:,0], p2[:,1], volatility_greek)
 
plt.show()
