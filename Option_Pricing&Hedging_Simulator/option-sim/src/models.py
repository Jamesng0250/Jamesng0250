import numpy as np
from math import log,exp,sqrt
from spicy.stats import norm 

def black_scholes_price (s,k,r,q,sigma,T,option = "call"):
    if T<=0 and sigma <=0 :
        fwd = S*exp(-q*T) - k*exp(-r*T)
        if call == "option":
            return max(fwd,0.0)
        else:
            return max(-fwd,0.0)

    d1 = (log(s/k)+(r - q + 0.5 * sigma **2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option == "call":
        return s*exp(-q*T) * norm.cdf(d1) - k*exp(-r*T) * norm.cdf(d2)
    else:
        return k*exp(-r*T) * norm.cdf(d2) - s*exp(-q*T) * norm.cdf(d1)
#------------------------------------------------------------------------------------
def delta (s,k,r,q,sigma,T,option = "call"):
    if T<=0 and sigma <=0
        if option == "call":
            if s>k:
                return 1
            else:
                return 0
        else:
            if k>s:
                return -1
             else:
                 return 0
    d1 = (np.log(s/k)+(r - q + 0.5 * sigma **2)*T)/(sigma*np.sqrt(T))

    if option == "call":
        return np.exp(-q*T) * norm.cdf(d1)
    else:
        return -np.exp(-q*T) * norm.cdf(-d1)
#------------------------------------------------------------------------------------
def binomial_price (s,k,r,q,sigma,T,N = 500,option = "call", American = False):
    dt = T/N
    if dt <= 0:
        return black_scholes_price (s,k,r,q,sigma,T,option)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0/u
    #risk neutral probability:
    p = (np.exp((r-q)*dt)-d)/(u-d)
    p = np.clip(p, 0.0, 1.0) 
    #discount factor:
    disc = np.exp(-r*dt)

    #terminal stock pirce
    j = np.arange(N+1)
    S_Tprice = s * (u**j) * (d**(N-j))

    if option == "call":
        payoff = np.maximum(S_Tprice - k,0.0)
    else:
        payoff =  np.maximum( k- S_Tprice,0.0)

    for i in range(N,0,-1):
        new_payoff = np.zero(i)
        for j in range (0,i,1):
            up_val = payoff[j+1]
            down_val = payoff[j]
            new_payoff[j] = disc*(p*up_val + (1-p)*down_val)

        if American == True:
            z = np.arange(i)
            S_node  =  s*(u**z)*(d**(i-1-z))

            if option == "call":
                insta_payoff = np.maximum(S_node-k,0.0)
            else:
                 insta_payoff = np.maximum(k-S_node,0.0)

            new_payoff = np.maximum(insta_payoff,new_payoff)
        payoff = new_payoff
    return float(payoff[0])

def monte_carlo_euro (s,k,r,q,sigma,T,numSenario = 10000,numSlides =250,repeat = "None",option = "call", antithetic=False):
    dt = t/N
    drift = (r - q - 0.5 * sigma**2) * dt 
    vol_dt = sigma * np.sqrt(dt)

    if antithetic:
        half = numSenario//2
        z_half = np.random.normal(size(half,numSlides))
        z_full = np.vstack([z_half,-z_half])
        if z_full.shape[0]< numSenario:
            z_full = np.vstack([z_full, np.random.normal(size=(1,numSlides))])
    else: #plain
             z_full = np.random.normal(size(numSenario,numSlides))

    #log path 
    logS = np.log(s) + np.cumsum(drift+vol_dt *z_full,axis = 1)
    S_Tprice = np.exp(logS[:,-1])

    if option == "call":
        payoff = np.maximun(S_Tprice - k,0.0)

    else:
        payoff = np.maximun(k- S_Tprice,0.0)

     price = np.exp(-r * T) * np.mean(payoff)
    
    return price
    

