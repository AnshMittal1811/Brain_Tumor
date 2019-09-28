
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot
import math
import os 
os.environ['NUMBAPRO_CUDALIB']='Anaconda3/envs/cudaenv/Library/bin'
from numba import vectorize, cuda
from pyculib import rand as curand


@vectorize(["float64(float64, float64, float64, float64, float64)"], target = "cuda")
def step(price, dt, c0, c1, noise):
    return price * math.exp(c0 * dt + c1 * noise)




def montecarlo(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility **2
    c1 = volatility * np.sqrt(dt)
    
    prng = curand.PRNG(rndtype = curand.PRNG.XORWOW)
    d_noises = cuda.device_array(paths.shape[0])
    
    d_current = cuda.to_device(paths[:,0])
    d_next = cuda.device_array(paths.shape[0])
    
    for j in range(1, paths.shape[1]):      #for each time step
        #prices = paths[:, j - 1]            #last prices # Now no slicing needed
        #gaussian noise for simulation
        #noises = np.random.normal(0., 1., prices.size)
        prng.normal(d_noises, 0., 1.)
        #simulate
        d_next = step(d_current, dt, c0, c1, d_noises)
        
        
        d_next.copy_to_host(paths[:,j])
            
        d_next, d_current = d_current, d_next
        
        
#Stock Information Parameters
StockPrice = 20.83
StrikePrice = 21.50
Volatility = 0.021
InterestRate = 0.20
Maturity = 5. / 12.


#monte carlo simulation parameters
NumPath = 3000000
NumStep = 100


#plotting 
MAX_PATH_IN_PLOT = 50


def driver(pricer, do_plot=False):
    paths = np. zeros((NumPath, NumStep + 1), order ='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep
    
    ts = timer()
    pricer(paths, DT, InterestRate, Volatility)
    te = timer()
    elapsed = te - ts
    
    ST = paths[:, -1]
    PaidOff = np. maximum(paths[:, -1] - StrikePrice, 0)
    print('Result')
    fmt = '%20s: %s'
    print(fmt % ('stock price', np.mean(ST)))
    print(fmt % ('standard error', np.std(ST) / np.sqrt(NumPath)))
    print(fmt % ('paid off', np.mean(PaidOff)))
    optionprice = np.mean(PaidOff) * np.exp(-InterestRate * Maturity)
    print(fmt % ('option price', optionprice))
    
    print('Performance')
    NumCompute = NumPath * NumStep
    print(fmt % ('Mstep/second', '%.2f' % (NumCompute / elapsed / 1e6)))
    print(fmt % ('time elapsed', '%.3fs' % (te - ts)))
    
    if do_plot:
        pathct = min(NumPath, MAX_PATH_IN_PLOT)
        for i in range(pathct):
            pyplot.plot(paths[i])
        print('Plotting %d/%d paths' % (pathct, NumPath))
        pyplot.show()

        
driver(montecarlo, do_plot = False)