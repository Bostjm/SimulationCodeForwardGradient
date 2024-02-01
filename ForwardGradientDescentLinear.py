#For calculations
import numpy as np

#For plotting
import matplotlib.pyplot as plt 
from matplotlib.ticker import ScalarFormatter
####################
#REMARKS AT THE START
####################
#Python version 3.10.4
#Numpy version 1.22.3
#Matplotlib version 3.5.2
####################
#Data Generation
####################

def LinearData(Sigma,n,d,ChildSeeds,Theta): #Data generated according to equation (3.1)
    Mean = np.zeros(d)
    RNG1 = np.random.default_rng(seed=ChildSeeds[1])
    X = RNG1.multivariate_normal(mean=Mean,cov=Sigma, size=n)
    RNG2 = np.random.default_rng(seed=ChildSeeds[2])
    Epsilon = RNG2.standard_normal(size=n)
    Y =  np.zeros(n)
    for i in range(0,n):
        Y[i]= np.inner(X[i,:],Theta)+Epsilon[i]
    return X,Y

####################
#Update Methods
####################

def ForwardGradient(X,Y,ChildSeeds,d,n,a,runs): #Simple implementation using that we know the expression for the gradient in the linear model
    
    RNGInt = np.random.default_rng(seed=ChildSeeds[4])
    ThetaEstimate = np.zeros([n,d,runs])
    InitialValue = RNGInt.standard_normal(size=d)
    for i in range (0,runs):
        ThetaEstimate[0,:,i] = InitialValue
        RNG = np.random.default_rng(seed=ChildSeeds[5+i])
        for k in range (1,n):
            alpha = a/(k+a*(d+2)*(d+2))
            Xi = RNG.standard_normal(size=d)
            ThetaEstimate[k,:,i]=ThetaEstimate[k-1,:,i]-alpha*np.inner( -(Y[k]-np.inner(X[k,:],ThetaEstimate[k-1,:,i]))*X[k,:] , Xi)*Xi
    return ThetaEstimate

def GradientDescent(X,Y,ChildSeeds,d,n,a): #Stochastic gradient descent using a single point for each update
    
    RNGInt = np.random.default_rng(seed=ChildSeeds[4]) #Same seed, so same initial value as the forward gradient runs
    ThetaEstimate = np.zeros([n,d])
    ThetaEstimate[0,:]= RNGInt.standard_normal(size=d)
    for k in range (1,n):
        alpha = a/(k+a*(d+2)*(d+2)) #Same as the forward gradient learning rate.
        ThetaEstimate[k,:]=ThetaEstimate[k-1,:]+alpha*(Y[k]-np.inner(X[k,:],ThetaEstimate[k-1,:]))*X[k,:]
    return ThetaEstimate

####################
#Plots
####################

def MSEPlotsLog(MSE,MSESGD,n,d,runs): #Plot the functions with their MSE on a log-10 scale
    plt.rcParams.update({'font.size': 16})
    Xplot = np.linspace(0.0,n, n)
    
    YBound = np.zeros(n)
    YBoundOptimal = np.zeros(n)
    YBound[0] = d*d*np.log(d)
    YBoundOptimal[0] = d
    for i in range (1,n):
        YBound[i] =d*d*np.log(d)/Xplot[i]
        YBoundOptimal[i] = d/Xplot[i]
    
    figureMSE, axMSE = plt.subplots()
    axMSE.set_xlabel('Iterations')
    axMSE.set_ylabel('MSE log 10 scale')  

    axMSE.plot(Xplot,MSESGD, color="red", linestyle='solid', linewidth= 2.5)
    axMSE.plot(Xplot,MSE,color="blue" ,linestyle='solid', linewidth= 1) 
    axMSE.plot(Xplot,YBound, color="black", linestyle='dashed', linewidth= 2.5)
    axMSE.plot(Xplot,YBoundOptimal, color="black", linestyle='dashed', linewidth= 2.5)  
    
    axMSE.set_yscale('log')
    axMSE.set_xlim([0.0,n])
    axMSE.get_xaxis().set_major_formatter(ScalarFormatter(useMathText= True))
    figureMSE.savefig('MSELogarithmicForwardGradientSGDn{}d{}.pdf'.format(n,d), bbox_inches='tight', dpi=300)


def MSEPlotsLogLog(MSE,MSESGD,n,d,runs): #Plot the functions with their MSE on a log-10 scale and the iterations on a log 10 scale
    plt.rcParams.update({'font.size': 16})
    Xplot = np.linspace(0.0,n, n)
    
    YBound = np.zeros(n)
    YBoundOptimal = np.zeros(n)
    YBound[0] = d*d*np.log(d)
    YBoundOptimal[0] = d
    for i in range (1,n):
        YBound[i] = d*d*np.log(d)/Xplot[i]
        YBoundOptimal[i] = d/Xplot[i]
    
    figureMSE, axMSE = plt.subplots()

    axMSE.set_xlabel('Iterations log 10 scale')
    axMSE.set_ylabel('MSE log 10 scale')  
    axMSE.plot(Xplot,MSESGD, color="red", linestyle='solid', linewidth= 2.5)
    axMSE.plot(Xplot,MSE,color="blue" ,linestyle='solid', linewidth= 1)
    axMSE.plot(Xplot,YBound, color="black", linestyle='dashed', linewidth= 2.5)
    axMSE.plot(Xplot,YBoundOptimal, color="black", linestyle='dashed', linewidth= 2.5)

    axMSE.set_xlim([1.0,n])
    axMSE.set_xscale('log')
    axMSE.set_yscale('log')
    figureMSE.savefig('MSELogLogForwardGradientSGDn{}d{}.pdf'.format(n,d), bbox_inches='tight', dpi=300)

####################
#Main Program
####################
def Main():
    #Number of runs of forward gradient descent
    runs = 10
    #Init random seeds
    Seed = 10
    SequenceSeeder = np.random.SeedSequence(Seed)
    ChildSeeds = SequenceSeeder.spawn(5+runs)

    #Init constants
    n = 1000000
    d = 100
    a = np.log(d)
    Sigma = np.identity(d)
    RNG = np.random.default_rng(seed=ChildSeeds[0])
    Theta = RNG.standard_normal(size=d)

    #Generate the data sample
    X,Y = LinearData(Sigma,n,d,ChildSeeds,Theta)

    #Forward gradient estimate
    ThetaEstimate = ForwardGradient(X,Y,ChildSeeds,d,n,a,runs)

    #Gradient descent estimate
    ThetaEstimateSGD = GradientDescent(X,Y,ChildSeeds,d,n,a)

    #Calculate the MSE
    MSE = np.zeros([n,runs])
    MSESGD = np.zeros(n)
    for k in range (0,n):
        MSESGD[k]= (0.5)*np.inner(Theta-ThetaEstimateSGD[k,:],Theta-ThetaEstimateSGD[k,:])
    for i in range (0,runs):
        for k in range (0,n):
            MSE[k,i]= (0.5)*np.inner(Theta-ThetaEstimate[k,:,i],Theta-ThetaEstimate[k,:,i])

    #Plot the MSE
    MSEPlotsLog(MSE,MSESGD,n,d,runs)
    MSEPlotsLogLog(MSE,MSESGD,n,d,runs)

####################
#RUN THE PROGRAM
####################
Main()