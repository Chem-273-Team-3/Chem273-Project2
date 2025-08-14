#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 19:33:57 2025

@author: joe21
"""

import numpy as np

import matplotlib.pyplot as plt



#__________________________________________________________________________#



#Set Random Step size

def randSteps(stepSize_mu = np.ones((2,1)),
              stepSize_std = np.ones((2,1)),
              N_steps: int = 4):
    """
    This function generates a random step <x,y> vector, where the user
    can tightly control the step size but has no influence on the direction

    Inputs:
        #stepSize_mu = the mean distance to travel
        #stepSize_std = the standard deviation from the mean distance to travel
        #N_steps = how many random step vectors to genrate: < x[0:N], y[0:N] >

    Outputs:
        x,y np.arrays((N_steps,1))
    """



    #Initialize x,y columns with one mroe row than steps
    x = np.zeros((N_steps+1,1))
    y = np.zeros((N_steps+1,1))

    x_tumbles = np.random.normal(stepSize_mu[0],stepSize_std[0],N_steps)
    y_tumbles = np.random.normal(stepSize_mu[1],stepSize_std[1],N_steps)

    x_tumbles *= np.random.choice([-1,1],size = np.shape(x_tumbles))
    y_tumbles *= np.random.choice([-1,1],size = np.shape(y_tumbles))


    x[1:,0] = np.cumsum(x_tumbles)
    y[1:,0] = np.cumsum(y_tumbles)



    return x,y





class Ecoli_Walk():



    def __init__(self,

                 A: float = -7/1000,
                 a: float = 1,
                 B: float = -1/1000,
                 b: float = 2,
                 C: float = 3,
                 iterationCount: int = 30,
                 X_init: float = 8,
                 Y_init: float = -8,
                 tumbles: int = 4,
                 grad_Est: bool = True,
                 plotLevels: int = 20,
                 eps_learning_rate: float = 0.8,
                 run_to_tumble_stepMu_Ratio = 5,
                 tumbleMu_to_tumbleSTD_Ratio: float = 100):



        self.A = A
        self.a = a
        self.B = B
        self.b = b
        self.C = C
        self.iterationCount = iterationCount
        self.X_init = X_init
        self.Y_init = Y_init
        self.tumbles = tumbles
        self.grad_Est = grad_Est
        self.plotLevels = plotLevels
        self.eps_learning_rate = eps_learning_rate
        self.run_to_tumble_stepMu_Ratio  = run_to_tumble_stepMu_Ratio
        self.tumbleMu_to_tumbleSTD_Ratio = tumbleMu_to_tumbleSTD_Ratio

    #__________________________________________________________________________#




    def FoodLoc_F1(self, x: np.array = np.zeros((1,1)), y: np.array = np.zeros((1,1))):

        """
        Food source profile:
            General equation for a 2D parabala

        Inputs:
            A,B = scalars for x,y directions
                Values closer to 0 make the slope more gradual
                Negative values are recquiered to find an absolute max

            a,b = scalars to manipulate the location of the abs max (if A and B < 0)

            C = scale to manipuate the height of each point in the G direction

        Ouptput:
            G: np.array((1,1))
        """



        A = self.A
        a = self.a
        B = self.B
        b = self.b
        C = self.C


        G = A*(x-a)**2 + B*(y-b)**2 + C

        return G



    #__________________________________________________________________________#



    def grad_F1_estimation(self,X,Y):

        """
        Estimate the gradient of F1

        Inputs:
            X,Y = np.array((N_steps+1 , 1))
                where:
                    first row = initial position OR position after run step
                    last row  = position after "N_steps" random steps

        Output:
            gradA = np.array((2,1))
                gradA[0] = partial derivative with respect to the x-direction (normalized distance)
                gradA[1] = partial derivative with respect to the y-direction (normalized distance)

        """

        #Bring in function
        FoodLoc_F1 = self.FoodLoc_F1

        A = self.A
        a = self.a
        B = self.B
        b = self.b



        dF1_dX = 2*A*(X[-1] - a) #Partial derivative with respect to X

        dF1_dY = 2*B*(Y[-1] - b) #Partial derivative with respect to Y


        return dF1_dX, dF1_dY



    #__________________________________________________________________________#



    def runEcoliWalk_F1(self):

        """
        Ecoli random walk simulation with food source function F1

        """


        #__________________________________________________________________________#
        ###Setting variables###
        iterationCount = self.iterationCount #Numebr of iterations

        tumblesPlusRun = self.tumbles + 1 #Set the tumble + run step size (t + n*dt)



        grad_F1_estimation = self.grad_F1_estimation


        #Set the Ecoli initial locations:
        X_init = self.X_init

        Y_init = self.Y_init

        #create column vectors to determine f(t: t+4dt) locations
        #Initially all rows of these columns = X_init or Y_init
        Ecoli_x = X_init*np.ones((tumblesPlusRun , iterationCount))

        Ecoli_y = Y_init*np.ones((tumblesPlusRun , iterationCount))

        #__________________________________________________________________________#

        ###Learning Rates###

        runStep_learningRate_eps = self.eps_learning_rate*np.ones((2,1)) #set the run step length for (x,y)

        #set the tumble step length for (x,y)
        tumbleStep_Mu = runStep_learningRate_eps/self.run_to_tumble_stepMu_Ratio

        #set the tumble step variance for (x,y)
        tumbleStep_std = tumbleStep_Mu  /self.tumbleMu_to_tumbleSTD_Ratio


        #__________________________________________________________________________#

        tumbles_x, tumbles_y = randSteps(stepSize_mu = tumbleStep_Mu,
                                     stepSize_std = tumbleStep_std ,
                                     N_steps = self.tumbles)

        #__________________________________________________________________________#

        ###Initial Steps###

        Ecoli_x[:,0] += tumbles_x[:,0] #Add the cumsum to X_init

        Ecoli_y[:,0] += tumbles_y[:,0] #Add the cumsum to Y_init

        #__________________________________________________________________________#
        #Optimizer Terms
        Gx=0
        Gy=0
        Gx_2 = 0
        Gy_2 = 0
        runX = 0
        runY = 0
        RMSP_Gx_2 = 0
        RMSP_Gy_2 = 0
        #__________________________________________________________________________#
        ###Loop through remaining steps with a similar logic to the initial step###


        for runIter in range(1,iterationCount):



            prev_X = Ecoli_x[:,runIter-1] #save the previous run+tumble steps
            prev_Y = Ecoli_y[:,runIter-1] #save the previous run+tumble steps

            runX, runY = grad_F1_estimation( X = prev_X , Y = prev_Y)

            #______________________________________________________________________#

            ###Momentum:
            #Momentum_decay = 0.9
            #Gx = Momentum_decay*Gx + runX
            #Gy = Momentum_decay*Gy + runY

            ####Replace with Adam Momentum:
            Momentum_decay = 0.9
            Gx = Momentum_decay*Gx + runX*(1-Momentum_decay)
            Gy = Momentum_decay*Gy + runY*(1-Momentum_decay)

            #______________________________________________________________________#
            ###Adagrad:
            #Gx_2 += runX**2
            #Gy_2 += runY**2

            #runStep_learningRate_eps[0] = self.eps_learning_rate/(np.sqrt(Gx_2) + 1e-8)
            #runStep_learningRate_eps[1] = self.eps_learning_rate/(np.sqrt(Gy_2) + 1e-8)

            ####Replace Adagrad with RMSProp
            RMSP_decayRate = 0.9999

            RMSP_Gx_2 = (1-RMSP_decayRate) * runX**2  +  RMSP_decayRate * RMSP_Gx_2
            RMSP_Gy_2 = (1-RMSP_decayRate) * runY**2  +  RMSP_decayRate * RMSP_Gy_2


            runStep_learningRate_eps[0] = self.eps_learning_rate/(np.sqrt(RMSP_Gx_2) + 1e-8)
            runStep_learningRate_eps[1] = self.eps_learning_rate/(np.sqrt(RMSP_Gy_2) + 1e-8)



            #______________________________________________________________________#

            #Determine first step (similar to init_X, init_Y before loop)

            ###NO L1L2:
            Ecoli_x[:,runIter] = (prev_X[-1] + runStep_learningRate_eps[0]*Gx) * np.ones(np.shape(prev_X))

            Ecoli_y[:,runIter] = (prev_Y[-1] + runStep_learningRate_eps[1]*Gy) * np.ones(np.shape(prev_X))

            ####L1L2 Test:
            #L1:
            #dX = prev_X[-1] - prev_X[0]
            #dY = prev_Y[-1] - prev_Y[0]
            #dXdx = np.sign(dX)
            #dYdy =  np.sign(dY)

            #L1x = 1e-8 * np.abs(dX)
            #L1y = 1e-8 * np.abs(dX)

            #L2:
            #dX2dx = dX**2 / dX
            #dY2dy = dY**2 / dY

            #L2x = 1e-4 * np.abs(dX)
            #L2y = 1e-4 * np.abs(dX)


            #Ecoli_x[:,runIter] = (prev_X[-1] + runStep_learningRate_eps[0]*Gx \
            #                      + runStep_learningRate_eps[0]*L1x*dXdx \
            #                          + runStep_learningRate_eps[0]*L2x*dX2dx) \
            #    * np.ones(np.shape(prev_X))

            #Ecoli_y[:,runIter] = (prev_Y[-1] + runStep_learningRate_eps[1]*Gy \
            #                      + runStep_learningRate_eps[1]*L1y*dYdy \
            #                          + runStep_learningRate_eps[1]*L2y*dY2dy) \
            #    * np.ones(np.shape(prev_X))

            #______________________________________________________________________#


            #Update Magnitude of random steps:

            tumbleStep_Mu = runStep_learningRate_eps*np.sqrt(runX**2 + runY**2)/self.run_to_tumble_stepMu_Ratio
            tumbleStep_std = tumbleStep_Mu / self.tumbleMu_to_tumbleSTD_Ratio

            tumbles_x, tumbles_y = randSteps(stepSize_mu = tumbleStep_Mu,
                                         stepSize_std = tumbleStep_std ,
                                         N_steps = self.tumbles)

            #______________________________________________________________________#

            Ecoli_x[:,runIter] += tumbles_x[:,0]

            Ecoli_y[:,runIter] += tumbles_y[:,0]



        #__________________________________________________________________________#


        #Reshape Ecoli arrays for plotting (from 2D to 1D) by stacking column(i)
        #below column(i-1):
        plotX = np.reshape(Ecoli_x.transpose(),(iterationCount*tumblesPlusRun,))
        plotY = np.reshape(Ecoli_y.transpose(),(iterationCount*tumblesPlusRun,))


        #Use plot_Ecloi function to genatre the plot
        plot_Ecoli = self.plot_Ecoli
        plot_Ecoli(plotX, plotY)
        print(Ecoli_x[:,-1])
        print(Ecoli_y[:,-1])


    #__________________________________________________________________________#

    def plot_Ecoli(self, plotX, plotY):


    #__________________________________________________________________________#
        FoodLoc_F1 = self.FoodLoc_F1



        ###Determine Axes min and max for plot:

        X_min = np.min(plotX)
        Y_min = np.min(plotY)
        X_max = np.max(plotX)
        Y_max = np.max(plotY)


        X_spacer = (X_max-X_min)/20
        Y_spacer = (Y_max-Y_min)/20
    #__________________________________________________________________________#

        iteration_size = self.tumbles +1

        iter_start_X = plotX[::iteration_size]
        iter_start_Y = plotY[::iteration_size]

    #__________________________________________________________________________#
        #Plotting for food source function in 2D:
        x = np.linspace(X_min - X_spacer, X_max + X_spacer, 1000)
        y = np.linspace(Y_min - Y_spacer, Y_max + Y_spacer, 1000)

        X, Y = np.meshgrid(x,y)
        Z = FoodLoc_F1(x = X, y = Y)
    #__________________________________________________________________________#



        #Plot Result:

        plt.contourf(X,Y,Z, levels = self.plotLevels) #Plotting for food source function in 2D:

        plt.plot(plotX, plotY, color = 'r',marker = 'o', markersize = 3) #Ecoli walk

        plt.scatter(iter_start_X,iter_start_Y, color = 'k', marker = '*', s = 100)

        plt.show()



