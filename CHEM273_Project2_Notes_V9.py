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
                 #Function Params:
                 foodFunctionType: str = 'Polynom_2',

                 #Gradient Params:
                 gradEst_MinDelta: float = 1e-4,
                 analytical_Grad: bool = False,


                 #Simulation Params
                 iterationCount: int = 30,
                 X_init: float = 8,
                 Y_init: float = -8,
                 tumbles: int = 4,
                 run2tumble_lengthRatio = 10,
                 avg2std_tumbleRatio: float = 100,

                 #Optimizer Params
                 optimizeType: int = 0,

                 #PLotting Params
                 plotLevels: int = 20,
                 **kwargs
                 ):




        #Simulation Params
        self.iterationCount = iterationCount
        self.X_init = X_init
        self.Y_init = Y_init
        self.tumbles = tumbles
        self.run2tumble_lengthRatio  = run2tumble_lengthRatio
        self.avg2std_tumbleRatio = avg2std_tumbleRatio

        #Gradient Param:
        self.gradEst_MinDelta = gradEst_MinDelta

        #Set Gradient Function as estimation; will update if anayltical is requested
        self.Grad_func = self.Grad_estimation



        #__________________________________________________________________________#
        ####Food Function Params:
        self.foodFunctionType = foodFunctionType

        foodFuncVars = {'Polynom_2__A': -3e-8,
                        'Polynom_2__a': 1,
                        'Polynom_2__B': -1e-8,
                        'Polynom_2__b': 2,
                        'Polynom_2__C': 3,

                        'Polynom_4__a1': -1/2,
                        'Polynom_4__b1': -1/4,
                        'Polynom_4__c1': 3/2,
                        'Polynom_4__d1': 0,
                        'Polynom_4__e1': 5,
                        'Polynom_4__a2': -1/2,
                        'Polynom_4__b2': -1/4,
                        'Polynom_4__c2': 3/2,
                        'Polynom_4__d2': 0,
                        'Polynom_4__e2': 5,


            }




        for key in kwargs:
            #Polynom_2 vars
            if key == 'Polynom_2__A':
                foodFuncVars['Polynom_2__A'] = kwargs[key]
            elif key == 'Polynom_2__a':
                foodFuncVars['Polynom_2__a'] = kwargs[key]
            elif key == 'Polynom_2__B':
                foodFuncVars['Polynom_2__B'] = kwargs[key]
            elif key == 'Polynom_2__b':
                foodFuncVars['Polynom_2__b'] = kwargs[key]
            elif key == 'Polynom_2__C':
                foodFuncVars['Polynom_2__C'] = kwargs[key]

            #Polynom 4 vars:
            elif key == 'Polynom_4__a1':
                foodFuncVars['Polynom_4__a1'] = kwargs[key]
            elif key == 'Polynom_4__b1':
                foodFuncVars['Polynom_4__b1'] = kwargs[key]
            elif key == 'Polynom_4__c1':
                foodFuncVars['Polynom_4__c1'] = kwargs[key]
            elif key == 'Polynom_4__d1':
                foodFuncVars['Polynom_4__d1'] = kwargs[key]
            elif key == 'Polynom_4__e1':
                foodFuncVars['Polynom_4__e1'] = kwargs[key]
            elif key == 'Polynom_4__a2':
                foodFuncVars['Polynom_4__a2'] = kwargs[key]
            elif key == 'Polynom_4__b2':
                foodFuncVars['Polynom_4__b2'] = kwargs[key]
            elif key == 'Polynom_4__c2':
                foodFuncVars['Polynom_4__c2'] = kwargs[key]
            elif key == 'Polynom_4__d2':
                foodFuncVars['Polynom_4__d2'] = kwargs[key]
            elif key == 'Polynom_4__e2':
                foodFuncVars['Polynom_4__e2'] = kwargs[key]




        if foodFunctionType == 'Polynom_2':
            self.foodLocFunc = self.FoodLoc_Polynom_2

            if analytical_Grad == True:
                self.Grad_func = self.analytical_Grad_Polynom_2

            self.A = foodFuncVars['Polynom_2__A']
            self.a = foodFuncVars['Polynom_2__a']
            self.B = foodFuncVars['Polynom_2__B']
            self.b = foodFuncVars['Polynom_2__b']
            self.C = foodFuncVars['Polynom_2__C']

        #elif <ENTER FUNCTION VARS HERE>
        else:
            self.foodLocFunc = self.FoodLoc_Polynom_4

            if analytical_Grad == True:
                self.Grad_func = self.analytical_Grad_Polynom_4

            self.a1 = foodFuncVars['Polynom_4__a1']
            self.b1 = foodFuncVars['Polynom_4__b1']
            self.c1 = foodFuncVars['Polynom_4__c1']
            self.d1 = foodFuncVars['Polynom_4__d1']
            self.e1 = foodFuncVars['Polynom_4__e1']
            self.a2 = foodFuncVars['Polynom_4__a2']
            self.b2 = foodFuncVars['Polynom_4__b2']
            self.c2 = foodFuncVars['Polynom_4__c2']
            self.d2 = foodFuncVars['Polynom_4__d2']
            self.e2 = foodFuncVars['Polynom_4__e2']


        #__________________________________________________________________________#
        ####Optimizer Params:

        self.optimizeType = optimizeType

        optimizerConfig= {
            'learningRate': 0.01,
            'momentumRate': 0.9,
            'ADAM_momentumRate': 0.9,
            #'AdaGradRate': 0.9,
            'RmsPropRate': 0.9999}

        for key in kwargs:
            if key == 'learningRate':
                optimizerConfig['learningRate'] = kwargs[key]
            elif key == 'momentumRate':
                optimizerConfig['momentumRate'] = kwargs[key]
            elif key == 'ADAM_momentumRate':
                optimizerConfig['ADAM_momentumRate'] = kwargs[key]
            elif key == 'RmsPropRate':
                optimizerConfig['RmsPropRate'] = kwargs[key]

        if optimizeType == 1:
            self.learning_R = optimizerConfig['learningRate']
            self.momentum_R = optimizerConfig['momentumRate']
            self.adaptive_R = 1

        elif optimizeType == 2:
            self.learning_R = optimizerConfig['learningRate']
            self.momentum_R = 1
            self.adaptive_R = 1

        elif optimizeType == 3:
            self.learning_R = optimizerConfig['learningRate']
            self.momentum_R = optimizerConfig['momentumRate']
            self.adaptive_R = 1

        elif optimizeType == 4:
            self.learning_R = optimizerConfig['learningRate']
            self.momentum_R = optimizerConfig['momentumRate']
            self.adaptive_R = optimizerConfig['RmsPropRate']

        elif optimizeType == 5:
            self.learning_R = optimizerConfig['learningRate']
            self.momentum_R = optimizerConfig['ADAM_momentumRate']
            self.adaptive_R = optimizerConfig['RmsPropRate']
        else:
            self.learning_R = 1
            self.momentum_R = 1
            self.adaptive_R = 1

        #__________________________________________________________________________#


        self.plotLevels = plotLevels

    #__________________________________________________________________________#



    def FoodLoc_Polynom_2(self, x: np.array = np.zeros((1,1)), y: np.array = np.zeros((1,1))):

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


        Z = A*(x-a)**2  + B*(y-b)**2  + C

        return Z

    #__________________________________________________________________________#



    def FoodLoc_Polynom_4(self, x: np.array = np.zeros((1,1)), y: np.array = np.zeros((1,1))):

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



        a1 = self.a1
        b1 = self.b1
        c1 = self.c1
        d1 = self.d1
        e1 = self.e1
        a2 = self.a2
        b2 = self.b2
        c2 = self.c2
        d2 = self.d2
        e2 = self.e2


        Z = a1*x**4 + b1*x**3 + c1*x**2 + d1*x + e1  + \
            a2*y**4 + b2*y**3 + c2*y**2 + d2*x + e2

        return Z

    def analytical_Grad_Polynom_4(self,X,Y):
        a1 = self.a1
        b1 = self.b1
        c1 = self.c1
        d1 = self.d1

        a2 = self.a2
        b2 = self.b2
        c2 = self.c2
        d2 = self.d2


        x = X[-1]
        y = Y[-1]

        dZ_dX = 4*a1*x**3 + 3*b1*x**2 + 2*c1*x + d1  #Partial derivative with respect to X

        dZ_dY = 4*a2*y**3 + 3*b2*y**2 + 2*c2*y + d2 #Partial derivative with respect to Y

        return dZ_dX, dZ_dY

    def analytical_Grad_Polynom_2(self,X,Y):
        A = self.A
        a = self.a
        B = self.B
        b = self.b



        dZ_dX = 2*A*(X[-1] - a) #Partial derivative with respect to X

        dZ_dY = 2*B*(Y[-1] - b) #Partial derivative with respect to Y

        return dZ_dX, dZ_dY


    def Grad_estimation(self,X,Y):

        """
        Here we go

        """

        #Bring in function (determined in __init__)
        foodLocFunc = self.foodLocFunc


        dZ = foodLocFunc(x = X[-1], y = Y[-1]) - foodLocFunc(x = X[0], y= Y[0])

        dX = (X[-1] - X[0])
        dY = (Y[-1] - Y[0])

        #Prevent delta from becoming too small, but maintain direction:
        dZ += np.sign(dZ)*self.gradEst_MinDelta
        dX += np.sign(dX)*self.gradEst_MinDelta
        dY += np.sign(dY)*self.gradEst_MinDelta


        dZ_dX = dZ / dX

        dZ_dY = dZ / dY


        return dZ_dX, dZ_dY



    #__________________________________________________________________________#

    def update_RandomStepLength(self, X_RunLength, Y_RunLength):
        ####Initial Run and Tumble lengths
        X = np.abs(X_RunLength)
        Y = np.abs(Y_RunLength)
        runStep = np.array(([X],[Y]))

        #set the tumble step length for (x,y)
        tumbleStep_avg = runStep / self.run2tumble_lengthRatio

        #set the tumble step variance for (x,y)
        tumbleStep_std = tumbleStep_avg  /self.avg2std_tumbleRatio

        return tumbleStep_avg, tumbleStep_std


    #__________________________________________________________________________#



    def EcoliWalk_Simulation(self):

        """
        Ecoli random walk simulation:

        """

        #__________________________________________________________________________#
        ####Setting variables
        iterationCount = self.iterationCount #Number of iterations

        tumblesPlusRun = self.tumbles + 1 #Set the tumble + run step size (t + n*dt)


        Grad_func = self.Grad_func
        update_RandomStepLength = self.update_RandomStepLength


        #Set the Ecoli initial locations:
        X_init = self.X_init
        Y_init = self.Y_init

        #create column vectors to determine f(t: t+4dt) locations
        #Initially all rows of these columns = X_init or Y_init
        Ecoli_x = X_init*np.ones((tumblesPlusRun , iterationCount))

        Ecoli_y = Y_init*np.ones((tumblesPlusRun , iterationCount))


        #Optimizer Terms
        Gx=0
        Gy=0
        Gx_2 = 0
        Gy_2 = 0
        runX = 0
        runY = 0


        #__________________________________________________________________________#

        ####Initial Run and Tumble lengths
        tumbleStep_avg, tumbleStep_std = update_RandomStepLength(X_RunLength = self.learning_R,
                                                                 Y_RunLength = self.learning_R)

        #__________________________________________________________________________#
        #capture the cumsum of "N_steps" amount of random steps:

        tumbles_x, tumbles_y = randSteps(stepSize_mu = tumbleStep_avg,
                                     stepSize_std = tumbleStep_std ,
                                     N_steps = self.tumbles)

        #__________________________________________________________________________#

        ####Initial Tumble Steps

        Ecoli_x[:,0] += tumbles_x[:,0] #Add the cumsum to X_positions

        Ecoli_y[:,0] += tumbles_y[:,0] #Add the cumsum to Y_positions


        #__________________________________________________________________________#
        ####Loop run and tumble steps


        for runIter in range(1,iterationCount):


            prev_X = Ecoli_x[:,runIter-1] #save the previous run+tumble steps
            prev_Y = Ecoli_y[:,runIter-1] #save the previous run+tumble steps

            #Grad_func determined by user input:
            runX, runY = Grad_func( X = prev_X , Y = prev_Y)


            #______________________________________________________________________#

            ###Momentum optimization:

            if self.optimizeType in [1, 3, 4]: #Simple Momentum
                Momentum_decay = self.momentum_R
                Gx = Momentum_decay*Gx + runX
                Gy = Momentum_decay*Gy + runY

            elif self.optimizeType == 5: #Adam Momentum:
                Momentum_decay = self.momentum_R
                Gx = Momentum_decay*Gx + runX*(1-Momentum_decay)
                Gy = Momentum_decay*Gy + runY*(1-Momentum_decay)

            else:
                Gx = runX
                Gy = runY

            #______________________________________________________________________#
            ###Adaptive Optimization:

            if self.optimizeType in [2, 3]: #Adagrad:
                Gx_2 += runX**2
                Gy_2 += runY**2

            elif self.optimizeType == 5: # RMSProp
                Adaptive_decay = self.adaptive_R
                Gx_2 = (1-Adaptive_decay) * runX**2  +  Adaptive_decay * Gx_2
                Gy_2 = (1-Adaptive_decay) * runY**2  +  Adaptive_decay * Gy_2
            else:
                Gx_2 = 1
                Gy_2 = 1


            #______________________________________________________________________#

            #Adjust learning rates with optimizers:

            X_lr_adj = self.learning_R * Gx / (np.sqrt(Gx_2))
            Y_lr_adj = self.learning_R * Gy / (np.sqrt(Gy_2))



            #______________________________________________________________________#

            #Determine first step (similar to init_X, init_Y before loop)

            Ecoli_x[:,runIter] = (prev_X[-1] + X_lr_adj) * np.ones(np.shape(prev_X))

            Ecoli_y[:,runIter] = (prev_Y[-1] + Y_lr_adj) * np.ones(np.shape(prev_X))




            #______________________________________________________________________#

            #Determine the avg and std tumblestep length:

            tumbleStep_avg, tumbleStep_std = update_RandomStepLength(X_RunLength = X_lr_adj, \
                                                                     Y_RunLength = Y_lr_adj)

            #______________________________________________________________________#

            #Capture "N_steps" amount of random steps:


            tumbles_x, tumbles_y = randSteps(stepSize_mu = tumbleStep_avg,
                                         stepSize_std = tumbleStep_std ,
                                         N_steps = self.tumbles)

            #______________________________________________________________________#

            #Add the tumble (random step cumsum) to the Ecoli Position

            Ecoli_x[:,runIter] += tumbles_x[:,0]

            Ecoli_y[:,runIter] += tumbles_y[:,0]



        #__________________________________________________________________________#

        ####Plotting E.Coli:
        #Reshape Ecoli arrays for plotting (from 2D to 1D) by stacking column(i)
        #below column(i-1):
        plotX = np.reshape(Ecoli_x.transpose(),(iterationCount*tumblesPlusRun,))
        plotY = np.reshape(Ecoli_y.transpose(),(iterationCount*tumblesPlusRun,))


        #Use plot_Ecloi function to generate the plot
        plot_Ecoli = self.plot_Ecoli
        plot_Ecoli(plotX, plotY)
        print(Ecoli_x[:,-1])
        print(Ecoli_y[:,-1])


    #__________________________________________________________________________#

    def plot_Ecoli(self, plotX, plotY):


    #__________________________________________________________________________#
        #Bring in function (determined in __init__)
        foodLocFunc = self.foodLocFunc



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
        Z = foodLocFunc(x = X, y = Y)
    #__________________________________________________________________________#



        #Plot Result:

        plt.contourf(X,Y,Z, levels = self.plotLevels, cmap = 'gray') #Plotting for food source function in 2D:



        plt.plot(plotX, plotY, color = 'r',marker = 'o', markersize = 1) #Ecoli walk

        plt.scatter(iter_start_X,iter_start_Y, color = 'g', marker = '*', s = 50, label = 'Iteration')

        plt.scatter(plotX[0],plotY[0],color = 'k',marker = 's', s = 100, zorder=9, label = 'Start')
        plt.scatter(plotX[-1],plotY[-1],color = 'b',marker = 'X', s = 200, zorder=10, label = 'End')

        plt.legend()
        plt.show()



