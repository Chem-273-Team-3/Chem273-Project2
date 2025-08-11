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

def randSteps(stepSize_mu: float = 0.03,

              stepSize_std: float = 0.04,

              N_steps: int = 4):

    

    rStepSize = np.random.normal(stepSize_mu,stepSize_std,N_steps)

    rStepAngle = np.random.uniform(0,2*np.pi,N_steps)

   

    x = np.zeros((N_steps+1,1))

    y = np.zeros((N_steps+1,1))

    x[1:,0] = np.cumsum(rStepSize*np.cos(rStepAngle))

    y[1:,0] = np.cumsum(rStepSize*np.sin(rStepAngle))

   

    return x,y

 

 

 

 

 

class Ecoli_Walk():

   

    def __init__(self,

                 A: float = -3/1000,

                 a: float = 0,

                 B: float = -1/1000,

                 b: float = 0,

                 C: float = 3,

                 iterationCount: int = 30,

                 X_init: float = -8,

                 Y_init: float = -8,

                 tumbles: int = 4,

                 stepSize_mu: float = 0.5,

                 stepSize_std: float = 0.5,

                 grad_Est: bool = True,

                 plotLevels: int = 20):

       

        self.A = A

        self.a = a

        self.B = B

        self.b = b

        self.C = C

        self.iterationCount = iterationCount

        self.X_init = X_init

        self.Y_init = Y_init

        self.tumbles = tumbles

        self.stepSize_mu = stepSize_mu

        self.stepSize_std = stepSize_std

        self.grad_Est = grad_Est

        self.plotLevels = plotLevels

 

  

    #__________________________________________________________________________#

 

    #Gradient Funcion:

 

    def FoodLoc_F1(self, x: np.array = np.zeros((1,1)),

                 y: np.array = np.zeros((1,1))):

       

        A = self.A

        a = self.a 

        B = self.B 

        b = self.b 

        C = self.C

        

        

        G = A*(x-a)**2 + B*(y-b)**2 + C

        return G

   

    #__________________________________________________________________________#

 

    def gradA_XY_estimation(self,X,Y):

       

        FoodLoc_F1 = self.FoodLoc_F1

       

        dA_dX = FoodLoc_F1(x=X[-1],y=Y[0]) - FoodLoc_F1(x=X[0],y=Y[0])

        dA_dY = FoodLoc_F1(x=X[0],y=Y[-1]) - FoodLoc_F1(x=X[0],y=Y[0])

        dX = X[-1] - X[0]

        dY = Y[-1] - Y[0]

       

        

        gradA = np.array([[dA_dX/dX],[dA_dY/dY]])

        normFactor = np.sqrt(np.sum(gradA**2)) #Euclidian Norm

       

        gradA /= normFactor

        return gradA

 

    #__________________________________________________________________________#

 

    def gradA_XY(self, X,Y):

        A = self.A

        a = self.a 

        B = self.B 

        b = self.b 

        C = self.C

       

        dA_dX = 2*A*X[-1]

        dA_dY = 2*B*Y[-1]

 

        gradA = np.array([[dA_dX],[dA_dY]])

        normFactor = np.sqrt(np.sum(gradA**2)) #Euclidian Norm

       

        gradA /= normFactor

        return gradA

 

 

   

    #__________________________________________________________________________#

   

    def runEcoliWalk(self):

        iterationCount = self.iterationCount

        tumblesPlusRun = self.tumbles + 1

       

        gradA_XY_estimation = self.gradA_XY_estimation

        gradA_XY = self.gradA_XY

       

        

        #Set the Ecoli location

        X_init = self.X_init

        Y_init = self.Y_init

        Ecoli_x = X_init*np.ones((tumblesPlusRun , iterationCount))

        Ecoli_y = Y_init*np.ones((tumblesPlusRun , iterationCount))

   

    

        #Random steps (tumble)

        steps_x, steps_y = randSteps(stepSize_mu = self.stepSize_mu,

                                     stepSize_std = self.stepSize_std,

                                     N_steps = self.tumbles)

   

        Ecoli_x[:,0] += steps_x[:,0]

   

        Ecoli_y[:,0] += steps_y[:,0]

   

        #__________________________________________________________________________#

   

        #Initial plot:

   

        #plt.contourf(X,Y,Z, levels = 12) #Gradient

        #plt.plot(Ecoli_x[:,0], Ecoli_y[:,0], color = 'r',marker = 'o') #Ecoli

        #plt.show()

 

 

        #__________________________________________________________________________#

        #Loop:

   

        for runIter in range(1,iterationCount):

            runStep_Magnitude = .8

            test_X = Ecoli_x[:,runIter-1]

            test_Y = Ecoli_y[:,runIter-1]

           

            if self.grad_Est == True:

                runXY = runStep_Magnitude * gradA_XY_estimation( X = test_X , Y = test_Y )

            else:

                runXY = runStep_Magnitude * gradA_XY( X = test_X , Y = test_Y )

           

            #Run Step Gradient Descent

            Ecoli_x[:,runIter] = Ecoli_x[0,runIter-1] + runXY[0]

           

            Ecoli_y[:,runIter] = Ecoli_y[0,runIter-1] + runXY[1]

           

            #______________________________________________________________________#

           

            #Random steps (tumble)

            steps_x, steps_y = randSteps(stepSize_mu=.6)

           

            Ecoli_x[:,runIter] += steps_x[:,0]

           

            Ecoli_y[:,runIter] += steps_y[:,0]

           

        #__________________________________________________________________________#

           

    

    

    

        #Reshape Ecoli arrays for plotting:

   

        plotX = np.reshape(Ecoli_x.transpose(),(iterationCount*tumblesPlusRun,))

        plotY = np.reshape(Ecoli_y.transpose(),(iterationCount*tumblesPlusRun,))

   

        plot_Ecoli = self.plot_Ecoli

   

        plot_Ecoli(plotX, plotY)


    

    #__________________________________________________________________________#

    def plot_Ecoli(self, plotX, plotY):

       

        FoodLoc_F1 = self.FoodLoc_F1

       

        

        X_min = np.min(plotX)

        Y_min = np.min(plotY)

        X_max = np.max(plotX)

        Y_max = np.max(plotY)

       

        X_spacer = (X_max-X_min)/20

        Y_spacer = (Y_max-Y_min)/20

       

      

        

        x = np.linspace(X_min - X_spacer, X_max + X_spacer, 1000)

        y = np.linspace(Y_min - Y_spacer, Y_max + Y_spacer, 1000)

       

        X, Y = np.meshgrid(x,y)

        

        Z = FoodLoc_F1(x = X, y = Y)

       

        

        

        #Plot Result:   

        plt.contourf(X,Y,Z, levels = self.plotLevels) #Gradient

        plt.plot(plotX, plotY, color = 'r',marker = 'o', markersize = 3) #Ecoli

        plt.show()

 

    #__________________________________________________________________________#
