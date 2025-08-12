#team member names
#Natalia Rivera, Joseph Gill, Andy Ho, Nathalie Murphy, Sibhi Sakthivel, Yashesha Kothari

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def randSteps(stepSize_mu: float = 0.4,
              stepSize_std: float = 0.04,
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



  #Randomly pick the distance to step "N_steps" times from a normal distribution:
  rStepSize = np.random.normal(stepSize_mu,stepSize_std,N_steps)

  #Randomly pick an angle in radians (0 to 2pi) from a uniform distrubution:
  rStepAngle = np.random.uniform(0,2*np.pi,N_steps)


  #Initialize x,y columns with one mroe row than steps
  x = np.zeros((N_steps+1,1))
  y = np.zeros((N_steps+1,1))

  #Convert length,angle cooridinates to x,y cooridinates:
  #Add all previous steps the current step to show walking affect:
  #First row remains 0
  x[1:,0] = np.cumsum(rStepSize*np.cos(rStepAngle))
  y[1:,0] = np.cumsum(rStepSize*np.sin(rStepAngle))


  return x,y




#define E.coli class
class Ecoli_Walk_Simulation():

    # Each cycle:
    #1) Tumble: 4 small unbiased steps --> monte carlo
    #2) Sense: compare C(t) vs C(t-4Δt)
    #3) Run: move along recent heading if improved, else opposite --> gradient step
  def __init__(self,

              A: float = -3/1000,
              a: float = 1,
              B: float = -1/1000,
              b: float = 2,
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
    self.Ecoli_x = None
    self.Ecoli_y = None


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


    #Take partial derivative in x and y directions
    #In question 1:
    Q1 = 1
    if Q1 == 1:
        #Evaluate the difference in Z at time step t+4dt vs t
        dA_dX = FoodLoc_F1(x=X[-1],y=Y[-1]) - FoodLoc_F1(x=X[0],y=Y[0])
        dA_dY = FoodLoc_F1(x=X[-1],y=Y[-1]) - FoodLoc_F1(x=X[0],y=Y[0])
    else:
        dA_dX = FoodLoc_F1(x=X[-1],y=Y[0]) - FoodLoc_F1(x=X[0],y=Y[0])
        dA_dY = FoodLoc_F1(x=X[0],y=Y[-1]) - FoodLoc_F1(x=X[0],y=Y[0])


    #In question 2:
    Q2 = 1
    if Q2 == 1:
        dX = X[-1] - X[0] #Take the difference of X before and after random steps
        dY = Y[-1] - Y[0] #Take the difference of Y before and after random steps
    else:
        dX = 2*(X[-1] - X[0]) #Take the 2 times the difference of X before and after random steps
        dY = 2*(Y[-1] - Y[0]) #Take the 2 times the difference of Y before and after random steps


    #Convert partial derivatives to a np.array ((2,1))
    gradA = np.array([[dA_dX/dX],[dA_dY/dY]])

    #Determine the Euclidian Norm of gradA for to control scaling
    normFactor = np.sqrt(np.sum(gradA**2))

    #Scale gradA
    gradA /= normFactor

    return gradA

  def grad_F1(self, X, Y):
    """
    Knowing F1, take the actual partial derivative evaluated after "N_steps"
    random steps with respect to both the X and Y directions

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

    A = self.A
    a = self.A
    B = self.B
    b = self.B

    dF1_dX = 2*A*(X[-1] - a) #Partial derivative with respect to X

    dF1_dY = 2*B*(Y[-1] - b) #Partial derivative with respect to Y


    #Convert partial derivatives to a np.array ((2,1)):
    gradF1= np.array([[dF1_dX],[dF1_dY]])
    normFactor = np.sqrt(np.sum(gradF1**2)) #Euclidian Norm

    #Scale gradF1
    gradF1 /= normFactor

    return gradF1


  def runEcoliWalk_F1(self):
    """
    Ecoli random walk simulation with food source function F1



    """


    #__________________________________________________________________________#
    ###Setting variables###
    iterationCount = self.iterationCount #Numebr of iterations

    tumblesPlusRun = self.tumbles + 1 #Set the tumble + run step size (t + n*dt)

    #Bring in F1 functions from self
    grad_F1_estimation = self.grad_F1_estimation

    grad_F1 = self.grad_F1



    #Set the Ecoli initial locations:
    X_init = self.X_init

    Y_init = self.Y_init

    #create column vectors to determine f(t: t+4dt) locations
    #Initially all rows of these columns = X_init or Y_init
    Ecoli_x = X_init*np.ones((tumblesPlusRun , iterationCount))

    Ecoli_y = Y_init*np.ones((tumblesPlusRun , iterationCount))





    #steps_X = the np.cumsum of Random steps (tumble) in the X-direction
    #steps_Y = the np.cumsum of Random steps (tumble) in the Y-direction
    steps_x, steps_y = randSteps(stepSize_mu = self.stepSize_mu,
                                  stepSize_std = self.stepSize_std,
                                  N_steps = self.tumbles)

    #__________________________________________________________________________#

    ###Initial Steps###

    Ecoli_x[:,0] += steps_x[:,0] #Add the cumsum to X_init

    Ecoli_y[:,0] += steps_y[:,0] #Add the cumsum to Y_init

    #__________________________________________________________________________#


    ###Loop through remaining steps with a similar logic to the initial step###


    for runIter in range(1,iterationCount):

        runStep_Magnitude_eps = .8 #set the run step length

        prev_X = Ecoli_x[:,runIter-1] #save the previous run+tumble steps
        prev_Y = Ecoli_y[:,runIter-1] #save the previous run+tumble steps



        #Determine the run vector direction and multiply the length of the vector by runStep_Magnitude_eps
            #User choice of using estimated partial derivatives or actual partial derivatives
        if self.grad_Est == True:

            runXY = runStep_Magnitude_eps * grad_F1_estimation( X = prev_X , Y = prev_Y)

        else:

            runXY = runStep_Magnitude_eps * grad_F1( X = prev_X , Y = prev_Y )


        #______________________________________________________________________#


        #Determine first step (similar to init_X, init_Y before loop)

        #Set all of the current step of Ecoli_x, Ecoli_y =
        #to the value of (the last tumble step) + (the run step)

        Ecoli_x[:,runIter] = prev_X[-1] + runXY[0]

        Ecoli_y[:,runIter] = prev_Y[-1] + runXY[1]



        #steps_X = the np.cumsum of Random steps (tumble) in the X-direction
        #steps_Y = the np.cumsum of Random steps (tumble) in the Y-direction
        steps_x, steps_y = randSteps(stepSize_mu=.6)#NEEDS UPDATED

        Ecoli_x[:,runIter] += steps_x[:,0]

        Ecoli_y[:,runIter] += steps_y[:,0]

    self.Ecoli_x = Ecoli_x
    self.Ecoli_y = Ecoli_y

    #__________________________________________________________________________#





    #Reshape Ecoli arrays for plotting (from 2D to 1D) by stacking column(i)
    #below column(i-1):
    plotX = np.reshape(Ecoli_x.transpose(),(iterationCount*tumblesPlusRun,))
    plotY = np.reshape(Ecoli_y.transpose(),(iterationCount*tumblesPlusRun,))


    #Use plot_Ecloi function to genatre the plot
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

  #Needs work due to updated code and array dimensions
  def plot_histogram(self):
    figure, axes = plt.subplots(nrows= 5, ncols = 1)
    empty_patch = mpatches.Patch(color = 'none')
    bins = 50
    max_distance = np.sqrt((self.Ecoli_x[0, 0] - self.a)**2 + (self.Ecoli_y[0, 0] - self.b)**2)

    #  I = 1
    distance = np.sqrt((self.Ecoli_x[0, 1] - self.a)**2 + (self.Ecoli_y[0, 1] - self.b)**2)
    axes[0].hist(distance, bins= bins, range=[0, max_distance], color = "black")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(empty_patch)
    labels.append("I = 1")
    axes[0].legend(handles, labels,loc='upper left', handlelength = 0, handleheight = 0)

    #  I = 10
    distance = np.sqrt((self.Ecoli_x[0, 9] - self.a)**2 + (self.Ecoli_y[0, 9] - self.b)**2)
    axes[1].hist(distance, bins= bins, range=[0, max_distance], color = "black")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(empty_patch)
    labels.append("I = 10")
    axes[1].legend(handles, labels, loc='upper left', handlelength = 0, handleheight = 0)

    #  I = 50
    distance = np.sqrt((self.Ecoli_x[0, 49] - self.a)**2 + (self.Ecoli_y[0, 49] - self.b)**2)
    axes[2].hist(distance, bins= bins, range=[0, max_distance], color = "black")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(empty_patch)
    labels.append("I = 50")
    axes[2].legend(handles, labels, loc='upper left', handlelength = 0, handleheight = 0)

    #  I = 100
    distance = np.sqrt((self.Ecoli_x[0, 99] - self.a)**2 + (self.Ecoli_y[0, 99] - self.b)**2)
    axes[3].hist(distance, bins= bins, range=[0, max_distance], color = "black")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(empty_patch)
    labels.append("I = 100")
    axes[3].legend(handles, labels, loc='upper left', handlelength = 0, handleheight = 0)

    #  I = 1000
    distance = np.sqrt((self.Ecoli_x[0, 999] - self.a)**2 + (self.Ecoli_y[0, 999] - self.b)**2)
    axes[4].hist(distance, bins= bins, range=[0, max_distance], color = "black")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(empty_patch)
    labels.append("I = 1000")
    axes[4].legend(handles, labels, loc='upper left', handlelength = 0, handleheight = 0)

    figure.supxlabel("distance from source")
    figure.supylabel("number of Ecoli")

    plt.tight_layout()
    plt.show()











#___________________________Previous Code___________________________#
#use gaussian/normal distribution to define continuous concentration field representing food source
#gradient maximum = (0,0) = highest food concentration
# def food_source(x, y, x0 = 0, y0 = 0, sigma = 3):
#   return np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))

#create concentration field gradient, 3 profiles total
#picked numbers
# def conc_exponential(x, y, alpha=0.5):
#   r = np.sqrt(x*x + y*y)
#   return np.exp(-alpha * r)

# def conc_linear(x, y, k = 0.2):
#   return k * x

#not sure if this is the best practice but I am creating a constant
#i.e right now, food_source is selected --> gaussian/normal distribution
# FIELD = food_source

#helper function to apply FIELD, can put this in main too
# def field_function(profile):
#     return np.vectorize(lambda x, y: profile(x, y))


  # def __init__(self):
  #   self.x = np.random.uniform(-10, 10)  #randomize initial position
  #   self.y = np.random.uniform(-10, 10)
  #   self.time_step = 0
  #   self.Ecolipath.append(np.array([self.x, self.y], float))    #store positions in list, starting w/ initial position

  # def tumble(self):
  #   for i in range(4):
  #     self.time_step += 1
  #     if self.time_step % 4 == 0:
  #       step_x = np.random.normal(-1.0, 1.0)
  #       step_y = np.random.normal(-1.0, 1.0)       #randomize steps
  #       self.x += step_x
  #       self.y += step_x              #update current position
  #       self.Ecolipath.append(np.array([self.x, self.y], float))  #changed to 2D

  # def run():
  #   pass

  # def cycle():
  #   pass

  # def plot_traj(field, cell, L = 15):
  #   pass

  # def plot_histograms(dist_by_time):

  #   pass

  # #run whole
  # if __name__ == "__main__":
  #   field = field_function(FIELD) #calling helper function depending on profile





