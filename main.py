#team member names
#Natalia Rivera, Joseph Gill, Andy Ho, Nathalie Murphy, Sibhi Sakthivel, Yashesha Kothari

import numpy as np
import matplotlib as plt

#use gaussian/normal distribution to define continuous concentration field representing food source
#gradient maximum = (0,0) = highest food concentration
def food_source(x, y, x0 = 0, y0 = 0, sigma = 3):
  return np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))

#create concentration field gradient, 3 profiles total
#picked numbers
def conc_exponential(x, y, alpha=0.5):
  r = np.sqrt(x*x + y*y)
  return np.exp(-alpha * r)

def conc_linear(x, y, k = 0.2):
  return k * x

#not sure if this is the best practice but I am creating a constant
#i.e right now, food_source is selected --> gaussian/normal distribution
FIELD = food_source

#helper function to apply FIELD, can put this in main too
def field_function(profile):
    return np.vectorize(lambda x, y: profile(x, y))

#define E.coli class
class Ecoli:
  
    # Each cycle:
    #1) Tumble: 4 small unbiased steps --> monte carlo
    #2) Sense: compare C(t) vs C(t-4Δt)
    #3) Run: move along recent heading if improved, else opposite --> gradient step
    

  def __init__(self):
    self.x = np.random.uniform(-10, 10)  #randomize initial position
    self.y = np.random.uniform(-10, 10)
    self.time_step = 0
    self.Ecolipath.append(np.array([self.x, self.y], float))    #store positions in list, starting w/ initial position
    
  def tumble(self):
    for i in range(4):
      self.time_step += 1
      if self.time_step % 4 == 0:
        step_x = np.random.normal(-1.0, 1.0)
        step_y = np.random.normal(-1.0, 1.0)       #randomize steps
        self.x += step_x   
        self.y += step_x              #update current position
        self.Ecolipath.append(np.array([self.x, self.y], float))  #changed to 2D

  def run():
    pass

  def cycle():
    pass

  def plot_traj(field, cell, L = 15):
    pass

  def plot_histograms(dist_by_time):
    pass

  #run whole
  if __name__ == "__main__":
    field = field_function(FIELD) #calling helper function depending on profile




    
