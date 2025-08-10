#team member names
#Natalia Rivera, Joseph Gill, Andy Ho, Nathalie Murphy, Sibhi Sakthivel, Yashesha Kothari

import numpy as np
import matplotlib as plt

#use gaussian/normal distribution to define continuous concentration field representing food source
#gradient maximum = (0,0) = highest food concentration
def food_source(x0 = 0, y0 = 0, x, y, sigma = 3):
  return np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))

#create concentration field gradient


#define E.coli class
class Ecoli:
  def __init__(self):
    self.x = np.random.uniform()  #randomize initial position
    self.Ecolipath = [self.x]     #store positions in list, starting w/ initial position
    
  def walking(self):
    step = np.random.normal()       #randomize steps
    self.x += step                  #update current position
    self.Ecolipath.append(self.x)   #save new x to positions list
    
