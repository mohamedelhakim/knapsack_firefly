import random
import math
from scipy.spatial.distance import cdist 
import numpy as np
import operator
import collections
import copy

class KNAPSACKFF():
	def __init__(self,value,weight,capacity,number_of_items,num_of_pop, iterations=200, beta=1,alfa=1):
		self.value=value
		self.weight=weight
		self.capacity = capacity
		self.number_of_items=number_of_items
		self.num_of_pop=num_of_pop 
		self.iterations=iterations
		self.beta=beta
		self.alfa=alfa
		self.absorptions=[]
		self.population=[]
		self.light_intensities=[]
		for i in range(self.num_of_pop):
			pop=np.zeros(self.number_of_items)
			w=0
			for j in range(self.number_of_items):
				pop[j]=np.random.random()
				pop[j]=round(pop[j])
				if pop[j]*self.weight[j]+w>self.capacity:
					pop[j]=0
				w+= pop[j]*self.weight[j]                   
			self.population.append(pop)  

	def f(self,x):
		t = self.prof(x)
		return self.w_kg(x, t)
  
	def prof(self,x):
		total_prof = 0
		for i in range(len(x)):
			total_prof += x[i] * self.value[i]  # the sum of profits for taken objects
		return total_prof
	def w_kg(self,x, profit):
		total_kg = 0
		for i in range(len(x)):
			total_kg += x[i] * self.weight[i] #the sum of weights for taken objects
		if total_kg <= self.capacity:
			return profit 
		elif total_kg > self.capacity:
			return - profit
	def determine_initial_light_intensities(self):
		"initializes light intensities"
		for x in self.population :
			self.light_intensities.append(self.f(x) )

	def generate_initial_absorptions(self):
		for i in range(len(self.population)):
			self.absorptions.append(random.random()*0.1+0.1 )

	def check_if_best_solution(self, index):
		new_cost = self.light_intensities[index]
		if new_cost > self.best_solution_cost: 
			self.best_solution = copy.deepcopy(self.population[index])
			self.best_solution_cost = new_cost

	def find_global_optimum(self):
		"finds the brightest firefly"
		index = self.light_intensities.index(max(self.light_intensities))
		return index
	def move_firefly(self, a, b):
		"moving firefly a to b in less than r swaps"
		popc=copy.copy(self.population[a])
		r = math.sqrt(sum(((popc[i]-self.population[b][i])**2 for i in range(number_of_items)))) 
		for i in range(self.number_of_items):
			popc[i]=popc[i]+self.beta*math.exp(-1.0 *self.absorptions[a]* r**2)*(self.population[b][i]-popc[i])+self.alfa*(random.uniform(0,1)-0.5)
			popc[i]=round(popc[i])
			popc[i]=max(0,popc[i])
		if self.light_intensities[a] < self.f(popc) :
			self.light_intensities[a] = self.f(popc)
			self.population[a]=copy.copy(popc)
	def run(self):
		"gamma is parameter for light intensities, beta is size of neighbourhood according to hamming distance"
		# hotfix, will rewrite later
		self.generate_initial_absorptions()
		self.determine_initial_light_intensities()
		self.best_solution=self.population[self.find_global_optimum()] 
		self.best_solution_cost = self.f(self.best_solution)
		individuals_indexes = range(self.num_of_pop)
		self.n = 0
		while self.n < self.iterations:
			for j in individuals_indexes:
				for i in individuals_indexes:
					if self.light_intensities[i] < self.light_intensities[j] :
						self.move_firefly(i, j)						
						self.check_if_best_solution(i)

			self.n += 1
		return self.best_solution,self.best_solution_cost 
weight = [695093 ,430667 ,233583 ,832832, 422539, 614321 ,898576 ,371233, 324029 ,958551 ,381955 ,519075, 94236, 835863, 352292, 983000 ,286126 ,924119 ,679730 ,229377 ,152007 ,794048, 566176 ,729710]
value =[1319972, 804470, 487111 ,1636416, 915637, 1334476 ,1703095, 727928 ,588848, 1974348, 818432 ,1012735 ,181004 ,1826340 ,736446 ,2070580 ,597255, 1765740, 1374118, 467036, 310529 ,1614819 ,1157316 ,1370965]
capacity = 6654569  # max weight of knapsack
number_of_items = 24  # set of items to consider
KNAPSACK=KNAPSACKFF(value,weight,capacity,number_of_items,50,iterations=200, beta=1,alfa=1)
x,y=KNAPSACK.run()
print(y, x)