from numpy import matrix
from numpy import zeros
from numpy import argmax, max
from numpy.random import randint
import time

#kaart nummering
#1 is starts
#0 is safe
#2 is goal
#3 is bad

#actie nummering
#0 naar boven
#1 naar links
#2 naar onderen
#3 naar rechts
print('importeren klaar')
class frozenlake():
	def __init__(self):
		self.field = [[1,0,0,3,0],
				[0,3,0,3,0],
				[3,0,0,3,0],
				[0,3,0,0,3],
				[0,3,0,0,2]]
		self.positie = [0,0]
		self.done = False
		print('spel geinitialiseerd')
		
	def render(self):
		print(self.positie)
		print(matrix(self.field))
		
	def stap(self, actie):
		if self.done == False:
			reward = 0
			#acties uitvoeren
			if  actie == 0:
				self.positie[1] -= 1
			elif actie == 1:
				self.positie[0] -= 1
			elif actie == 2:
				self.positie[1] += 1
			elif actie == 3:
				self.positie[0] += 1
			
			#controleren of positie buiten het veld licht, en dan terug plaatsen
			if self.positie[1] < 0:
				self.positie[1] = 0
				reward = -0.25
			elif self.positie[1] > len(self.field)-1:
				self.positie[1] = len(self.field)-1
				reward = -0.25
			elif self.positie[0] < 0:
				self.positie[0] = 0
				reward = -0.25
			elif self.positie[0] > len(self.field[0])-1:
				self.positie[0] = len(self.field[0])-1
				reward = -0.25
			
			if self.positie == [4,4]:
				self.done = True
				reward = 1
			elif self.field[self.positie[1]][self.positie[0]] == 3:
				self.done = True
				reward = -1
			else:
				self.done = False
			if reward == 0:
				reward = -0.2
		else:
			print('het spel is afgelopen')
		
		return self.done, reward, self.positie
	
	def reset(self):
		self.positie = [0,0]
		self.done = False
class nn():
	def __init__(self):
		self.q_values = zeros((25,4))
		self.lr = .8
		self.y = 0.95
	
	def get(self,positie):
		action = argmax(self.q_values[positie[0] + positie[1] * 5])
		#action = int(input('input'))
		return action
		
	def update(self,positie,new_positie,action,reward):
		position = positie[0] + positie[1] * 5
		new_position = new_positie[0] + new_positie[1] * 5
		self.q_values[position,action] = reward + self.lr * max(self.q_values[new_position])
		
game = frozenlake()
network = nn()
done = False
episodes = 100
max_steps = 50
positie_oud = [0,0]
succes = 0
starttime = time.time()
for i in range(episodes):
	x = 0
	game.reset()
	done = False
	positie_oud = [0,0]
	while done == False and x in range(max_steps):
		actie = network.get(game.positie)
		done, reward, positie_new = game.stap(actie)
		network.update(positie_oud, positie_new, actie, reward)
		positie_oud = positie_new.copy()
		x += 1
		if reward == 1:
			succes += 1
endtime = time.time()
print('duurde',endtime-starttime)
print(succes,' gelukt van de episodes ',episodes)

print(network.q_values)