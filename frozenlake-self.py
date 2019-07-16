from numpy import matrix
from numpy import zeros
from numpy import argmax, max, full
from numpy.random import randint
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
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
		self.lr = .8
		self.y = 0.95
		self.x_replay = []
		self.y_replay = []
		self.network = input_data(shape = [None, 25, 1], name='input')
		self.network = fully_connected(self.network, 500, activation='tanh')
		self.network = fully_connected(self.network, 4, activation='softmax')
		self.network = regression(self.network,optimizer='SGD', learning_rate = 0.1, loss='mean_square', name='targets')
		self.model = tflearn.DNN(self.network)

	def get(self,positie):
		one_hot_positie = zeros(25)
		one_hot_positie[positie[0] + positie[1]*5] = 1
		q_value = self.model.predict(one_hot_positie.reshape(-1,25,1))
		action = argmax(q_value)
		#action = int(input('input'))
		return action
		
	def update(self,positie,new_positie,action,reward):
		one_hot_positie_old = zeros(25)
		one_hot_positie_old[positie[0] + positie[1]*5] = 1
		position = positie[0] + positie[1] * 5
		one_hot_positie_new = zeros(25)
		one_hot_positie_new[new_positie[0] + new_positie[1]*5] = 1
		new_position = new_positie[0] + new_positie[1] * 5
		q_value = reward + max(self.model.predict(one_hot_positie_new.reshape(-1,25,1)))
		q_train = full((4),0.5)
		q_train[action] = max(q_value)
		self.y_replay.append(q_train)
		self.x_replay.append(one_hot_positie_old.reshape(25,1))
		self.model.fit(self.x_replay, self.y_replay, n_epoch=1, show_metric=False)
		
game = frozenlake()
network = nn()
done = False
episodes = 100
max_steps = 50
positie_oud = [0,0]
succes = 0
starttime = time.time()
for i in range(episodes):
	print(succes)
	x = 0
	game.reset()
	done = False
	positie_oud = [0,0]
	while done == False and x in range(max_steps):
		actie = network.get(game.positie)
		done, reward, positie_new = game.stap(actie)
		reward = reward + 0.5
		network.update(positie_oud, positie_new, actie, reward)
		positie_oud = positie_new.copy()
		x += 1
		if reward == 1:
			succes += 1
endtime = time.time()
print('duurde',endtime-starttime)
print(succes,' gelukt van de episodes ',episodes)