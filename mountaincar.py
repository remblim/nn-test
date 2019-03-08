import gym
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import random

env = gym.make('MountainCarContinuous-v0')
goal_steps = 500

def some_random_games_first(amount):
	approved_actions = []
	approved_observation = []
	completed = 0
	for episodes in range(amount):
		action_list = []
		observation_list = []
		env.reset()
		for t in range(goal_steps):
			#env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			action_list.append(action)
			observation_list.append(observation)
			if done:
				approved_actions.append(action_list)
				approved_observation.append(observation_list)
				completed += 1
				print(completed)
				break
		print(episodes)
				
def neural_network_model(input_size):
	network = input_data(shape = [None, input_size, 1], name='input')
	
	network = fully_connected(network,50,activation='tanh')
	network = dropout(network, 0.8)

	network = fully_connected(network,150,activation='tanh')
	network = dropout(network, 0.8)
	
	network = fully_connected(network,300,activation='tanh')
	network = dropout(network, 0.8)

	network = fully_connected(network,150,activation='tanh')
	network = dropout(network, 0.8)

	network = fully_connected(network,50,activation='tanh')
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network,optimizer='adam', learning_rate = LR, loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')
	
	return model
	
def train_model(X,Y):
	X = X.reshape(-1, len(X[0][0]), 1)
	Y = Y
	
	if not model:
		model = neural_network_model(input_size= len(X[0]))
	
	model.fit({'input':X},{'targets':Y}, n_epoch=1, snapshot_step=500, show_metric=True, run_id='openaistuff')
	
	return model
	
def game(max_episodes):
	for episodes in range(max_episodes):
		env.reset()
		prev_obs = []
		score = 0
		for t in range(goal_steps):
			if len(prev_obs) == 0:
				action = random.randrange(0,2)
			else:
				action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs),1))[0])
			new_observation, reward, done, info = env.step([action])
			prev_obs = new_observation
			train_model(new_observation,reward)
			score += reward
		print(score)
some_random_games_first(5000)