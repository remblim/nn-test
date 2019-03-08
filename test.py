import gym
env = gym.make('MountainCarContinuous-v0')
goal_steps = 500
def some_random_games_first(amount):
	for episodes in range(amount):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = [0]
			observation, reward, done, info = env.step(action)
			print(reward)
			if done:
				break
	env.close()

some_random_games_first(1)