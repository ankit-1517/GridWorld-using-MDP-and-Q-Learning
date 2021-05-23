import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

'''
0 -- up
1 -- right
2 -- down
3 -- left
'''
def get_motion_direction(x, epsilon):
	if np.random.uniform(0,1) > epsilon:
		return x
	return np.random.randint(0, 4)

def is_wall(x, y):
	if x == 0 or x == 49:
		return True
	if y == 0 or y == 24:
		return True
	if x == 25 or x == 26:
		if y != 12:
			return True
	return False

def is_goal(x, y):
	if x == 48 and y == 12:
		return True
	return False

def reward(x, y, hit_wall):
	if is_goal(x, y):
		return 100
	if hit_wall:
		return -1
	return 0

def transition(x, y, dir, epsilon):
	# add arbitrariness
	dir = get_motion_direction(dir, epsilon)

	xf = x
	yf = y
	if dir == 0:
		yf += 1
	elif dir == 1:
		xf += 1
	elif dir == 2:
		yf -= 1
	else:
		xf -= 1
	return (xf, yf)

def get_next_state(qsa_grid, x, y, epsilon):
	dir = np.argmax(qsa_grid[x][y])
	xf, yf = transition(x, y, dir, epsilon)
	hit_wall = False
	if is_wall(xf, yf):
		hit_wall = True
		xf, yf = x, y
	r = reward(xf, yf, hit_wall)
	return (xf, yf, r, dir)

def initialise_qsa_grid():
	qsa_grid = np.zeros((50, 25, 4))
	for x in range(50):
		for y in range(25):
			for a in range(4):
				qsa_grid[x][y][a] = 1
	return qsa_grid

def get_start_state():
	x = 0
	y = 0
	while (is_wall(x, y) or is_goal(x, y)):
		x = np.random.randint(0, 50)
		y = np.random.randint(0, 25)
	return (x, y)

def run_one_episode(qsa_grid, learning_rate, epsilon, episode_length):
	discount_factor = 0.99
	x, y = get_start_state()
	reward_per_episode = 0
	for ep_len in range(episode_length):
		xf, yf, r, action = get_next_state(qsa_grid, x, y, epsilon)
		qsa_grid[x][y][action] += learning_rate*(r + discount_factor*np.max(qsa_grid[xf][yf]) - qsa_grid[x][y][action])
		reward_per_episode += r
		x, y = xf, yf
		if is_goal(xf, yf):
			break
	return qsa_grid, reward_per_episode

def q_learning(learning_rate, epsilon, episodes = 4000, episode_length = 1000):
	qsa_grid = initialise_qsa_grid()
	reward_per_episode = []
	for eps in range(episodes):
		qsa_grid, r = run_one_episode(qsa_grid, learning_rate, epsilon, episode_length)
		reward_per_episode.append(r)
	
	val_grid = np.zeros((50, 25))
	action_grid = np.zeros((50, 25))
	for x in range(qsa_grid.shape[0]):
		for y in range(qsa_grid.shape[1]):
			if is_wall(x, y) or is_goal(x, y):
				action_grid[x][y] = -1
			else:
				action_grid[x][y] = np.argmax(qsa_grid[x][y])
				val_grid[x][y] = np.max(qsa_grid[x][y])
	return (val_grid, action_grid, reward_per_episode)

def show_heatmap(arr):
	ax = sns.heatmap(arr.T, cmap='gray', linewidth=0.1)
	ax.invert_yaxis()
	ax.add_patch(Rectangle((48, 12), 1, 1, fill=False, edgecolor='red', lw=2))
	ax.add_patch(Rectangle((0, 0), 1, 25))
	ax.add_patch(Rectangle((0, 24), 50, 1))
	ax.add_patch(Rectangle((49, 0), 1, 25))
	ax.add_patch(Rectangle((0, 0), 50, 1))
	ax.add_patch(Rectangle((25, 0), 2, 12))
	ax.add_patch(Rectangle((25, 13), 2, 12))
	plt.show()

def plot_reward(r):
	y = [i for i in range(len(r))]
	plt.plot(y, r)
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.show()

def plot_reward_avg(r):
	r = [np.sum(r[10*i:10*(i+1)])/10 for i in range(len(r)//10)]
	y = [10*i for i in range(len(r))]
	plt.plot(y, r)
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.show()


def action_indicator_diag(arr):
	x = np.arange(0.5, 50.5, 1)
	y = np.arange(0.5, 25.5, 1)

	X, Y = np.meshgrid(x, y)
	u = np.zeros((25, 50))
	v = np.zeros((25, 50))
	
	for i in range(50):
		for j in range(25):
			if (arr[i][j] == 0):
				v[j][i] = 1
			elif (arr[i][j] == 1):
				u[j][i] = 1
			elif (arr[i][j] == 2):
				v[j][i] = -1
			else:
				u[j][i] = -1

	# creating plot
	# fig, ax = plt.subplots(figsize =fsize)
	fig, ax = plt.subplots()
	ax.quiver(X, Y, u, v, angles='xy', pivot='mid')

	major_ticks_x = np.arange(0, 50, 1)
	#minor_ticks_x = np.arange(0, 49, 1)
	major_ticks_y = np.arange(0, 25, 1)
	#minor_ticks_y = np.arange(0, 24, 1)

	ax.axis([0, 50, 0, 25])
	ax.set_xticks(major_ticks_x)
	ax.set_yticks(major_ticks_y)
	ax.add_patch(Rectangle((0, 0), 1, 25))
	ax.add_patch(Rectangle((0, 24), 50, 1))
	ax.add_patch(Rectangle((49, 0), 1, 25))
	ax.add_patch(Rectangle((0, 0), 50, 1))
	ax.add_patch(Rectangle((25, 0), 2, 12))
	ax.add_patch(Rectangle((25, 13), 2, 12))
	ax.add_patch(Rectangle((48, 12), 1, 1, color = 'red', alpha = 0.3))
	
	ax.grid(True)
	plt.minorticks_on
	ax.set_axisbelow(True)
  
	plt.show()


def part_b():
	np.random.seed(1)
	val_grid, action_grid, reward_per_episode = q_learning(0.25, 0.05)
	show_heatmap(val_grid)
	action_indicator_diag(action_grid)

def part_c():
	np.random.seed(1)
	val_grid, action_grid, reward_per_episode = q_learning(0.25, 0.005)
	show_heatmap(val_grid)
	action_indicator_diag(action_grid)

	np.random.seed(1)
	val_grid, action_grid, reward_per_episode = q_learning(0.25, 0.05)
	show_heatmap(val_grid)
	action_indicator_diag(action_grid)

	np.random.seed(1)
	val_grid, action_grid, reward_per_episode = q_learning(0.25, 0.5)
	show_heatmap(val_grid)
	action_indicator_diag(action_grid)

def part_d():
	np.random.seed(1)
	val_grid, action_grid, reward_per_episode = q_learning(0.25, 0.05)
	plot_reward(reward_per_episode)
	plot_reward_avg(reward_per_episode)

	np.random.seed(1)
	val_grid, action_grid, reward_per_episode = q_learning(0.25, 0.5)
	plot_reward(reward_per_episode)
	plot_reward_avg(reward_per_episode)

part_b()
part_c()
part_d()


