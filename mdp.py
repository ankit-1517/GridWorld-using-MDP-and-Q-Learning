#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


# In[2]:


def abs(x):
    if x < 0:
        return -x
    return x


# * 0 -- up 
# * 1 -- right
# * 2 -- down
# * 3 -- left

# In[3]:


# Choose the action using the action model in this function
def get_motion_direction(x):
    if np.random.random() < 0.8:
        return x
    return (x + np.random.randint(1, 4))%4


# In[4]:


def is_wall(x, y):
    if x == 0 or x == 49:
        return True
    if y == 0 or y == 24:
        return True
    if x == 25 or x == 26:
        if y != 12:
            return True
    return False


# In[5]:


def is_goal(x, y):
    if x == 48 and y == 12:
        return True
    return False


# In[6]:


def next_state(x, y, dir):
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
    if is_wall(xf, yf):
        return (x, y, True)
    return (xf, yf, False)


# In[7]:


def reward(x, y, hit_wall):
    if is_goal(x, y):
        return 100
    if hit_wall:
        return -1
    return 0


# In[30]:


def initialise_val_grid():
    val_grid = np.zeros((50, 25))
    for x in range(50):
        for y in range(25):
            val_grid[x][y] = np.random.random() + 0.0000001
    return val_grid


# In[9]:


def get_val_for_action(val_grid, x, y, discount_factor, action):
    val = 0
    for dir in range(4):
        xf, yf, hit_wall = next_state(x, y, dir)
        r = reward(xf, yf, hit_wall)
        probab = 0.2/3
        if dir == action:
            probab = 0.8
        val += probab*(r + discount_factor*val_grid[xf][yf])
    return val


# In[10]:


def get_new_value(val_grid, x, y, discount_factor):
    val = -np.inf
    act = 0
    for action in range(4):
        temp = get_val_for_action(val_grid, x, y, discount_factor, action)
        if temp > val:
            val = temp
            act = action
    return val, act


# In[11]:


def value_iteration(discount_factor, threshold, iterations):
    val_grid = initialise_val_grid()
    action_grid = np.zeros((50, 25))
    for itr in range(iterations):
        delta = 0
        for x in range(50):
            for y in range(25):
                if is_wall(x, y):
                    action_grid[x][y] = -1
                    val_grid[x][y] = 0
                    continue
                v = val_grid[x][y]
                val_grid[x][y], action_grid[x][y] = get_new_value(val_grid, x, y, discount_factor)
                delta = max(delta, abs(v - val_grid[x][y]))
        if delta < threshold:
            break
    return val_grid, action_grid


# In[52]:


def show_heatmap(arr, annot, fsize = (20, 15)):
    ax = plt.subplots(figsize = fsize)
    ax = sns.heatmap(arr.T, annot = annot, cmap = 'gray', linewidth=0.1)
    ax.add_patch(Rectangle((0, 0), 1, 25, color='blue'))
    ax.add_patch(Rectangle((0, 24), 50, 1, color='blue'))
    ax.add_patch(Rectangle((49, 0), 1, 25, color='blue'))
    ax.add_patch(Rectangle((0, 0), 50, 1, color='blue'))
    ax.add_patch(Rectangle((25, 0), 2, 12, color='blue'))
    ax.add_patch(Rectangle((25, 13), 2, 12,color='blue'))
    ax.add_patch(Rectangle((48, 12), 1, 1, edgecolor='red', fill=False, lw=2))
    ax.invert_yaxis()
    plt.show()


# In[ ]:





# In[60]:


def action_indicator_diag(arr, fsize):
    x = np.arange(0.5, 50.5, 1)
    y = np.arange(0.5, 25.5, 1)

    X, Y = np.meshgrid(x, y)
    u = np.zeros((25, 50))
    v = np.zeros((25, 50))
#     print(X.shape)
#     print(Y.shape)
#     print(u.shape)
#     print(v.shape)
    
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
    fig, ax = plt.subplots(figsize =fsize)
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


# In[31]:


def part_a_simple():
    np.random.seed(0)
    val_grid, action_grid = value_iteration(0.1, 0.1, 100)
    show_heatmap(val_grid, False)
    show_heatmap(action_grid, False)
    action_indicator_diag(action_grid, (20, 15))
    print(action_grid.T)
    

part_a_simple()


# In[27]:


def part_a():
    np.random.seed(0)
    val_grid, action_grid = value_iteration(0.1, 0.1, 10)
    print(val_grid[48])
    x = np.ptp(val_grid)
    y = np.min(val_grid) - 0.1
    print(y)
    u = val_grid - y
    print(x)
    show_heatmap(np.log((u)/x), False)
    show_heatmap(action_grid, False)

part_a()


# In[29]:


def part_b():
    np.random.seed(0)
    val_grid_20, action_grid = value_iteration(0.99, 0.1, 20)
    
    val_grid_50, _ = value_iteration(0.99, 0.1, 50)
    val_grid_100, _ = value_iteration(0.99, 0.1, 100)
    show_heatmap(val_grid_20, False)
    show_heatmap(val_grid_50, False)
    show_heatmap(val_grid_100, False)
    
part_b()
#     show_hea__grid)
#     action_indicator_diag(action_grid)
#     print(action_grid.T)


# In[114]:


def part_c():
    np.random.seed(0)
    val_grid, action_grid = value_iteration(0.99, 0.1, 100)
    action_indicator_diag(action_grid, (20, 15))

part_c()


# In[46]:


def policy_execution(arr, x, y, steps):
    pos_x = x
    pos_y = y
    pos_list_x = []
    pos_list_y = []
    pos_list_x.append(pos_x)
    pos_list_y.append(pos_y)
    visits = np.zeros((25, 50))
    visits[pos_y][pos_x] += 1
    for t in range(steps):
        r = get_motion_direction(arr[pos_x][pos_y])
        pos_x, pos_y, _ = next_state(pos_x, pos_y, r)
        pos_list_x.append(pos_x)
        pos_list_y.append(pos_y)
        visits[pos_y][pos_x] += 1
    return pos_list_x, pos_list_y, visits


# In[119]:


def part_c_a(arr, x, y, steps):
    l_x, l_y, _ = policy_execution(arr, x, y, steps)
    list_x = [n + 0.5 for n in l_x]
    list_y = [n + 0.5 for n in l_y]
    fig, ax = plt.subplots(figsize =(20, 15))
    plt.plot(list_x, list_y, 'bo')
    for i in range(len(list_x)-1):
        plt.plot([list_x[i], list_x[i+1]], [list_y[i], list_y[i+1]], 'k-')
    major_ticks_x = np.arange(0, 50, 1)
    
    major_ticks_y = np.arange(0, 25, 1)
    ax.add_patch(Rectangle((0, 0), 1, 25, alpha = 0.4))
    ax.add_patch(Rectangle((0, 24), 50, 1, alpha = 0.4))
    ax.add_patch(Rectangle((49, 0), 1, 25, alpha = 0.4))
    ax.add_patch(Rectangle((0, 0), 50, 1, alpha = 0.4))
    ax.add_patch(Rectangle((25, 0), 2, 12, alpha = 0.4))
    ax.add_patch(Rectangle((25, 13), 2, 12, alpha = 0.4))
    ax.add_patch(Rectangle((48, 12), 1, 1, color = 'red', alpha = 0.3))
    ax.axis([0, 50, 0, 25])
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.grid(True)
    plt.show()

np.random.seed(0)
val_grid, action_grid = value_iteration(0.99, 0.1, 100)
part_c_a(action_grid, 1, 1, 1000)


# In[54]:


def part_c_b(arr, x, y, steps, times):
    state_visit = np.zeros((25, 50))
    for t in range(times):
        _, _, state_visitation = policy_execution(arr, x, y, steps)
        state_visit += state_visitation
    show_heatmap(state_visit.T, True, (60, 45))
    show_heatmap(np.log(state_visit.T+1), False)


# In[118]:


np.random.seed(0)
val_grid, action_grid = value_iteration(0.99, 0.1, 100)
part_c_b(action_grid, 1, 1, 1000, 200)


# In[103]:


def value_iteration_for_maxnorm_nine(discount_factor, threshold):
    val_grid = initialise_val_grid()
    action_grid = np.zeros((50, 25))
    action_grid_prev = np.zeros((50, 25))
    i = 1
    k = 0
    r = np.random.randint(1, 51, 10) + 50*np.arange(10)
    delta_list = []
    while True:
        delta = 0
        action_grid_prev = action_grid.copy()
        for x in range(50):
            for y in range(25):
                if is_wall(x, y):
                    action_grid[x][y] = -1
                    val_grid[x][y] = 0
                    continue
                v = val_grid[x][y]
                val_grid[x][y], action_grid[x][y] = get_new_value(val_grid, x, y, discount_factor)
                delta = max(delta, abs(v - val_grid[x][y]))
        delta_list.append(delta)
        if i == 1:
            print(i)

            print(delta)
            action_indicator_diag(action_grid, (10, 8))
        if k < 10 and i == r[k]: 
            if np.array_equal(action_grid_prev, action_grid):
                print("CONVERGED_POLICY")
            print(i)

            print(delta)
            action_indicator_diag(action_grid, (10, 8))
            k += 1
#         elif i > 101 and i < 154:
#             print(i)

#             print(delta)
#             action_indicator_diag(action_grid, (10, 8))
        if np.array_equal(action_grid_prev, action_grid) == False:
            print("CHANGED_POLICY")
            print(i)

            print(delta)
            action_indicator_diag(action_grid, (10, 8))
        if delta < threshold:
            print(i)

            print(delta)
            action_indicator_diag(action_grid, (10, 8))
            break
        i += 1
    return val_grid, action_grid, delta_list


# In[115]:


def value_iteration_for_maxnorm_one(discount_factor, threshold, iterations):
    val_grid = initialise_val_grid()
    action_grid = np.zeros((50, 25))
    action_grid_prev = np.zeros((50, 25))
    i = 1
    m = 0
    k = np.random.randint(1, 4)
    delta_list = []
    while True:
        delta = 0
        action_grid_prev = action_grid.copy()
        for x in range(50):
            for y in range(25):
                if is_wall(x, y):
                    action_grid[x][y] = -1
                    val_grid[x][y] = 0
                    continue
                v = val_grid[x][y]
                val_grid[x][y], action_grid[x][y] = get_new_value(val_grid, x, y, discount_factor)
                delta = max(delta, abs(v - val_grid[x][y]))
        delta_list.append(delta)
        if np.array_equal(action_grid_prev, action_grid):
            print("CONVERGED_POLICY for {i}".format(i = i))
        if (i >= 19 and i <= 25):
            print(i)
            
            print(delta)
            action_indicator_diag(action_grid, (10, 8))
        if (i == k):
            print(i)
            
            print(delta)
            action_indicator_diag(action_grid, (10, 8))
            m += 1
            k = 3*m + np.random.randint(1, 4)
#         if np.array_equal(action_grid_prev, action_grid) == False:
#             print("CHANGED_POLICY")
#             print(i)

#             print(delta)
#             action_indicator_diag(action_grid, (10, 8))
        
        if i > iterations:
            print(i)

            print(delta)
            action_indicator_diag(action_grid, (10, 8))
            break
        i += 1
    return val_grid, action_grid, delta_list


# In[104]:


print("Discount Factor = 0.99")
np.random.seed(0)
_, _, delta_list1 = value_iteration_for_maxnorm_nine(0.99, 0.1)


# In[116]:


print("Discount Factor = 0.01")
np.random.seed(0)

_, _, delta_list2 = value_iteration_for_maxnorm_one(0.01, 1e-50, 100)


# In[112]:


#l = max(len(delta_list1), len(delta_list2))
x1 = np.arange(len(delta_list1)) + 1
x2 = np.arange(len(delta_list2)) + 1
plt.plot(x1, delta_list1, 'b', label = 'gamma = 0.99')
plt.plot(x2, delta_list2, 'r', label = 'gamma = 0.01')
plt.xlabel('Number of Iterations')
plt.ylabel('Maxnorm')
plt.legend()
plt.title('Maxnorm vs number of iterations for convergence of value iteration')
plt.show()

