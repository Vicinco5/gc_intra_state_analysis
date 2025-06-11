import numpy as np
import matplotlib.pyplot as plt

# Parameters for the system
num_steps = 1000
attractors = [np.array([0, 0]), np.array([5, 5]), np.array([-5, -5])]
hop_points = [300, 600]

# Initialize system state
state = np.array([2, 2])
states = []
times = np.arange(num_steps)

# Function to update the state
def update_state(state, attractor):
    noise = np.random.normal(0, 0.1, size=2)
    direction = attractor - state
    new_state = state + 0.05 * direction + noise
    return new_state

# Simulate the system
current_attractor = attractors[0]
for t in times:
    if t in hop_points:
        current_attractor = attractors[hop_points.index(t) + 1]
    state = update_state(state, current_attractor)
    states.append(state)

states = np.array(states)

# Plotting the results
fig, ax = plt.subplots(3, 2, figsize=(14, 12))

# Plot the state in 2D space
ax[0, 0].plot(states[:, 0], states[:, 1], label='Trajectory')
ax[0, 0].scatter(*zip(*attractors), c='red', label='Attractors')
ax[0, 0].legend()
ax[0, 0].set_title('2D Trajectory of the System')
ax[0, 0].set_xlabel('State Dimension 1')
ax[0, 0].set_ylabel('State Dimension 2')

# Plot the x dimension over time
ax[0, 1].plot(times, states[:, 0], label='State Dimension 1')
ax[0, 1].axvline(x=hop_points[0], color='r', linestyle='--', label='Hop Point')
ax[0, 1].axvline(x=hop_points[1], color='r', linestyle='--')
ax[0, 1].set_title('State Dimension 1 Over Time')
ax[0, 1].set_xlabel('Time')
ax[0, 1].set_ylabel('State Dimension 1')
ax[0, 1].legend()

# Plot the y dimension over time
ax[1, 0].plot(times, states[:, 1], label='State Dimension 2')
ax[1, 0].axvline(x=hop_points[0], color='r', linestyle='--', label='Hop Point')
ax[1, 0].axvline(x=hop_points[1], color='r', linestyle='--')
ax[1, 0].set_title('State Dimension 2 Over Time')
ax[1, 0].set_xlabel('Time')
ax[1, 0].set_ylabel('State Dimension 2')
ax[1, 0].legend()

# Combined plot of both dimensions over time
ax[1, 1].plot(times, states[:, 0], label='State Dimension 1')
ax[1, 1].plot(times, states[:, 1], label='State Dimension 2')
ax[1, 1].axvline(x=hop_points[0], color='r', linestyle='--', label='Hop Point')
ax[1, 1].axvline(x=hop_points[1], color='r', linestyle='--')
ax[1, 1].set_title('State Dimensions Over Time')
ax[1, 1].set_xlabel('Time')
ax[1, 1].set_ylabel('State')
ax[1, 1].legend()

# Basin of Attraction (simplified visualization)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        dists = [np.linalg.norm(np.array([X[i, j], Y[i, j]]) - attractor) for attractor in attractors]
        Z[i, j] = np.argmin(dists)

ax[2, 0].contourf(X, Y, Z, levels=len(attractors) - 1, cmap='viridis')
ax[2, 0].scatter(*zip(*attractors), c='red', label='Attractors')
ax[2, 0].set_title('Basin of Attraction')
ax[2, 0].set_xlabel('State Dimension 1')
ax[2, 0].set_ylabel('State Dimension 2')

# Potential Landscape (simplified visualization)
potential = lambda x, y: np.min([np.linalg.norm(np.array([x, y]) - attractor) for attractor in attractors])
Z = np.vectorize(potential)(X, Y)

ax[2, 1].contourf(X, Y, Z, levels=20, cmap='viridis')
ax[2, 1].scatter(*zip(*attractors), c='red', label='Attractors')
ax[2, 1].set_title('Potential Landscape')
ax[2, 1].set_xlabel('State Dimension 1')
ax[2, 1].set_ylabel('State Dimension 2')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters for the system
num_steps = 1000
attractors = [np.array([0, 0]), np.array([5, 5]), np.array([-5, -5])]
hop_points = [300, 600]

# Initialize system state
state = np.array([2, 2])
states = []
times = np.arange(num_steps)

# Function to update the state for limit cycle
def update_state(state, attractor, omega):
    noise = np.random.normal(0, 0.1, size=2)
    theta = np.arctan2(state[1] - attractor[1], state[0] - attractor[0])
    r = np.linalg.norm(state - attractor)
    new_state = attractor + np.array([r * np.cos(theta + omega), r * np.sin(theta + omega)]) + noise
    return new_state

# Simulate the system with changing dynamics
current_attractor = attractors[0]
omega = 0.1  # Angular velocity for limit cycle
for t in times:
    if t in hop_points:
        current_attractor = attractors[hop_points.index(t) + 1]
    state = update_state(state, current_attractor, omega)
    states.append(state)

states = np.array(states)

# Plotting the results
fig, ax = plt.subplots(3, 2, figsize=(14, 12))

# Plot the state in 2D space (Phase Portrait)
ax[0, 0].plot(states[:, 0], states[:, 1], label='Trajectory')
ax[0, 0].scatter(*zip(*attractors), c='red', label='Attractors')
ax[0, 0].legend()
ax[0, 0].set_title('Phase Portrait (Limit Cycle)')
ax[0, 0].set_xlabel('State Dimension 1')
ax[0, 0].set_ylabel('State Dimension 2')

# Plot the x dimension over time
ax[0, 1].plot(times, states[:, 0], label='State Dimension 1')
ax[0, 1].axvline(x=hop_points[0], color='r', linestyle='--', label='Hop Point')
ax[0, 1].axvline(x=hop_points[1], color='r', linestyle='--')
ax[0, 1].set_title('State Dimension 1 Over Time')
ax[0, 1].set_xlabel('Time')
ax[0, 1].set_ylabel('State Dimension 1')
ax[0, 1].legend()

# Plot the y dimension over time
ax[1, 0].plot(times, states[:, 1], label='State Dimension 2')
ax[1, 0].axvline(x=hop_points[0], color='r', linestyle='--', label='Hop Point')
ax[1, 0].axvline(x=hop_points[1], color='r', linestyle='--')
ax[1, 0].set_title('State Dimension 2 Over Time')
ax[1, 0].set_xlabel('Time')
ax[1, 0].set_ylabel('State Dimension 2')
ax[1, 0].legend()

# Combined plot of both dimensions over time
ax[1, 1].plot(times, states[:, 0], label='State Dimension 1')
ax[1, 1].plot(times, states[:, 1], label='State Dimension 2')
ax[1, 1].axvline(x=hop_points[0], color='r', linestyle='--', label='Hop Point')
ax[1, 1].axvline(x=hop_points[1], color='r', linestyle='--')
ax[1, 1].set_title('State Dimensions Over Time')
ax[1, 1].set_xlabel('Time')
ax[1, 1].set_ylabel('State')
ax[1, 1].legend()

# Poincaré Section
crossings = states[(times % 100 == 0)]
ax[2, 0].scatter(crossings[:, 0], crossings[:, 1], c='blue', label='Poincaré Section')
ax[2, 0].set_title('Poincaré Section')
ax[2, 0].set_xlabel('State Dimension 1')
ax[2, 0].set_ylabel('State Dimension 2')
ax[2, 0].legend()

# Potential Landscape (simplified visualization)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        dists = [np.linalg.norm(np.array([X[i, j], Y[i, j]]) - attractor) for attractor in attractors]
        Z[i, j] = np.min(dists)

ax[2, 1].contourf(X, Y, Z, levels=20, cmap='viridis')
ax[2, 1].scatter(*zip(*attractors), c='red', label='Attractors')
ax[2, 1].set_title('Potential Landscape')
ax[2, 1].set_xlabel('State Dimension 1')
ax[2, 1].set_ylabel('State Dimension 2')

plt.tight_layout()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Parameters for the system
num_steps = 1000
attractors = [np.array([3, 3]), np.array([8, 8])]
hop_points = [500]

# Initialize system state
state = np.array([1, 1])
states = []
times = np.arange(num_steps)

# Function to update the state for limit cycle
def update_state(state, attractor, omega):
    noise = np.random.normal(0, 0.1, size=2)
    theta = np.arctan2(state[1] - attractor[1], state[0] - attractor[0])
    r = np.linalg.norm(state - attractor)
    # Ensure r is within a certain range to keep limit cycle sizes similar
    r = max(1.5, min(2.5, r))
    new_state = attractor + np.array([r * np.cos(theta + omega), r * np.sin(theta + omega)]) + noise
    return new_state

# Simulate the system with changing dynamics
current_attractor = attractors[0]
omega = 0.1  # Angular velocity for limit cycle
for t in times:
    if t in hop_points:
        current_attractor = attractors[hop_points.index(t) + 1]
    state = update_state(state, current_attractor, omega)
    states.append(state)

states = np.array(states)

# Ensure that the absolute minimum for both hypothetical neurons is zero
min_state_values = np.min(states, axis=0)
states -= min_state_values

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the state in 2D space (Phase Portrait)
ax[0].plot(states[:, 0], states[:, 1], label='Trajectory')
ax[0].scatter(*zip(*attractors), c='red', label='Attractors')
ax[0].legend()
ax[0].set_title('Limit Cycle Phase Portrait')
ax[0].set_xlabel('Hypothetical Neuron 1 Firing Rate')
ax[0].set_ylabel('Hypothetical Neuron 2 Firing Rate')

# Combined plot of both dimensions over time
ax[1].plot(times, states[:, 0], label='Hypothetical Neuron 1 Firing Rate')
ax[1].plot(times, states[:, 1], label='Hypothetical Neuron 2 Firing Rate')
ax[1].axvline(x=hop_points[0], color='r', linestyle='--', label='Changepoint')
ax[1].set_title('Hypothetical Ensemble Behavior Over Time')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Firing Rate (Hz)')
ax[1].legend()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters for the system
num_steps = 600
attractors = [np.array([1, 1]), np.array([5,7]), np.array([7,5])]
hop_points = [150, 400]

# Initialize system state
state = np.array([1, 1])
states = []
times = np.arange(num_steps)

# Function to update the state for point attractor
def update_state(state, attractor):
    noise = np.random.normal(0, 0.03, size=2)
    direction = attractor - state
    new_state = state + (0.04 * direction) + noise
    return new_state

# Simulate the system with changing dynamics
current_attractor = attractors[0]
for t in times:
    if t in hop_points:
        current_attractor = attractors[hop_points.index(t) + 1]
    state = update_state(state, current_attractor)
    states.append(state)

states = np.array(states)

# Ensure that the absolute minimum for both hypothetical neurons is zero
min_state_values = np.min(states, axis=0)
#states -= min_state_values

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the state in 2D space (Phase Portrait)
ax[0].plot(states[:, 0], states[:, 1], label='Ensemble Trajectory')
ax[0].scatter(*zip(*attractors), c='red', label='Attractors')
ax[0].legend()
ax[0].set_title('Point Attractor Phase Portrait')
ax[0].set_xlabel('Hypothetical Neuron 1 Firing Rate')
ax[0].set_ylabel('Hypothetical Neuron 2 Firing Rate')

# Combined plot of both dimensions over time
ax[1].plot(times, states[:, 0], label='Hypothetical Neuron 1 Firing Rate')
ax[1].plot(times, states[:, 1], label='Hypothetical Neuron 2 Firing Rate')
ax[1].axvline(x=hop_points[0], color='r', linestyle='--', label='Changepoint')
ax[1].set_title('Hypothetical Ensemble Behavior Over Time')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Firing Rate (Hz)')
ax[1].legend()

plt.tight_layout()
plt.show()




## POINT ATTRACTOR WE LIKE 


import numpy as np
import matplotlib.pyplot as plt

# Parameters for the system
num_steps = 2500
attractors = [np.array([2, 2]), np.array([4, 1]), np.array([2, 6]), np.array([7, 7])]
hop_points = [500, 1300, 2000]

# Custom colors for shading
shading_colors = ['#FC6ECB', '#BC7FBC', '#E8BBDF', '#C7CEE7']

# Custom colors for the phase portrait
trajectory_color = '#185682'  
attractor_color = '#D00000'  # Red-Orange

# Font size control
title_fontsize = 18
label_fontsize = 18
legend_fontsize = 16
ticks_fontsize = 16

# Initialize system state
state = np.array([2, 2])
states = []
times = np.arange(num_steps)

# Function to update the state for point attractor
def update_state(state, attractor):
    noise = np.random.normal(0, 0.01, size=2)  # Reduced noise for slower convergence
    direction = attractor - state 
    new_state = state + 0.008 * direction + noise  # Slower convergence factor
    return new_state

# Simulate the system with changing dynamics
current_attractor = attractors[0]
for t in times:
    if t in hop_points:
        current_attractor = attractors[hop_points.index(t) + 1]
    state = update_state(state, current_attractor)
    states.append(state)

states = np.array(states)

# Calculate the average of both hypothetical neurons
average_firing_rate = np.mean(states, axis=1)

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot the state in 2D space (Phase Portrait)
ax[0].plot(states[:, 0], states[:, 1], color=trajectory_color, label='Trajectory')
ax[0].scatter(*zip(*attractors), c=attractor_color, label='Attractors')
ax[0].legend(fontsize=legend_fontsize)
ax[0].set_title('Point Attractor Phase Portrait', fontsize=title_fontsize)
ax[0].set_xlabel('Hypothetical Neuron 1 Firing Rate', fontsize=label_fontsize)
ax[0].set_ylabel('Hypothetical Neuron 2 Firing Rate', fontsize=label_fontsize)
ax[0].tick_params(axis='both', labelsize=ticks_fontsize)

for i in range(len(hop_points) + 1):
    if i == 0:
        start = 0
    else:
        start = hop_points[i-1]
    if i == len(hop_points):
        end = num_steps
    else:
        end = hop_points[i]
    
    # Shade background
    ax[1].axvspan(start, end, color=shading_colors[i % len(shading_colors)], alpha=0.3)
    
    # Draw expected attractor values only for this segment
    attractor = attractors[i]  # Get the i-th attractor
    ax[1].hlines(y=attractor[0], xmin=start, xmax=end, color='#011638', linestyle='--', linewidth=1.5, alpha=0.7)  # Neuron 1
    ax[1].hlines(y=attractor[1], xmin=start, xmax=end, color='#185674', linestyle='--', linewidth=1.5, alpha=0.7)  # Neuron 2


ax[1].plot(times, states[:, 0], label='Neuron 1 Firing Rate', color='#011638')
ax[1].plot(times, states[:, 1], label='Neuron 2 Firing Rate', color='#185674')
ax[1].plot(times, average_firing_rate, label='Average Firing Rate', color='red', linestyle='-', linewidth=2)

# Add expected attractor firing rate lines
# for attractor in attractors:
#     ax[1].axhline(y=attractor[0], color='red', linestyle='--', linewidth=1.5, alpha=0.7)  # Neuron 1
#     ax[1].axhline(y=attractor[1], color='red', linestyle='--', linewidth=1.5, alpha=0.7)  # Neuron 2

# Changepoint vertical dashed lines
ax[1].axvline(x=hop_points[0], color='#D00000', linestyle='--', label='Changepoint')
ax[1].axvline(x=hop_points[1], color='#D00000', linestyle='--')
ax[1].axvline(x=hop_points[2], color='#D00000', linestyle='--')

ax[1].set_title('Point Attractor Behavior Over Time', fontsize=title_fontsize)
ax[1].set_xlabel('Time (ms)', fontsize=label_fontsize)
ax[1].set_ylabel('Firing Rate (Hz)', fontsize=label_fontsize)
ax[1].tick_params(axis='both', labelsize=ticks_fontsize)
ax[1].legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.show()



### LIMIT CYCLE ONE WE LIKE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Parameters for the system
num_steps = 2500
attractors = [np.array([3, 3]), np.array([8, 8]), np.array([3, 8]), np.array([6, 3])]
hop_points = [700, 1500, 2000]

# Custom colors for shading
shading_colors = ['#FC6ECB', '#BC7FBC', '#E8BBDF', '#C7CEE7']

# Custom colors for the phase portrait
trajectory_color = '#185682'  # Blue
attractor_color = '#D00000'  # Red

# Font size control
title_fontsize = 18
label_fontsize = 18
legend_fontsize = 14
ticks_fontsize = 16

# Initialize system state
state = np.array([1, 1])
states = []
times = np.arange(num_steps)

# Function to update the state for limit cycle
def update_state(state, attractor, omega):
    noise = np.random.normal(0, 0.03, size=2)  # Reduced noise for smoother limit cycles
    theta = np.arctan2(state[1] - attractor[1], state[0] - attractor[0])
    r = np.linalg.norm(state - attractor)
    # Ensure r is within a certain range to keep limit cycle sizes similar
    r = max(1, min(2, r))
    new_state = attractor + np.array([r * np.cos(theta + omega), r * np.sin(theta + omega)]) + noise
    return new_state

# Simulate the system with changing dynamics
current_attractor = attractors[0]
omega = 0.1  # Angular velocity for limit cycle
for t in times:
    if t in hop_points:
        current_attractor = attractors[hop_points.index(t) + 1]
    state = update_state(state, current_attractor, omega)
    states.append(state)

states = np.array(states)

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot the state in 2D space (Phase Portrait)
ax[0].plot(states[:, 0], states[:, 1], color=trajectory_color, label='Trajectory')
ax[0].scatter(*zip(*attractors), c=attractor_color, label='Attractors')
ax[0].legend(fontsize=legend_fontsize)
ax[0].set_title('Limit Cycle Phase Portrait', fontsize=title_fontsize)
ax[0].set_xlabel('Hypothetical Neuron 1 Firing Rate', fontsize=label_fontsize)
ax[0].set_ylabel('Hypothetical Neuron 2 Firing Rate', fontsize=label_fontsize)
ax[0].tick_params(axis='both', labelsize=ticks_fontsize)

# Combined plot of both dimensions over time with shading
for i in range(len(hop_points) + 1):
    if i == 0:
        start = 0
    else:
        start = hop_points[i-1]
    if i == len(hop_points):
        end = num_steps
    else:
        end = hop_points[i]
    ax[1].axvspan(start, end, color=shading_colors[i % len(shading_colors)], alpha=0.3)

    # Calculate and plot the trendline for each segment
    segment_times = times[start:end]
    combined_firing_rates = (states[start:end, 0] + states[start:end, 1]) / 2
    X_segment = segment_times.reshape(-1, 1)
    y_segment = combined_firing_rates

    model = LinearRegression()
    model.fit(X_segment, y_segment)
    trendline = model.predict(X_segment)

    ax[1].plot(segment_times, trendline, color='red', linestyle='-', linewidth=2)
    # Draw expected attractor values only for this segment
    attractor = attractors[i]  # Get the i-th attractor
    ax[1].hlines(y=attractor[0], xmin=start, xmax=end, color='#011638', linestyle='--', linewidth=1.5, alpha=0.7)  # Neuron 1
    ax[1].hlines(y=attractor[1], xmin=start, xmax=end, color='#185674', linestyle='--', linewidth=1.5, alpha=0.7)  # Neuron 2



# Plot the firing rates of each neuron
ax[1].plot(times, states[:, 0], label='Neuron 1 Firing Rate', color='#011638')
ax[1].plot(times, states[:, 1], label='Neuron 2 Firing Rate', color='#FF8C42')
ax[1].axvline(x=hop_points[0], color='#D00000', linestyle='--', label='Changepoint')
ax[1].axvline(x=hop_points[1], color='#D00000', linestyle='-', label='Firing Trend')
ax[1].axvline(x=hop_points[2], color='#D00000', linestyle='--')
ax[1].set_title('Limit Cycle Behavior Over Time', fontsize=title_fontsize)
ax[1].set_xlabel('Time (ms)', fontsize=label_fontsize)
ax[1].set_ylabel('Firing Rate (Hz)', fontsize=label_fontsize)
ax[1].tick_params(axis='both', labelsize=ticks_fontsize)
ax[1].legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.show()







