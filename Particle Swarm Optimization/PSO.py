import numpy as np
import matplotlib.pyplot as plt

# -------------------- 1. NETWORK PARAMETERS --------------------

np.random.seed(42)

NUM_NODES = 30
NUM_CONTROLLERS = 5
AREA_SIZE = 100

INITIAL_ENERGY_MIN = 0.5
INITIAL_ENERGY_MAX = 1.0

# Generate random sensor node locations
nodes = np.random.uniform(0, AREA_SIZE, (NUM_NODES, 2))

# Assign residual energy to each sensor node
energy = np.random.uniform(
    INITIAL_ENERGY_MIN,
    INITIAL_ENERGY_MAX,
    NUM_NODES
)

# -------------------- 2. DISTANCE CALCULATION --------------------

def calculate_distance(nodes):
    diff = nodes[:, None, :] - nodes[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

distance_matrix = calculate_distance(nodes)

# Normalize distances
MAX_DISTANCE = np.sqrt(2) * AREA_SIZE

# -------------------- 3. FITNESS FUNCTION --------------------

def fitness_function(particle):

    # Select control nodes using the highest particle values
    controller_indices = np.argsort(particle)[-NUM_CONTROLLERS:]

    # Calculate distances from all nodes to selected controllers
    distances = distance_matrix[:, controller_indices]

    # Assign each sensor to its nearest controller
    nearest_controller = np.argmin(distances, axis=1)

    minimum_distances = np.min(distances, axis=1)

    # Objective 1: Minimize transmission distance
    distance_cost = np.mean(minimum_distances) / MAX_DISTANCE

    # Objective 2: Prefer controllers with higher residual energy
    controller_energy = energy[controller_indices]
    energy_cost = np.mean(
        1 - controller_energy / INITIAL_ENERGY_MAX
    )

    # Objective 3: Balance the number of nodes per controller
    loads = np.bincount(
        nearest_controller,
        minlength=NUM_CONTROLLERS
    )

    load_cost = np.std(loads) / NUM_NODES

    # Combined fitness function (lower is better)
    fitness = (
        0.60 * distance_cost +
        0.30 * energy_cost +
        0.10 * load_cost
    )

    return fitness


# -------------------- 4. PSO PARAMETERS --------------------

NUM_PARTICLES = 40
MAX_ITERATIONS = 100

W = 0.7       # Inertia weight
C1 = 1.5      # Cognitive coefficient
C2 = 1.5      # Social coefficient

# Initialize particle positions and velocities
positions = np.random.uniform(
    0, 1, (NUM_PARTICLES, NUM_NODES)
)

velocities = np.random.uniform(
    -0.1, 0.1, (NUM_PARTICLES, NUM_NODES)
)

# Personal best initialization
personal_best_positions = positions.copy()

personal_best_scores = np.array([
    fitness_function(p) for p in positions
])

# Global best initialization
best_index = np.argmin(personal_best_scores)

global_best_position = personal_best_positions[
    best_index
].copy()

global_best_score = personal_best_scores[best_index]

convergence = []

# -------------------- 5. PSO OPTIMIZATION --------------------

for iteration in range(MAX_ITERATIONS):

    for i in range(NUM_PARTICLES):

        r1 = np.random.random(NUM_NODES)
        r2 = np.random.random(NUM_NODES)

        # Update velocity
        velocities[i] = (
            W * velocities[i]
            + C1 * r1 * (
                personal_best_positions[i] - positions[i]
            )
            + C2 * r2 * (
                global_best_position - positions[i]
            )
        )

        # Update position
        positions[i] = positions[i] + velocities[i]

        # Keep positions within limits
        positions[i] = np.clip(positions[i], 0, 1)

        # Evaluate fitness
        current_score = fitness_function(positions[i])

        # Update personal best
        if current_score < personal_best_scores[i]:

            personal_best_scores[i] = current_score
            personal_best_positions[i] = positions[i].copy()

        # Update global best
        if current_score < global_best_score:

            global_best_score = current_score
            global_best_position = positions[i].copy()

    convergence.append(global_best_score)

    if (iteration + 1) % 10 == 0:
        print(
            f"Iteration {iteration + 1}: "
            f"Best Fitness = {global_best_score:.6f}"
        )

# -------------------- 6. FINAL CONTROL NODE SELECTION --------------------

best_controllers = np.argsort(
    global_best_position
)[-NUM_CONTROLLERS:]

# Assign every sensor node to its nearest controller
final_distances = distance_matrix[:, best_controllers]

assignments = np.argmin(final_distances, axis=1)

minimum_distances = np.min(final_distances, axis=1)

loads = np.bincount(
    assignments,
    minlength=NUM_CONTROLLERS
)

# -------------------- 7. RESULTS --------------------

print("\n========== PSO OPTIMIZATION RESULTS ==========")

print("Total Sensor Nodes:", NUM_NODES)
print("Selected Control Nodes:", NUM_CONTROLLERS)

print("\nSelected Control Node IDs:")

for controller in best_controllers:
    print(
        f"Node {controller + 1} | "
        f"Residual Energy: {energy[controller]:.3f} J | "
        f"Position: {nodes[controller]}"
    )

print("\nNode Assignments:")

for i, controller_position in enumerate(best_controllers):
    assigned_nodes = np.where(assignments == i)[0]

    print(
        f"Controller Node {controller_position + 1}: "
        f"{[int(x + 1) for x in assigned_nodes]}"
    )

print("\nAverage Transmission Distance:",
      round(np.mean(minimum_distances), 3))

print("Estimated Fitness Value:",
      round(global_best_score, 6))

print("Average Controller Residual Energy:",
      round(np.mean(energy[best_controllers]), 3), "J")

print("Controller Loads:", loads)

# -------------------- 8. NETWORK VISUALIZATION --------------------

plt.figure(figsize=(10, 8))

# Plot all sensor nodes
plt.scatter(
    nodes[:, 0],
    nodes[:, 1],
    c='blue',
    s=70,
    label='Sensor Nodes'
)

# Plot selected control nodes
plt.scatter(
    nodes[best_controllers, 0],
    nodes[best_controllers, 1],
    c='red',
    marker='*',
    s=300,
    label='Selected Control Nodes'
)

# Draw communication links
for i in range(NUM_NODES):

    controller = best_controllers[assignments[i]]

    if i != controller:
        plt.plot(
            [nodes[i, 0], nodes[controller, 0]],
            [nodes[i, 1], nodes[controller, 1]],
            'gray',
            linestyle='--',
            alpha=0.5
        )

# Label nodes
for i in range(NUM_NODES):
    plt.text(
        nodes[i, 0] + 1,
        nodes[i, 1] + 1,
        str(i + 1),
        fontsize=8
    )

plt.title("PSO-Based Energy-Efficient WSN Control Node Selection")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.show()

# -------------------- 9. CONVERGENCE GRAPH --------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, MAX_ITERATIONS + 1),
    convergence,
    color='green',
    linewidth=2
)

plt.title("PSO Convergence Curve")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness Value")
plt.grid(True)
plt.show()

# -------------------- 10. ENERGY COMPARISON --------------------

plt.figure(figsize=(9, 5))

plt.bar(
    range(1, NUM_NODES + 1),
    energy,
    color='skyblue',
    label='Residual Energy'
)

plt.bar(
    best_controllers + 1,
    energy[best_controllers],
    color='red',
    label='Selected Control Nodes'
)

plt.title("Residual Energy of Sensor Nodes")
plt.xlabel("Sensor Node ID")
plt.ylabel("Energy (Joules)")
plt.legend()
plt.grid(axis='y')
plt.show()

print("\nPSO Optimization Completed Successfully!")
