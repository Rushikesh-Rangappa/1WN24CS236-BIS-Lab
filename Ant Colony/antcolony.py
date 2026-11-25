
# ============================================================
# ANT COLONY OPTIMIZATION FOR WASTE COLLECTION ROUTE
# OPTIMIZATION - GOOGLE COLAB IMPLEMENTATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ------------------------------------------------------------
# 1. INITIALIZE THE PROBLEM
# ------------------------------------------------------------

np.random.seed(42)
random.seed(42)

# Number of garbage collection vehicles
NUM_VEHICLES = 3

# Maximum carrying capacity of each vehicle
VEHICLE_CAPACITY = 20

# ACO parameters
NUM_ANTS = 40
NUM_ITERATIONS = 100

ALPHA = 1.0       # Pheromone importance
BETA = 3.0        # Distance importance
RHO = 0.5         # Pheromone evaporation rate
Q = 100.0         # Pheromone deposit constant

# Vehicle operating parameters
AVERAGE_SPEED = 25       # km/hour
FUEL_RATE = 0.35          # liters/km
FUEL_PRICE = 100           # INR/liter

# ------------------------------------------------------------
# 2. CREATE DEPOT AND WASTE COLLECTION POINTS
# ------------------------------------------------------------

# Depot coordinates
depot = np.array([10.0, 10.0])

# Generate 18 waste collection points
NUM_BINS = 18

bin_coordinates = np.random.uniform(
    low=0,
    high=20,
    size=(NUM_BINS, 2)
)

# Each bin has a waste demand between 1 and 3 units
bin_demands = np.random.randint(
    1, 4, size=NUM_BINS
)

# Collection points are identified from 1 to 18
bin_ids = list(range(1, NUM_BINS + 1))

# Simulate bins becoming full at different times
# Phase 1: 8 bins
# Phase 2: 6 additional bins
# Phase 3: 4 additional bins

arrival_phases = [
    bin_ids[:8],
    bin_ids[8:14],
    bin_ids[14:]
]

# ------------------------------------------------------------
# 3. CALCULATE DISTANCE MATRIX
# ------------------------------------------------------------

# Node 0 represents the depot
# Nodes 1 to 18 represent waste collection bins

all_coordinates = np.vstack([
    depot,
    bin_coordinates
])

N = len(all_coordinates)

distance_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):

        distance_matrix[i][j] = np.linalg.norm(
            all_coordinates[i] - all_coordinates[j]
        )

print("Number of collection points:", NUM_BINS)
print("Number of vehicles:", NUM_VEHICLES)
print("Vehicle capacity:", VEHICLE_CAPACITY)
print("Total waste demand:", sum(bin_demands))


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------

def calculate_route_distance(route):
    """
    Calculate total distance of a route.
    A route starts and ends at the depot.
    """

    if len(route) == 0:
        return 0

    full_route = [0] + route + [0]

    total_distance = 0

    for i in range(len(full_route) - 1):

        current = full_route[i]
        next_node = full_route[i + 1]

        total_distance += distance_matrix[
            current
        ][
            next_node
        ]

    return total_distance


def calculate_total_distance(routes):
    """
    Calculate total distance of all vehicles.
    """

    return sum(
        calculate_route_distance(route)
        for route in routes
    )


def calculate_route_load(route):
    """
    Calculate total waste collected by a vehicle.
    """

    return sum(
        bin_demands[node - 1]
        for node in route
    )


# ------------------------------------------------------------
# 5. LOCAL SEARCH USING 2-OPT
# ------------------------------------------------------------

def two_opt(route):
    """
    Improve a route by reversing sections of the route.
    Keep improvements that reduce total distance.
    """

    best_route = route.copy()

    best_distance = calculate_route_distance(
        best_route
    )

    improved = True

    while improved:

        improved = False

        for i in range(len(best_route) - 1):

            for j in range(i + 1, len(best_route)):

                new_route = (
                    best_route[:i]
                    + best_route[i:j+1][::-1]
                    + best_route[j+1:]
                )

                new_distance = calculate_route_distance(
                    new_route
                )

                if new_distance < best_distance:

                    best_route = new_route
                    best_distance = new_distance

                    improved = True

    return best_route


# ------------------------------------------------------------
# 6. ANT COLONY OPTIMIZATION
# ------------------------------------------------------------

def ant_colony_optimization(active_bins):
    """
    Optimize waste collection routes for the active bins.

    Each ant creates routes for multiple vehicles.
    Every vehicle starts and ends at the depot.
    Vehicle capacity is not exceeded.
    """

    if len(active_bins) == 0:

        return [[] for _ in range(NUM_VEHICLES)], 0

    # Initialize pheromone matrix
    pheromone = np.ones((N, N))

    # Heuristic information:
    # Closer bins have higher heuristic values
    heuristic = 1 / (distance_matrix + 1e-10)

    np.fill_diagonal(heuristic, 0)

    best_routes = None
    best_distance = float("inf")

    # --------------------------------------------------------
    # MAIN ACO ITERATION
    # --------------------------------------------------------

    for iteration in range(NUM_ITERATIONS):

        all_ant_routes = []
        all_ant_distances = []

        # Each ant constructs a complete solution
        for ant in range(NUM_ANTS):

            unvisited = set(active_bins)

            routes = []

            # Assign collection points to vehicles
            for vehicle in range(NUM_VEHICLES):

                current_node = 0

                current_load = 0

                route = []

                while len(unvisited) > 0:

                    # Find bins that fit in vehicle capacity
                    feasible_bins = [
                        node
                        for node in unvisited
                        if (
                            current_load
                            + bin_demands[node - 1]
                            <= VEHICLE_CAPACITY
                        )
                    ]

                    # Move to next vehicle if no bins fit
                    if len(feasible_bins) == 0:
                        break

                    # Calculate selection probabilities
                    probabilities = []

                    for node in feasible_bins:

                        tau = pheromone[
                            current_node
                        ][
                            node
                        ] ** ALPHA

                        eta = heuristic[
                            current_node
                        ][
                            node
                        ] ** BETA

                        probabilities.append(
                            tau * eta
                        )

                    probabilities = np.array(
                        probabilities
                    )

                    probabilities /= probabilities.sum()

                    # Select next bin probabilistically
                    next_node = np.random.choice(
                        feasible_bins,
                        p=probabilities
                    )

                    # Update route and vehicle load
                    route.append(next_node)

                    current_load += bin_demands[
                        next_node - 1
                    ]

                    current_node = next_node

                    unvisited.remove(next_node)

                routes.append(route)

            # Reject solutions that cannot serve all bins
            if len(unvisited) > 0:
                continue

            # Apply local search to each vehicle route
            routes = [
                two_opt(route)
                for route in routes
            ]

            total_distance = calculate_total_distance(
                routes
            )

            all_ant_routes.append(routes)

            all_ant_distances.append(
                total_distance
            )

            # Keep the best solution
            if total_distance < best_distance:

                best_distance = total_distance

                best_routes = [
                    route.copy()
                    for route in routes
                ]

        # ----------------------------------------------------
        # PHEROMONE EVAPORATION
        # ----------------------------------------------------

        pheromone *= (1 - RHO)

        # ----------------------------------------------------
        # PHEROMONE DEPOSIT
        # ----------------------------------------------------

        for routes, distance in zip(
            all_ant_routes,
            all_ant_distances
        ):

            deposit = Q / distance

            for route in routes:

                full_route = [0] + route + [0]

                for i in range(len(full_route) - 1):

                    current = full_route[i]

                    next_node = full_route[i + 1]

                    pheromone[
                        current
                    ][
                        next_node
                    ] += deposit

                    pheromone[
                        next_node
                    ][
                        current
                    ] += deposit

        # Display progress
        if (iteration + 1) % 20 == 0:

            print(
                f"Iteration {iteration + 1}: "
                f"Best distance = {best_distance:.2f} km"
            )

    return best_routes, best_distance



# ------------------------------------------------------------
# 7. DYNAMIC WASTE COLLECTION
# ------------------------------------------------------------

collected_bins = set()
all_routes = []

for phase_number, new_bins in enumerate(arrival_phases, 1):

    active_bins = sorted(set(new_bins) - collected_bins)

    routes, distance = ant_colony_optimization(active_bins)

    collected_bins.update(active_bins)
    all_routes.append(routes)

    print(f"\nPhase {phase_number}: {len(active_bins)} new bins")
    print(f"Distance: {distance:.2f} km")

# ------------------------------------------------------------
# 8. FINAL RESULTS
# ------------------------------------------------------------

final_routes = all_routes[-1]

total_distance = sum(
    calculate_total_distance(routes)
    for routes in all_routes
)

total_time = total_distance / AVERAGE_SPEED * 60
fuel_cost = total_distance * FUEL_RATE * FUEL_PRICE

print("\nFINAL RESULTS")
print("-" * 30)

for i, route in enumerate(final_routes, 1):
    print(f"Vehicle {i}: {[0] + route + [0]}")

print(f"\nTotal bins collected: {len(collected_bins)}")
print(f"Total distance: {total_distance:.2f} km")
print(f"Estimated time: {total_time:.2f} minutes")
print(f"Estimated fuel cost: Rs. {fuel_cost:.2f}")

# ------------------------------------------------------------
# 9. FINAL ROUTE VISUALIZATION
# ------------------------------------------------------------

plot_routes(
    final_routes,
    "Final Waste Collection Routes"
)
