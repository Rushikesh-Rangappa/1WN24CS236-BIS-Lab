
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. CONFIGURATION
# =====================================================

NUM_MACHINES = 4
NUM_JOBS = 12

POPULATION_SIZE = 40
GENERATIONS = 35
HEAD_LENGTH = 5

random.seed(42)
np.random.seed(42)

# =====================================================
# 2. GENERATE DYNAMIC JOB DATA
# =====================================================

jobs = []

for jid in range(NUM_JOBS):

    num_operations = random.randint(2, 4)

    machines = random.sample(
        range(NUM_MACHINES),
        num_operations
    )

    operations = []

    for machine in machines:
        processing_time = random.randint(2, 10)
        operations.append((machine, processing_time))

    arrival_time = random.randint(0, 15)

    jobs.append({
        "id": jid,
        "arrival": arrival_time,
        "operations": operations
    })

# Sort jobs by arrival time
jobs.sort(key=lambda x: x["arrival"])

job_map = {job["id"]: job for job in jobs}

print("========== JOB DETAILS ==========")

for job in jobs:
    print(
        f"Job {job['id']} | Arrival: {job['arrival']} | "
        f"Operations: {job['operations']}"
    )


# =====================================================
# 3. GEP FUNCTION SET
# =====================================================

FUNCTIONS = ["+", "-", "*", "min", "max"]
TERMINALS = ["p", "r", "w", "t"]

GENE_LENGTH = HEAD_LENGTH * 2 + 1


def generate_expression():

    gene = []

    for i in range(GENE_LENGTH):

        if i < HEAD_LENGTH:
            gene.append(
                random.choice(FUNCTIONS + TERMINALS)
            )
        else:
            gene.append(
                random.choice(TERMINALS)
            )

    return gene


def generate_chromosome():
    return generate_expression()


# =====================================================
# 4. GEP EXPRESSION EVALUATION
# =====================================================

def evaluate_gene(gene, p, r, w, t):

    values = {
        "p": p,
        "r": r,
        "w": w,
        "t": t
    }

    # Build expression tree using prefix notation
    tokens = iter(gene)

    def evaluate():

        try:
            symbol = next(tokens)
        except StopIteration:
            return 0

        if symbol in TERMINALS:
            return values[symbol]

        left = evaluate()
        right = evaluate()

        if symbol == "+":
            result = left + right

        elif symbol == "-":
            result = left - right

        elif symbol == "*":
            result = left * right

        elif symbol == "min":
            result = min(left, right)

        else:
            result = max(left, right)

        # Prevent extreme numerical values
        return max(-10000, min(10000, result))

    try:
        return evaluate()
    except (RecursionError, OverflowError):
        return 0


# =====================================================
# 5. COMMON SCHEDULING SIMULATOR
# =====================================================

def simulate_schedule(chromosome=None, policy="GEP"):

    machine_available = [0] * NUM_MACHINES

    job_available = {
        job["id"]: job["arrival"]
        for job in jobs
    }

    job_next_operation = {
        job["id"]: 0
        for job in jobs
    }

    schedule = []

    total_waiting = 0

    total_operations = sum(
        len(job["operations"]) for job in jobs
    )

    completed_operations = 0

    while completed_operations < total_operations:

        candidates = []

        # Find each job's next available operation
        for job in jobs:

            jid = job["id"]
            op_index = job_next_operation[jid]

            if op_index >= len(job["operations"]):
                continue

            machine, duration = job["operations"][op_index]

            start = max(
                job_available[jid],
                machine_available[machine]
            )

            waiting = max(
                0,
                start - job_available[jid]
            )

            remaining = (
                len(job["operations"]) - op_index
            )

            if policy == "GEP":

                priority = evaluate_gene(
                    chromosome,
                    duration,
                    remaining,
                    waiting,
                    start
                )

            elif policy == "FIFO":

                # Earliest arrival gets priority
                priority = -job["arrival"]

            candidates.append({
                "priority": priority,
                "job": jid,
                "machine": machine,
                "duration": duration,
                "start": start,
                "waiting": waiting,
                "arrival": job["arrival"]
            })

        if not candidates:
            break

        # Select the operation with the highest priority
        selected = max(
            candidates,
            key=lambda x: (
                x["priority"],
                -x["arrival"],
                -x["job"]
            )
        )

        jid = selected["job"]
        machine = selected["machine"]
        duration = selected["duration"]
        start = selected["start"]

        end = start + duration

        schedule.append({
            "job": jid,
            "machine": machine,
            "start": start,
            "end": end,
            "duration": duration
        })

        total_waiting += selected["waiting"]

        machine_available[machine] = end
        job_available[jid] = end

        job_next_operation[jid] += 1

        completed_operations += 1

    makespan = max(machine_available)

    return makespan, total_waiting, schedule


# =====================================================
# 6. FITNESS FUNCTION
# =====================================================

def fitness(chromosome):

    makespan, waiting, _ = simulate_schedule(
        chromosome,
        policy="GEP"
    )

    return makespan + 0.3 * waiting


# =====================================================
# 7. GEP GENETIC OPERATORS
# =====================================================

def selection(population, fitness_values):

    selected_indices = random.sample(
        range(len(population)),
        3
    )

    best_index = min(
        selected_indices,
        key=lambda i: fitness_values[i]
    )

    return copy.deepcopy(population[best_index])


def crossover(parent1, parent2):

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    point = random.randint(0, GENE_LENGTH - 1)

    child1[point:], child2[point:] = (
        child2[point:],
        child1[point:]
    )

    return child1, child2


def mutation(chromosome, rate=0.08):

    child = copy.deepcopy(chromosome)

    for i in range(GENE_LENGTH):

        if random.random() < rate:

            if i < HEAD_LENGTH:

                child[i] = random.choice(
                    FUNCTIONS + TERMINALS
                )

            else:

                child[i] = random.choice(TERMINALS)

    return child


# =====================================================
# 8. GEP OPTIMIZATION
# =====================================================

population = [
    generate_chromosome()
    for _ in range(POPULATION_SIZE)
]

best_chromosome = None
best_fitness = float("inf")

best_history = []

print("\n========== GEP OPTIMIZATION ==========")

for generation in range(GENERATIONS):

    fitness_values = [
        fitness(chromosome)
        for chromosome in population
    ]

    best_index = np.argmin(fitness_values)

    if fitness_values[best_index] < best_fitness:

        best_fitness = fitness_values[best_index]

        best_chromosome = copy.deepcopy(
            population[best_index]
        )

    best_history.append(best_fitness)

    new_population = [
        copy.deepcopy(best_chromosome)
    ]

    while len(new_population) < POPULATION_SIZE:

        parent1 = selection(
            population,
            fitness_values
        )

        parent2 = selection(
            population,
            fitness_values
        )

        child1, child2 = crossover(
            parent1,
            parent2
        )

        child1 = mutation(child1)
        child2 = mutation(child2)

        new_population.append(child1)

        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    population = new_population

    if (generation + 1) % 5 == 0:

        print(
            f"Generation {generation + 1}: "
            f"Best Fitness = {best_fitness:.2f}"
        )


# =====================================================
# 9. FINAL GEP AND FIFO SCHEDULES
# =====================================================

gep_makespan, gep_waiting, gep_schedule = simulate_schedule(
    best_chromosome,
    policy="GEP"
)

fifo_makespan, fifo_waiting, fifo_schedule = simulate_schedule(
    policy="FIFO"
)


# =====================================================
# 10. RESULTS
# =====================================================

print("\n========== FINAL RESULTS ==========")

print("\nGEP RESULTS")

print("Makespan:", gep_makespan)
print("Total Waiting Time:", round(gep_waiting, 2))
print("Best Fitness:", round(best_fitness, 2))

print("\nFIFO RESULTS")

print("Makespan:", fifo_makespan)
print("Total Waiting Time:", round(fifo_waiting, 2))

if fifo_makespan > 0:

    improvement = (
        (fifo_makespan - gep_makespan)
        / fifo_makespan
    ) * 100

    print(
        "\nMakespan Improvement:",
        round(improvement, 2),
        "%"
    )


# =====================================================
# 11. MACHINE UTILIZATION
# =====================================================

def calculate_utilization(schedule, makespan):

    utilization = []

    for machine in range(NUM_MACHINES):

        busy_time = sum(
            item["duration"]
            for item in schedule
            if item["machine"] == machine
        )

        percentage = (
            busy_time / makespan * 100
            if makespan > 0 else 0
        )

        utilization.append(percentage)

    return utilization


gep_utilization = calculate_utilization(
    gep_schedule,
    gep_makespan
)

fifo_utilization = calculate_utilization(
    fifo_schedule,
    fifo_makespan
)

print("\n========== MACHINE UTILIZATION ==========")

for i in range(NUM_MACHINES):

    print(
        f"Machine {i}: "
        f"GEP = {gep_utilization[i]:.2f}% | "
        f"FIFO = {fifo_utilization[i]:.2f}%"
    )


# =====================================================
# 12. CONVERGENCE GRAPH
# =====================================================

plt.figure(figsize=(8, 4))

plt.plot(
    range(1, GENERATIONS + 1),
    best_history,
    marker="o"
)

plt.xlabel("Generation")
plt.ylabel("Best Fitness")

plt.title("GEP Optimization Convergence")

plt.grid(True)
plt.tight_layout()
plt.show()


# =====================================================
# 13. GANTT CHART
# =====================================================

def plot_gantt(schedule, title, makespan):

    fig, ax = plt.subplots(figsize=(12, 5))

    for item in schedule:

        machine = item["machine"]

        ax.barh(
            machine,
            item["duration"],
            left=item["start"],
            edgecolor="black"
        )

        ax.text(
            item["start"] + item["duration"] / 2,
            machine,
            f"J{item['job']}",
            ha="center",
            va="center",
            fontsize=8
        )

    ax.set_yticks(range(NUM_MACHINES))

    ax.set_yticklabels([
        f"Machine {i}"
        for i in range(NUM_MACHINES)
    ])

    ax.set_xlim(0, makespan + 2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")

    ax.set_title(title)

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()
    plt.show()


plot_gantt(
    gep_schedule,
    "GEP Dynamic Job Shop Scheduling",
    gep_makespan
)

plot_gantt(
    fifo_schedule,
    "FIFO Job Shop Scheduling",
    fifo_makespan
)


# =====================================================
# 14. PERFORMANCE COMPARISON
# =====================================================

labels = ["Makespan", "Waiting Time"]

gep_values = [
    gep_makespan,
    gep_waiting
]

fifo_values = [
    fifo_makespan,
    fifo_waiting
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 4))

plt.bar(
    x - width / 2,
    gep_values,
    width,
    label="GEP"
)

plt.bar(
    x + width / 2,
    fifo_values,
    width,
    label="FIFO"
)

plt.xticks(x, labels)

plt.ylabel("Time")
plt.title("GEP vs FIFO Performance")

plt.legend()
plt.tight_layout()
plt.show()

print("\n========== EXECUTION COMPLETED ==========")
