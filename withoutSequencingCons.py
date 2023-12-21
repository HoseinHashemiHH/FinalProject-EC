import random, numpy as np
from OPPSModel import generate_initial_chromosome,m,n, block_value, sequencingConstraints,wn,searchSpace
def block_value(x:np.array)->np.array:
    # print(x.shape)
    o=0
    c=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if x[i][j]==1:
                c[i][j]=80.8
                o+=1
            elif x[i][j]==2:
                c[i][j]=24.26
                o+=1
            elif x[i][j]==3:
                c[i][j]=13.14
                o+=1
            elif x[i][j]==4:
                c[i][j]=15.84
                o+=1
            elif x[i][j]==5:
                c[i][j]=6.54
                o+=1
            else:
                c[i][j]=0.0
    return c
def create_individual(search_space):
    return [random.choice(search_space) for _ in range(m * n)]

def evaluate_objective(x, c):
    # You need to define your own objective function based on your problem
    # For now, let's use a simple sum of block values as the objective
    objective_value = np.sum(np.multiply(c,x))
    return objective_value

def genetic_programming(population_size, generations, wn):
    population = [create_individual(block_value(searchSpace)) for _ in range(population_size)]
    k=0
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        # Evaluate fitness for each individual
        fitness_values = []
        for individual in population:
            # print(individual.shape)
            x = np.array(individual).reshape(10,50)
            # .reshape((m, n))
            c = block_value(x)
            # print(x.shape)
            
            # # Apply constraints
            if True: #np.sum(np.matmul(x, wn)) >= 6: #and sequencingConstraints(x):np.sum(np.matmul(x, wn)) <= 10 and 
                fitness = evaluate_objective(x, c)
            # else:
            #     fitness = float('-inf')
                
            fitness_values.append(fitness)

        # Select the best individual
        if np.max(fitness_values) >= best_fitness:
            best_fitness = np.max(fitness_values)
            best_individual = population[np.argmax(fitness_values)]

        # if fitness_values[int(max_fitness_index)] >= best_fitness:
        #     best_fitness = fitness_values[max_fitness_index]
        #     best_individual = population[max_fitness_index]
        #     if fitness_values[max_fitness_index]=='None':
        #         best_fitness=0

        # Display the first and best individuals in each generation
        if generation == 0 or generation == generations - 1:
            print(f"Generation {generation + 1} - First Individual: {population[0]}, Best Individual: {best_individual}, Best Fitness: {best_fitness}")

        # Create a new generation using tournament selection and crossover
        new_population = []
        for _ in range(population_size):
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            selected_index = tournament_indices[np.argmax(tournament_fitness)]
            new_population.append(population[selected_index])

        # Crossover (you might need to customize this part based on your specific problem)
        crossover_point = random.randint(1, m * n - 1)
        for i in range(0, population_size, 2):
            parent1 = new_population[i]
            parent2 = new_population[i + 1]
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            new_population[i] = child1
            new_population[i + 1] = child2

        population = new_population

    return best_individual, best_fitness

# Usage
population_size = 400  # Adjust as needed
generations = 400  # Adjust as needed
best_individual, best_fitness = genetic_programming(population_size, generations, wn)

print("\nFinal Result:")
print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)
